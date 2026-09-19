"""One-off import of a torrent qBittorrent already has (added outside
AutoBangumi's own subscriptions) into the standard library layout.

Deliberately does not create a Bangumi row or any tracked subscription --
this is purely "rename+move these files the same way everything else in the
library is named", for a release you obtained by some other means (manual
search hit, a batch DMHY/Nyaa/Mikan release the normal RSS filter excluded,
etc.). See rss/analyser.py + downloader/path.py for the automated equivalent
this mirrors.
"""

import logging
from dataclasses import dataclass, field

from module.conf import settings
from module.downloader import DownloadClient
from module.downloader.path import check_files, gen_save_path
from module.models import BangumiUpdate
from module.models.torrent import EpisodeFile, SubtitleFile
from module.parser import TitleParser

from .renamer import Renamer

logger = logging.getLogger(__name__)

_MANAGED_TAG_PREFIX = "ab:"


@dataclass(frozen=True, slots=True)
class ImportCandidate:
    """A qBittorrent torrent not already tracked by any AutoBangumi tag."""

    hash: str
    name: str
    save_path: str
    category: str
    size: int
    progress: float
    state: str


@dataclass(frozen=True, slots=True)
class FileMapping:
    """One file's proposed rename+move, relative to the torrent's own content."""

    source_path: str
    target_path: str
    episode: int | float | None
    parsed: bool
    kind: str  # "media" | "subtitle"


@dataclass(frozen=True, slots=True)
class ImportPreview:
    torrent_hash: str
    official_title: str
    year: str | None
    tvdb_id: int | None
    id_source: str | None
    target_folder: str
    mappings: list[FileMapping] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


def _is_managed(tags: str | None) -> bool:
    return any(
        t.strip().startswith(_MANAGED_TAG_PREFIX) for t in (tags or "").split(",")
    )


async def list_import_candidates(client: DownloadClient) -> list[ImportCandidate]:
    """Torrents qBittorrent knows about that AutoBangumi never tagged --
    i.e. added directly through qBittorrent's own UI/API, not via a
    subscription or Search/Collect."""
    infos = await client.get_torrent_info(category=None, status_filter=None, tag=None)
    return [
        ImportCandidate(
            hash=info["hash"],
            name=info.get("name", ""),
            save_path=info.get("save_path", ""),
            category=info.get("category", ""),
            size=info.get("size", 0),
            progress=info.get("progress", 0.0),
            state=info.get("state", ""),
        )
        for info in infos
        if not _is_managed(info.get("tags"))
    ]


async def preview_import(
    client: DownloadClient,
    torrent_hash: str,
    official_title: str,
    season: int,
) -> ImportPreview | None:
    """Resolve the destination folder (via the same TMDB/TVDB lookup and
    save-path logic as an automated subscription) and propose a per-file
    rename+move, without touching anything yet."""
    infos = await client.get_torrent_info(category=None, status_filter=None, tag=None)
    info = next((i for i in infos if i["hash"] == torrent_hash), None)
    if info is None:
        logger.warning("Import preview: torrent %s not found", torrent_hash)
        return None

    files = await client.get_torrent_files(torrent_hash)
    media_list, subtitle_list = check_files(files)

    language = settings.rss_parser.language
    (
        resolved_title,
        resolved_season,
        year,
        _poster_link,
        meta_id,
        id_source,
    ) = await TitleParser.tmdb_parser(official_title, season, language)

    bangumi = BangumiUpdate(
        official_title=resolved_title,
        year=year,
        season=season,  # the user's season, not TMDB's -- they picked the folder
        tvdb_id=meta_id,
        id_source=id_source,
        # BangumiUpdate declares these Optional[...] but Pydantic v2 doesn't
        # treat that as an implicit default -- gen_save_path only reads
        # official_title/year/season/season_offset/episode_type/tvdb_id/
        # id_source, so the rest are irrelevant filler for this one-off,
        # unpersisted object.
        season_raw=None,
        group_name=None,
        dpi=None,
        source=None,
        subtitle=None,
        poster_link=None,
        rule_name=None,
        save_path=None,
    )
    target_folder = gen_save_path(bangumi)
    method = settings.bangumi_manage.rename_method

    mappings: list[FileMapping] = []
    unparsed: list[str] = []

    for media_path in media_list:
        episode_file = TitleParser.torrent_parser(
            torrent_path=media_path,
            torrent_name=info.get("name"),
            season=season,
            file_type="media",
        )
        if not isinstance(episode_file, EpisodeFile):
            unparsed.append(media_path)
            continue
        target_name = Renamer.gen_path(episode_file, resolved_title, method)
        mappings.append(
            FileMapping(
                source_path=media_path,
                target_path=target_name,
                episode=episode_file.episode,
                parsed=True,
                kind="media",
            )
        )

    subtitle_method = "subtitle_" + method
    for subtitle_path in subtitle_list:
        subtitle_file = TitleParser.torrent_parser(
            torrent_path=subtitle_path,
            torrent_name=info.get("name"),
            season=season,
            file_type="subtitle",
        )
        if not isinstance(subtitle_file, SubtitleFile):
            unparsed.append(subtitle_path)
            continue
        target_name = Renamer.gen_path(subtitle_file, resolved_title, subtitle_method)
        mappings.append(
            FileMapping(
                source_path=subtitle_path,
                target_path=target_name,
                episode=subtitle_file.episode,
                parsed=True,
                kind="subtitle",
            )
        )

    return ImportPreview(
        torrent_hash=torrent_hash,
        official_title=resolved_title,
        year=year,
        tvdb_id=meta_id,
        id_source=id_source,
        target_folder=target_folder,
        mappings=mappings,
        unparsed=unparsed,
    )


async def apply_import(
    client: DownloadClient,
    torrent_hash: str,
    target_folder: str,
    mappings: list[FileMapping],
) -> list[tuple[FileMapping, bool, str | None]]:
    """Move the torrent's content to target_folder, then rename each mapped
    file within it. Returns (mapping, succeeded, detail) per file so the
    caller can report partial failures instead of an all-or-nothing result.
    """
    await client.move_torrent(torrent_hash, target_folder)

    results: list[tuple[FileMapping, bool, str | None]] = []
    for mapping in mappings:
        if mapping.source_path == mapping.target_path:
            results.append((mapping, True, None))
            continue
        result = await client.rename_torrent_file(
            _hash=torrent_hash,
            old_path=mapping.source_path,
            new_path=mapping.target_path,
        )
        results.append((mapping, result.succeeded, result.detail))
    return results
