"""One-off import of a torrent qBittorrent already has (added outside
AutoBangumi's own subscriptions) into the standard library layout.

Deliberately does not create a Bangumi row or any tracked subscription --
this is purely "rename+move these files the same way everything else in the
library is named", for a release you obtained by some other means (manual
search hit, a batch DMHY/Nyaa/Mikan release the normal RSS filter excluded,
etc.). See rss/analyser.py + downloader/path.py for the automated equivalent
this mirrors.
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from module.conf import settings
from module.downloader import AddResult, DownloadClient
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


async def _resolve_title_and_target_folder(
    official_title: str, season: int
) -> tuple[str, str | None, int | None, str | None, str]:
    """Shared by both preview paths: TMDB/TVDB lookup + the same save-path
    logic an automated subscription uses. Returns (resolved_title, year,
    tvdb_id, id_source, target_folder)."""
    language = settings.rss_parser.language
    (
        resolved_title,
        _resolved_season,
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
    return resolved_title, year, meta_id, id_source, target_folder


async def upload_torrent(
    client: DownloadClient, torrent_bytes: bytes
) -> ImportCandidate | None:
    """Add an uploaded .torrent file directly (untagged, same as adding it
    through the downloader's own UI), then look up the resulting candidate
    so the caller can proceed straight into preview_import()/apply_import()
    -- no new preview/apply path needed, this only covers getting the file
    into the downloader in the first place.

    A short bounded poll covers the brief window between adding a torrent
    with full metadata already attached (no magnet resolution needed) and it
    appearing in the downloader's own listing.
    """
    result, info_hash = await client.add_torrent_file(torrent_bytes)
    if result is AddResult.FAILED or info_hash is None:
        return None

    for attempt in range(5):
        infos = await client.get_torrent_info(
            category=None, status_filter=None, tag=None
        )
        info = next(
            (i for i in infos if i.get("hash", "").lower() == info_hash.lower()), None
        )
        if info is not None:
            return ImportCandidate(
                hash=info["hash"],
                name=info.get("name", ""),
                save_path=info.get("save_path", ""),
                category=info.get("category", ""),
                size=info.get("size", 0),
                progress=info.get("progress", 0.0),
                state=info.get("state", ""),
            )
        if attempt < 4:
            await asyncio.sleep(0.3)

    logger.warning(
        "Uploaded torrent %s was accepted but never appeared in the "
        "downloader's listing",
        info_hash,
    )
    return None


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

    (
        resolved_title,
        year,
        meta_id,
        id_source,
        target_folder,
    ) = await _resolve_title_and_target_folder(official_title, season)
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


@dataclass(frozen=True, slots=True)
class FolderFileMapping:
    """One file's proposed rename+move, as absolute filesystem paths (unlike
    FileMapping, which is relative to a torrent's own content root and only
    meaningful to the downloader's own rename/move API)."""

    source_path: str
    target_path: str
    episode: int | float | None
    parsed: bool
    kind: str  # "media" | "subtitle"


@dataclass(frozen=True, slots=True)
class FolderImportPreview:
    source_root: str
    official_title: str
    year: str | None
    tvdb_id: int | None
    id_source: str | None
    target_folder: str
    mappings: list[FolderFileMapping] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


def _scan_media_and_subtitles(root: Path) -> tuple[list[str], list[str]]:
    """Recursively find media/subtitle files under root, as paths relative
    to root -- same shape check_files()/torrent_parser() expect for a
    torrent's own content listing, so this can reuse both unchanged."""
    files = [
        {"name": str(path.relative_to(root))}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return check_files(files)


async def preview_folder_import(
    root_path: str,
    official_title: str,
    season: int,
) -> FolderImportPreview | None:
    """Same TMDB/TVDB lookup and naming convention as preview_import(), but
    for a plain filesystem folder that was never added through the
    downloader at all (dropped in by hand, synced from elsewhere) -- scans
    recursively rather than asking a download client for a torrent's file
    list."""
    root = Path(root_path)
    if not root.is_dir():
        logger.warning("Folder import preview: %s is not a directory", root_path)
        return None

    media_list, subtitle_list = _scan_media_and_subtitles(root)

    (
        resolved_title,
        year,
        meta_id,
        id_source,
        target_folder,
    ) = await _resolve_title_and_target_folder(official_title, season)
    method = settings.bangumi_manage.rename_method
    target_folder_path = Path(target_folder)

    mappings: list[FolderFileMapping] = []
    unparsed: list[str] = []

    for rel_path in media_list:
        episode_file = TitleParser.torrent_parser(
            torrent_path=rel_path,
            torrent_name=root.name,
            season=season,
            file_type="media",
        )
        if not isinstance(episode_file, EpisodeFile):
            unparsed.append(rel_path)
            continue
        target_name = Renamer.gen_path(episode_file, resolved_title, method)
        mappings.append(
            FolderFileMapping(
                source_path=str(root / rel_path),
                target_path=str(target_folder_path / target_name),
                episode=episode_file.episode,
                parsed=True,
                kind="media",
            )
        )

    subtitle_method = "subtitle_" + method
    for rel_path in subtitle_list:
        subtitle_file = TitleParser.torrent_parser(
            torrent_path=rel_path,
            torrent_name=root.name,
            season=season,
            file_type="subtitle",
        )
        if not isinstance(subtitle_file, SubtitleFile):
            unparsed.append(rel_path)
            continue
        target_name = Renamer.gen_path(subtitle_file, resolved_title, subtitle_method)
        mappings.append(
            FolderFileMapping(
                source_path=str(root / rel_path),
                target_path=str(target_folder_path / target_name),
                episode=subtitle_file.episode,
                parsed=True,
                kind="subtitle",
            )
        )

    return FolderImportPreview(
        source_root=str(root),
        official_title=resolved_title,
        year=year,
        tvdb_id=meta_id,
        id_source=id_source,
        target_folder=target_folder,
        mappings=mappings,
        unparsed=unparsed,
    )


async def apply_folder_import(
    mappings: list[FolderFileMapping],
) -> list[tuple[FolderFileMapping, bool, str | None]]:
    """Move+rename each mapped file directly on disk -- no download client
    involved, since these files were never known to one. Never overwrites
    an existing destination, and never deletes/cleans up the source folder
    (any subfolders it had are left as-is, just emptied of moved files)."""
    results: list[tuple[FolderFileMapping, bool, str | None]] = []
    for mapping in mappings:
        source = Path(mapping.source_path)
        target = Path(mapping.target_path)
        if source == target:
            results.append((mapping, True, None))
            continue
        if not source.is_file():
            results.append((mapping, False, "source file not found"))
            continue
        if target.exists():
            results.append((mapping, False, "destination already exists"))
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        except OSError as e:
            results.append((mapping, False, str(e)))
        else:
            results.append((mapping, True, None))
    return results
