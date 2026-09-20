import logging
from dataclasses import dataclass

from module.conf import settings
from module.database import Database
from module.downloader import AddResult, DownloadClient
from module.downloader.path import gen_save_path
from module.models import Bangumi, ResponseModel
from module.network import RequestContent
from module.parser import TitleParser
from module.rss import RSSEngine
from module.searcher import SearchTorrent

logger = logging.getLogger(__name__)


async def _ensure_bangumi_id(db: Database, data: Bangumi) -> bool:
    """确保 ``data.id`` 可用：新番剧插入拿 id，重复番剧解析出已存在行的 id。

    ``add()`` 对精确重复（title_raw+group_name）与语义重复（并入别名）都
    返回 False 且不回填 id——不解析已存在行的话，add_torrent 的 ab:<id>
    标签与种子行的 bangumi_id 关联都会丢失，种子被记成孤儿。

    解析到被软删除（禁用）的行时会重新启用它：显式订阅/收集一个已禁用
    的规则只能是用户想要它回来——不启用的话种子要么变孤儿、要么挂到
    对下游所有查询都不可见的行上。传入的 id 指向已不存在的行时清空，
    避免种子挂到悬空外键。返回是否插入了新行（调用方据此决定失败时
    是否回滚删除）。
    """
    if await db.bangumi.add(data):
        return True
    existing = await db.bangumi.find_duplicate(data)
    if existing is None:
        existing = await db.bangumi.find_semantic_duplicate(data)
    if existing is not None:
        if existing.deleted:
            await db.bangumi.restore_one(existing.id)
        data.id = existing.id
    elif data.id is not None and await db.bangumi.search_id(data.id) is None:
        data.id = None  # type: ignore[assignment]
    return False


async def resolve_search_metadata(data: Bangumi, parser: str) -> Bangumi:
    """Run the same TMDB/TVDB cross-reference subscribe_season persists, but
    without saving anything -- lets the search UI preview resolved metadata
    (year, poster, tvdb_id) once a specific torrent/source is picked, before
    the user commits to Subscribe. Mirrors what Add RSS -> Analyze already
    shows for that flow. A no-op if `data` is already resolved.

    parser=="mikan" gets year/tvdb_id/id_source filled in too (mikan's own
    homepage scrape never provides those), but keeps its own
    official_title/season -- Mikan's scrape is more reliable for those than
    a TMDB text search, and overwriting season with TMDB's own guess is a
    plausible cause of separate season-mislabeling reports.
    """
    if parser in ("tmdb", "mikan") and data.tvdb_id is None:
        try:
            (
                official_title,
                season,
                year,
                poster_link,
                meta_id,
                id_source,
            ) = await TitleParser.tmdb_parser(
                data.official_title,
                data.season,
                settings.rss_parser.language,
                episode_type=data.episode_type,
            )
        except Exception as e:
            logger.warning(f"TMDB cross-reference failed for search result: {e}")
        else:
            if parser == "tmdb":
                data.official_title = official_title
                data.season = season
                if poster_link:
                    data.poster_link = poster_link
            data.year = year
            data.tvdb_id = meta_id
            data.id_source = id_source
    return data


@dataclass(frozen=True, slots=True)
class ReparseResult:
    """What changed when re-running the TMDB/TVDB parser on an existing
    bangumi -- surfaced to the UI so a "fix the folder" action explains
    itself instead of silently moving files."""

    old_official_title: str
    new_official_title: str
    old_year: str | None
    new_year: str | None
    old_tvdb_id: int | None
    new_tvdb_id: int | None
    old_id_source: str | None
    new_id_source: str | None
    old_folder: str
    new_folder: str
    folder_changed: bool
    torrents_found: int
    torrents_moved: int
    torrents_failed: int


async def reparse_bangumi(
    db: Database, client: DownloadClient, bangumi_id: int
) -> ReparseResult | None:
    """Re-run the TMDB/TVDB parser for an existing bangumi (e.g. one
    subscribed before the search-subscribe TMDB fix, or added with only a
    bare title) and move its already-downloaded torrents into the corrected
    folder.

    Deliberately does not touch `season` -- same reasoning as
    resolve_search_metadata's mikan branch: overwriting the season a
    release was actually organized under with TMDB's own guess at the
    "current" season is a plausible cause of season-mislabeling, and this
    action's whole point is to fix organization, not risk repeating that
    bug. Deliberately does not rename individual files either -- only which
    folder they live in; per-file names are the periodic rename loop's job
    (or a manual cleanup pass for files renamed before a naming fix).
    """
    bangumi = await db.bangumi.search_id(bangumi_id)
    if bangumi is None:
        return None

    old_folder = gen_save_path(bangumi)
    old_official_title = bangumi.official_title
    old_year = bangumi.year
    old_tvdb_id = bangumi.tvdb_id
    old_id_source = bangumi.id_source

    (
        official_title,
        _season,
        year,
        poster_link,
        meta_id,
        id_source,
    ) = await TitleParser.tmdb_parser(
        bangumi.official_title,
        bangumi.season,
        settings.rss_parser.language,
        episode_type=bangumi.episode_type,
    )
    bangumi.official_title = official_title
    bangumi.year = year
    if poster_link:
        bangumi.poster_link = poster_link
    bangumi.tvdb_id = meta_id
    bangumi.id_source = id_source

    new_folder = gen_save_path(bangumi)
    bangumi.save_path = new_folder
    await db.bangumi.update(bangumi)

    folder_changed = old_folder != new_folder
    hashes: set[str] = set()
    torrents_moved = 0
    torrents_failed = 0
    if folder_changed:
        hashes = await _find_torrent_hashes(db, client, bangumi_id)
        for torrent_hash in hashes:
            try:
                await client.move_torrent(torrent_hash, new_folder)
                torrents_moved += 1
            except Exception as e:
                logger.warning(
                    f"Failed to move torrent {torrent_hash} during "
                    f"reparse of bangumi {bangumi_id}: {e}"
                )
                torrents_failed += 1

    return ReparseResult(
        old_official_title=old_official_title,
        new_official_title=official_title,
        old_year=old_year,
        new_year=year,
        old_tvdb_id=old_tvdb_id,
        new_tvdb_id=meta_id,
        old_id_source=old_id_source,
        new_id_source=id_source,
        old_folder=old_folder,
        new_folder=new_folder,
        folder_changed=folder_changed,
        torrents_found=len(hashes),
        torrents_moved=torrents_moved,
        torrents_failed=torrents_failed,
    )


async def _find_torrent_hashes(
    db: Database, client: DownloadClient, bangumi_id: int
) -> set[str]:
    """Every torrent hash associated with a bangumi, from both sources this
    codebase uses for that link (see renamer.py's _lookup_offsets, which
    faces the identical problem): the Torrent table's bangumi_id FK, and the
    downloader's own "ab:<id>" tag. A bangumi whose folder needs reparsing/
    fixing is exactly the kind whose DB bookkeeping may be inconsistent --
    relying on the FK alone silently finds nothing to move for such a
    bangumi even though the downloader still knows exactly which torrents
    are its via the tag.
    """
    hashes = {
        t.qb_hash
        for t in await db.torrent.search_by_bangumi_id(bangumi_id)
        if t.qb_hash
    }
    tagged = await client.get_torrent_info(
        category=None, status_filter=None, tag=f"ab:{bangumi_id}"
    )
    hashes.update(info["hash"] for info in tagged if info.get("hash"))
    return hashes


class SeasonCollector:
    def __init__(self, client: DownloadClient):
        self.client = client

    async def collect_season(self, bangumi: Bangumi, link: str | None = None):
        logger.info(
            f"Start collecting {bangumi.official_title} Season {bangumi.season}..."
        )
        st = SearchTorrent()
        if not link:
            torrents = await st.search_season(bangumi)
        else:
            async with RequestContent() as req:
                torrents = await req.get_torrents(
                    link, bangumi.filter.replace(",", "|")
                )
        async with Database() as db:
            # bangumi 必须先落库拿到 id：add_torrent 用它打 ab:<id> 标签，
            # 种子行也要用它关联 bangumi_id——否则种子会被记成孤儿，
            # track_orphans 开关对这些"已匹配"的种子完全失效。
            # update() returns False when no existing row matches
            # bangumi.id (i.e. this is a brand-new bangumi), in which
            # case it needs to be inserted (or resolved to the existing
            # duplicate row) instead.
            inserted = False
            if bangumi.id is None or not await db.bangumi.update(bangumi):
                inserted = await _ensure_bangumi_id(db, bangumi)
            else:
                # 载荷带着已禁用行的 id 也能 update 成功：显式收集
                # 即用户想重新启用，否则种子会挂到不可见的行上
                await db.bangumi.restore_one(bangumi.id)
            if await self.client.add_torrent(torrents, bangumi) is AddResult.ADDED:
                logger.info(
                    f"Collections of {bangumi.official_title} Season {bangumi.season} completed."
                )
                for torrent in torrents:
                    torrent.downloaded = True
                    torrent.bangumi_id = bangumi.id
                bangumi.eps_collect = True
                # 只更新 eps_collect 单个字段：解析到已存在行时，整行
                # update 会用刚解析的默认值覆盖用户调好的 offset/filter
                if bangumi.id is not None:
                    await db.bangumi.mark_eps_collect(bangumi.id)
                await db.torrent.add_all(torrents)
                return ResponseModel(
                    status=True,
                    status_code=200,
                    msg_en=f"Collections of {bangumi.official_title} Season {bangumi.season} completed.",
                    msg_zh=f"收集 {bangumi.official_title} 第 {bangumi.season} 季完成。",
                )
            else:
                if inserted and bangumi.id is not None:
                    # 收集失败时回滚刚插入的行，不留下幽灵订阅规则
                    await db.bangumi.delete_one(bangumi.id)
                    bangumi.id = None  # type: ignore[assignment]
                logger.warning(
                    f"Already collected {bangumi.official_title} Season {bangumi.season}."
                )
                return ResponseModel(
                    status=False,
                    status_code=406,
                    msg_en=f"Collection of {bangumi.official_title} Season {bangumi.season} failed.",
                    msg_zh=f"收集 {bangumi.official_title} 第 {bangumi.season} 季失败, 种子已经添加。",
                )

    @staticmethod
    async def subscribe_season(data: Bangumi, parser: str = "mikan"):
        # Search results are built with fetch_poster=False for listing
        # responsiveness (searcher.py/analyse_keyword), which skips the
        # entire TMDB/TVDB resolution branch in official_title_parser --
        # only official_title/poster_link get a lightweight patch
        # afterward. subscribe_season persists `data` as given with no
        # re-parsing, so a search-sourced subscribe would otherwise save
        # a bangumi with no year/tvdb_id/id_source forever. Resolve it
        # here, once, before it's ever written -- harmless no-op for a
        # bangumi that already resolved (e.g. via the Add-RSS analysis
        # flow, which does call the full parser, or the /rss/resolve
        # preview the search UI calls before the user even gets here).
        data = await resolve_search_metadata(data, parser)

        async with Database() as db:
            engine = RSSEngine(db)
            data.added = True
            data.eps_collect = True
            await engine.add_rss(
                rss_link=data.rss_link,
                name=data.official_title,
                aggregate=False,
                parser=parser,
            )
            # 先落库拿到 id（重复订阅时解析已存在行的 id）：download_bangumi
            # 里 add_torrent 的 ab:<id> 标签和种子行的 bangumi_id 关联都依赖它
            await _ensure_bangumi_id(db, data)
            return await engine.download_bangumi(data)


async def eps_complete():
    async with Database() as db:
        datas = await db.bangumi.not_complete()
        if datas:
            logger.info("Start collecting full season...")
            async with DownloadClient() as client:
                collector = SeasonCollector(client)
                for data in datas:
                    if not data.eps_collect:
                        await collector.collect_season(data)
                    data.eps_collect = True
            await db.bangumi.update_all(datas)
