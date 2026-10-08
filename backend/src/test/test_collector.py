"""Tests for SeasonCollector: torrents must be persisted with bangumi_id.

Regression tests for the orphan-torrent bug: torrents added via
subscribe/collect were stored with ``bangumi_id=None`` and showed up as
orphans regardless of the ``track_orphans`` setting.
"""

import stat
from unittest.mock import AsyncMock, patch

from module.database import Database
from module.downloader import AddResult
from module.downloader.path import gen_save_path
from module.manager.collector import SeasonCollector, reparse_bangumi
from module.models import Torrent
from test.factories import make_bangumi


def _async_ctx(inner):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=inner)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _req_with_torrents(torrents):
    req = AsyncMock()
    req.get_torrents = AsyncMock(return_value=torrents)
    return _async_ctx(req)


def _make_torrents():
    return [
        Torrent(
            name="[TestGroup] Test Anime Raw - 01 [1080p].mkv",
            url="https://example.com/ep1.torrent",
        ),
        Torrent(
            name="[TestGroup] Test Anime Raw - 02 [1080p].mkv",
            url="https://example.com/ep2.torrent",
        ),
    ]


class TestCollectSeason:
    async def test_torrents_persisted_with_bangumi_id(self):
        bangumi = make_bangumi(filter="")
        torrents = _make_torrents()
        client = AsyncMock()
        client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        collector = SeasonCollector(client)

        with patch(
            "module.manager.collector.RequestContent",
            return_value=_req_with_torrents(torrents),
        ):
            resp = await collector.collect_season(bangumi, "https://example.com/rss")

        assert resp.status is True
        async with Database() as db:
            stored = await db.torrent.search_all()
            assert len(stored) == 2
            assert all(t.bangumi_id == bangumi.id for t in stored)
            assert await db.torrent.count_orphans() == 0

    async def test_duplicate_resolution_preserves_existing_row_fields(self):
        """解析到已存在行时不能用刚解析的默认值覆盖用户调好的配置。"""
        existing = make_bangumi(filter="", episode_offset=12)
        async with Database() as db:
            assert await db.bangumi.add(existing)

        payload = make_bangumi(filter="", episode_offset=0)  # 同 key，默认 offset
        client = AsyncMock()
        client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        collector = SeasonCollector(client)
        with patch(
            "module.manager.collector.RequestContent",
            return_value=_req_with_torrents(_make_torrents()),
        ):
            resp = await collector.collect_season(payload, "https://example.com/rss")

        assert resp.status is True
        async with Database() as db:
            row = await db.bangumi.search_id(existing.id)
            assert row is not None
            assert row.episode_offset == 12
            assert row.eps_collect is True

    async def test_failed_collect_leaves_no_phantom_bangumi(self):
        """下载器投递失败时不能留下幽灵订阅规则。"""
        bangumi = make_bangumi(filter="")
        client = AsyncMock()
        client.add_torrent = AsyncMock(return_value=AddResult.FAILED)
        collector = SeasonCollector(client)
        with patch(
            "module.manager.collector.RequestContent",
            return_value=_req_with_torrents(_make_torrents()),
        ):
            resp = await collector.collect_season(bangumi, "https://example.com/rss")

        assert resp.status is False
        async with Database() as db:
            assert await db.bangumi.search_all() == []

    async def test_add_torrent_called_with_persisted_bangumi(self):
        """add_torrent 打 ab:<id> 标签依赖调用时 bangumi.id 已存在。"""
        bangumi = make_bangumi(filter="")
        client = AsyncMock()
        client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        collector = SeasonCollector(client)

        with patch(
            "module.manager.collector.RequestContent",
            return_value=_req_with_torrents(_make_torrents()),
        ):
            await collector.collect_season(bangumi, "https://example.com/rss")

        passed_bangumi = client.add_torrent.call_args.args[1]
        assert passed_bangumi.id is not None


class TestSubscribeSeason:
    async def test_existing_bangumi_links_torrents_to_existing_id(self):
        """重复订阅已存在的番剧（add 返回 False）时，种子必须挂到
        已存在行的 id 上，不能被记成孤儿。"""
        existing = make_bangumi(filter="")
        async with Database() as db:
            assert await db.bangumi.add(existing)

        data = make_bangumi(filter="")  # 同 title_raw+group_name -> 精确重复
        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(_make_torrents()),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
        ):
            result = await SeasonCollector.subscribe_season(data)

        assert result.status is True
        async with Database() as db:
            stored = await db.torrent.search_all()
            assert len(stored) == 2
            assert all(t.bangumi_id == existing.id for t in stored)
            assert await db.torrent.count_orphans() == 0

    async def test_deleted_duplicate_is_restored_and_linked(self):
        """重新订阅已禁用的规则时重新启用该行并把种子挂上去——否则种子
        要么变孤儿、要么挂到对下游查询不可见的行上。"""
        existing = make_bangumi(filter="", deleted=True)
        async with Database() as db:
            assert await db.bangumi.add(existing)

        data = make_bangumi(filter="")
        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(_make_torrents()),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
        ):
            await SeasonCollector.subscribe_season(data)

        async with Database() as db:
            row = await db.bangumi.search_id(existing.id)
            assert row is not None
            assert row.deleted is False
            stored = await db.torrent.search_all()
            assert stored and all(t.bangumi_id == existing.id for t in stored)

    async def test_torrents_persisted_with_bangumi_id(self):
        data = make_bangumi(filter="")
        torrents = _make_torrents()

        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(torrents),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
        ):
            result = await SeasonCollector.subscribe_season(data)

        assert result.status is True
        assert data.id is not None
        async with Database() as db:
            stored = await db.torrent.search_all()
            assert len(stored) == 2
            assert all(t.bangumi_id == data.id for t in stored)
            assert await db.torrent.count_orphans() == 0


class TestSubscribeSeasonTmdbResolution:
    """Regression tests: subscribing directly from a Search result (parser=
    "tmdb") previously saved a bangumi with no year/tvdb_id/id_source
    forever, because searcher.py builds search results with
    fetch_poster=False (skipping the entire TMDB/TVDB branch in
    official_title_parser for listing responsiveness) and subscribe_season
    persisted whatever it was given with no re-parsing. Confirmed by the
    user: Search -> pick -> Subscribe never invoked the parser, while
    Add RSS -> Analyze -> Subscribe (which does call the full parser before
    subscribe_season ever sees the data) worked fine.

    parser="mikan" gets the same treatment for year/tvdb_id/id_source (mikan
    never resolved those at all, search-sourced or not -- a follow-up gap
    found once "tmdb" was fixed and the inconsistency became obvious), but
    keeps its own official_title/season since Mikan's homepage scrape is the
    more reliable source for those two.
    """

    async def test_search_sourced_subscribe_resolves_before_saving(self):
        """tvdb_id=None (never resolved, as a search result would be) must
        get a full TMDB/TVDB resolution before the row is persisted."""
        data = make_bangumi(
            filter="", year=None, tvdb_id=None, official_title="Test Anime"
        )
        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        tmdb_result = ("Resolved Title", 2, "2019", "poster.jpg", 359274, "tvdb")
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(_make_torrents()),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
            patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ) as mock_tmdb,
        ):
            await SeasonCollector.subscribe_season(data, parser="tmdb")

        mock_tmdb.assert_awaited_once()
        assert data.official_title == "Resolved Title"
        assert data.year == "2019"
        assert data.season == 2
        assert data.tvdb_id == 359274
        assert data.id_source == "tvdb"
        async with Database() as db:
            row = await db.bangumi.search_id(data.id)
            assert row is not None
            assert row.year == "2019"
            assert row.tvdb_id == 359274
            assert row.id_source == "tvdb"

    async def test_already_resolved_subscribe_is_not_re_resolved(self):
        """A bangumi that already went through full resolution (e.g. via the
        Add-RSS analysis flow, which calls the parser before subscribe_season
        ever sees the data) must not be re-fetched -- tvdb_id already being
        set is the signal that it was."""
        data = make_bangumi(filter="", year="2019", tvdb_id=359274, id_source="tvdb")
        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(_make_torrents()),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
            patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(),
            ) as mock_tmdb,
        ):
            await SeasonCollector.subscribe_season(data, parser="tmdb")

        mock_tmdb.assert_not_awaited()

    async def test_mikan_parser_subscribe_resolves_year_and_tvdb_but_keeps_title(self):
        """parser="mikan" gets the same year/tvdb_id/id_source cross-
        reference as "tmdb" (mikan's own homepage scrape never provides
        those), but keeps its own official_title/season -- Mikan's scrape is
        more reliable for those than a TMDB text search, and overwriting
        season with TMDB's own guess is a plausible cause of separate
        season-mislabeling reports."""
        data = make_bangumi(
            filter="", year=None, tvdb_id=None, official_title="Mikan Title", season=1
        )
        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        tmdb_result = ("TMDB Title", 2, "2019", "poster.jpg", 359274, "tvdb")
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(_make_torrents()),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
            patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ) as mock_tmdb,
        ):
            await SeasonCollector.subscribe_season(data, parser="mikan")

        mock_tmdb.assert_awaited_once()
        assert data.official_title == "Mikan Title"
        assert data.season == 1
        assert data.year == "2019"
        assert data.tvdb_id == 359274
        assert data.id_source == "tvdb"

    async def test_already_resolved_mikan_subscribe_is_not_re_resolved(self):
        """Same not-re-resolved guard as the "tmdb" case applies to "mikan"."""
        data = make_bangumi(filter="", year="2019", tvdb_id=359274, id_source="tvdb")
        downloader_client = AsyncMock()
        downloader_client.add_torrent = AsyncMock(return_value=AddResult.ADDED)
        with (
            patch(
                "module.rss.engine.RequestContent",
                return_value=_req_with_torrents(_make_torrents()),
            ),
            patch(
                "module.rss.engine.DownloadClient",
                return_value=_async_ctx(downloader_client),
            ),
            patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(),
            ) as mock_tmdb,
        ):
            await SeasonCollector.subscribe_season(data, parser="mikan")

        mock_tmdb.assert_not_awaited()


def _client_with_all_torrents(all_torrents=None, **overrides):
    """AsyncMock download client with get_torrent_info() defaulting to an
    empty list -- reparse_bangumi() always calls it once, broadly, to learn
    every torrent's *actual* current save_path (ground truth for "does this
    need moving", not anything derived from stored metadata). Leaving it
    unconfigured would iterate a bare MagicMock and blow up with TypeError.
    """
    client = AsyncMock()
    client.get_torrent_info = AsyncMock(return_value=all_torrents or [])
    for name, value in overrides.items():
        setattr(client, name, value)
    return client


class TestReparseBangumi:
    """reparse_bangumi(): re-run TMDB for an existing bangumi (e.g. one
    subscribed before the search-subscribe fix, or added with a bare title)
    and move its already-downloaded torrents into the corrected folder.

    Always checks each tracked torrent's actual reported save_path against
    the freshly-computed folder -- never short-circuits on whether
    official_title/year/tvdb_id textually changed, since the DB's metadata
    can already be "correct" while the files were never actually moved to
    match (#1044: reparse silently did nothing for exactly this reason)."""

    async def test_reparses_metadata_and_moves_a_misplaced_torrent(self):
        bangumi = make_bangumi(
            filter="",
            official_title="Wrong Title",
            year=None,
            tvdb_id=None,
            id_source=None,
            season=1,
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        client = _client_with_all_torrents(
            all_torrents=[
                {
                    "hash": "abc123",
                    "save_path": "/downloads/Bangumi/Wrong Title/Season 1",
                    "tags": "",
                }
            ]
        )
        tmdb_result = ("Correct Title", 2, "2019", "poster.jpg", 359274, "tvdb")

        async with Database() as db:
            with patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.old_official_title == "Wrong Title"
        assert result.new_official_title == "Correct Title"
        assert result.new_year == "2019"
        assert result.new_tvdb_id == 359274
        assert result.new_id_source == "tvdb"
        assert result.metadata_changed is True
        assert result.torrents_found == 1
        assert result.torrents_already_correct == 0
        assert result.torrents_moved == 1
        assert result.torrents_failed == 0
        client.move_torrent.assert_awaited_once_with("abc123", result.new_folder)

        async with Database() as db:
            updated = await db.bangumi.search_id(bangumi_id)
            assert updated is not None
            assert updated.official_title == "Correct Title"
            assert updated.season == 1  # unchanged despite tmdb_result season=2
            assert updated.tvdb_id == 359274
            assert updated.save_path == result.new_folder

    async def test_returns_none_for_unknown_bangumi(self):
        client = _client_with_all_torrents()
        async with Database() as db:
            result = await reparse_bangumi(db, client, 999999)

        assert result is None
        client.move_torrent.assert_not_awaited()

    async def test_torrent_already_in_correct_folder_is_not_moved(self):
        """Regression for the exact bug reported: metadata was already
        correct in the DB (so old/new metadata-derived paths matched), but
        that alone must not be treated as proof the files were ever moved
        -- only a torrent whose *actual* reported save_path already equals
        the new folder should be skipped."""
        bangumi = make_bangumi(
            filter="",
            official_title="Already Correct",
            year="2019",
            tvdb_id=359274,
            id_source="tvdb",
            season=1,
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        tmdb_result = ("Already Correct", 1, "2019", "poster.jpg", 359274, "tvdb")
        expected_new_folder = gen_save_path(
            make_bangumi(
                official_title="Already Correct",
                year="2019",
                tvdb_id=359274,
                id_source="tvdb",
                season=1,
            )
        )
        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": expected_new_folder, "tags": ""}
            ]
        )

        async with Database() as db:
            with patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.metadata_changed is False
        assert result.torrents_found == 1
        assert result.torrents_already_correct == 1
        assert result.torrents_moved == 0
        client.move_torrent.assert_not_awaited()

    async def test_reports_per_torrent_move_failure_without_aborting(self):
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="hash1",
                    bangumi_id=bangumi_id,
                )
            )
            await db.torrent.add(
                Torrent(
                    name="ep2.mkv",
                    url="https://example.com/ep2.torrent",
                    qb_hash="hash2",
                    bangumi_id=bangumi_id,
                )
            )

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "hash1", "save_path": "/downloads/Old/Season 1", "tags": ""},
                {"hash": "hash2", "save_path": "/downloads/Old/Season 1", "tags": ""},
            ]
        )
        client.move_torrent = AsyncMock(side_effect=[None, RuntimeError("boom")])
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_found == 2
        assert result.torrents_moved == 1
        assert result.torrents_failed == 1

    async def test_finds_torrents_via_ab_tag_when_db_link_is_missing(self):
        """Regression: a bangumi whose folder needs fixing is exactly the
        kind whose Torrent.bangumi_id FK may be stale/missing (e.g. broken
        by whatever caused the wrong metadata in the first place) -- the
        downloader's own "ab:<id>" tag must still find its torrents so
        reparse doesn't silently report 0 torrents moved."""
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            # Deliberately no db.torrent.add() -- nothing in the Torrent
            # table links to this bangumi_id, only the downloader's tag does.

        client = _client_with_all_torrents(
            all_torrents=[
                {
                    "hash": "tagged-hash",
                    "save_path": "/downloads/Old/Season 1",
                    "tags": f"ab:{bangumi_id}",
                }
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_found == 1
        assert result.torrents_moved == 1
        client.move_torrent.assert_awaited_once_with("tagged-hash", result.new_folder)
        client.get_torrent_info.assert_awaited_once_with(
            category=None, status_filter=None, tag=None
        )

    async def test_deduplicates_a_torrent_found_via_both_db_and_tag(self):
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="same-hash",
                    bangumi_id=bangumi_id,
                )
            )

        client = _client_with_all_torrents(
            all_torrents=[
                {
                    "hash": "same-hash",
                    "save_path": "/downloads/Old/Season 1",
                    "tags": f"ab:{bangumi_id}",
                }
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_found == 1
        assert result.torrents_moved == 1

    async def test_reports_zero_found_when_no_torrent_is_tracked_anywhere(self):
        """TMDB resolution succeeds and the folder changes, but nothing is
        tracked under this bangumi_id in either the DB or the downloader's
        tags -- 0 moved, and torrents_found says why."""
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id

        client = _client_with_all_torrents()
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with patch(
                "module.manager.collector.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.metadata_changed is True
        assert result.torrents_found == 0
        assert result.torrents_moved == 0
        client.move_torrent.assert_not_awaited()

    async def test_removes_now_empty_old_folder_after_moving(self, tmp_path):
        """Reported: reparse moved the torrent but left the old, now-empty
        show/season directory behind on disk."""
        bangumi = make_bangumi(
            filter="",
            official_title="Wrong Title",
            year=None,
            tvdb_id=None,
            season=1,
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        old_show_dir = tmp_path / "Wrong Title"
        old_season_dir = old_show_dir / "Season 1"
        old_season_dir.mkdir(parents=True)

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_dir), "tags": ""}
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_moved == 1
        assert result.folders_removed == 2  # "Season 1", then "Wrong Title"
        assert not old_season_dir.exists()
        assert not old_show_dir.exists()
        assert tmp_path.exists()  # download root itself is never removed

    async def test_cleanup_never_removes_the_download_root(self, tmp_path):
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        # The torrent's old save_path *is* the download root -- nothing to
        # walk up past, and the root itself must survive.
        client = _client_with_all_torrents(
            all_torrents=[{"hash": "abc123", "save_path": str(tmp_path), "tags": ""}]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.folders_removed == 0
        assert tmp_path.exists()

    async def test_cleanup_leaves_a_still_occupied_parent_alone(self, tmp_path):
        """A sibling season folder (or any other file) still under the old
        show directory must stop the walk-up -- only the emptied leaf goes."""
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        old_show_dir = tmp_path / "Wrong Title"
        old_season_1 = old_show_dir / "Season 1"
        old_season_2 = old_show_dir / "Season 2"
        old_season_1.mkdir(parents=True)
        old_season_2.mkdir(parents=True)  # sibling -- keeps "Wrong Title" occupied

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_1), "tags": ""}
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.folders_removed == 1  # just "Season 1"
        assert not old_season_1.exists()
        assert old_show_dir.exists()  # "Wrong Title" survives -- Season 2 is in it
        assert old_season_2.exists()

    async def test_cleanup_tolerates_a_folder_two_torrents_already_shared(
        self, tmp_path
    ):
        """Two torrents from the same old season folder: the second cleanup
        attempt hits an already-removed directory and must not raise."""
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="hash1",
                    bangumi_id=bangumi_id,
                )
            )
            await db.torrent.add(
                Torrent(
                    name="ep2.mkv",
                    url="https://example.com/ep2.torrent",
                    qb_hash="hash2",
                    bangumi_id=bangumi_id,
                )
            )

        old_show_dir = tmp_path / "Wrong Title"
        old_season_dir = old_show_dir / "Season 1"
        old_season_dir.mkdir(parents=True)

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "hash1", "save_path": str(old_season_dir), "tags": ""},
                {"hash": "hash2", "save_path": str(old_season_dir), "tags": ""},
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_moved == 2
        # Same shared old path -- only removed (and counted) once.
        assert result.folders_removed == 2
        assert not old_show_dir.exists()

    async def test_cleanup_retries_rmdir_before_giving_up(self, tmp_path):
        """move_torrent returns as soon as qBittorrent *accepts* the move,
        not once it's actually finished relocating the content -- the old
        directory can briefly still report non-empty. Cleanup must retry
        instead of giving up on the very first attempt."""
        bangumi = make_bangumi(
            filter="",
            official_title="Wrong Title",
            year=None,
            tvdb_id=None,
            season=1,
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        old_show_dir = tmp_path / "Wrong Title"
        old_season_dir = old_show_dir / "Season 1"
        old_season_dir.mkdir(parents=True)

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_dir), "tags": ""}
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        import errno as errno_module
        from pathlib import Path

        real_rmdir = Path.rmdir
        attempts_left = {"n": 2}

        def flaky_rmdir(self):
            if self == old_season_dir and attempts_left["n"] > 0:
                attempts_left["n"] -= 1
                raise OSError(errno_module.ENOTEMPTY, "Directory not empty")
            return real_rmdir(self)

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
                patch("module.manager.collector._CLEANUP_RMDIR_RETRY_DELAY", 0),
                patch.object(Path, "rmdir", flaky_rmdir),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert attempts_left["n"] == 0  # actually exercised the retry path
        assert result.folders_removed == 2
        assert not old_season_dir.exists()
        assert not old_show_dir.exists()

    async def test_cleanup_gives_up_and_logs_after_exhausting_retries(
        self, tmp_path, caplog
    ):
        """A directory that's genuinely still occupied (not just briefly
        racing an in-progress move) must eventually be left alone --
        forever retrying would hang the one-shot Reparse request -- and the
        reason must be logged instead of silently vanishing."""
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        old_show_dir = tmp_path / "Wrong Title"
        old_season_dir = old_show_dir / "Season 1"
        old_season_dir.mkdir(parents=True)

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_dir), "tags": ""}
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        import errno as errno_module
        from pathlib import Path

        def always_not_empty(self):
            if self == old_season_dir:
                raise OSError(errno_module.ENOTEMPTY, "Directory not empty")
            raise AssertionError("should never walk past the occupied directory")

        with caplog.at_level("INFO", logger="module.manager.collector"):
            async with Database() as db:
                with (
                    patch(
                        "module.manager.collector.settings.downloader.path",
                        str(tmp_path),
                    ),
                    patch(
                        "module.manager.collector.TitleParser.tmdb_parser",
                        AsyncMock(return_value=tmdb_result),
                    ),
                    patch("module.manager.collector._CLEANUP_RMDIR_ATTEMPTS", 2),
                    patch("module.manager.collector._CLEANUP_RMDIR_RETRY_DELAY", 0),
                    patch.object(Path, "rmdir", always_not_empty),
                ):
                    result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.folders_removed == 0
        assert old_season_dir.exists()
        assert "still had content after waiting" in caplog.text

    async def test_cleanup_logs_warning_when_vacated_path_outside_download_root(
        self, tmp_path, caplog
    ):
        """The downloader reporting a save_path that isn't under the
        configured download root (e.g. AutoBangumi and the downloader
        mounting the shared volume at different paths in separate
        containers) can never be cleaned up by this process no matter how
        the path is sliced -- must be logged clearly instead of looking
        identical to "nothing needed cleaning up"."""
        download_root = tmp_path / "downloads"
        download_root.mkdir()
        other_mount = tmp_path / "elsewhere"
        old_season_dir = other_mount / "Wrong Title" / "Season 1"
        old_season_dir.mkdir(parents=True)

        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_dir), "tags": ""}
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        with caplog.at_level("WARNING", logger="module.manager.collector"):
            async with Database() as db:
                with (
                    patch(
                        "module.manager.collector.settings.downloader.path",
                        str(download_root),
                    ),
                    patch(
                        "module.manager.collector.TitleParser.tmdb_parser",
                        AsyncMock(return_value=tmdb_result),
                    ),
                ):
                    result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.folders_removed == 0
        assert old_season_dir.exists()
        assert "not under the configured download root" in caplog.text

    async def test_reparse_fixes_a_folder_that_was_missing_its_year(self, tmp_path):
        """Regression: _media_folder() used to truncate "title (year)" as
        one blob for a long enough title, dropping the year entirely (and
        with it, Plex/HAMA's "Title (Year)" match) -- the exact title
        below did this in production. Reparse re-derives the folder with
        the fixed generator and must move the already-downloaded torrent
        out of the old, year-less folder into the corrected one -- a user
        should be able to fix an already-broken show with one click,
        without re-downloading anything."""
        long_title = (
            "追放されたチート付与魔術師は気ままなセカンドライフを謳歌する。"
            "～俺は武器だけじゃなく、あらゆるものに『強化ポイント』を付与できるし、"
            "俺の意思でいつでも効果を解除できるけど、残った人たち大丈夫？～"
        )
        # The exact folder beta.13 produced: title truncated, year dropped,
        # tag intact.
        old_folder_name = (
            "追放されたチート付与魔術師は気ままなセカンドライフを謳歌する。"
            "～俺は武器だけじゃなく、あらゆるものに『強化ポイント』を付与できるし、"
            "俺の意思でいつでも効果を解除 [tvdb-473642]"
        )
        bangumi = make_bangumi(
            filter="",
            official_title=long_title,
            year="2026",
            tvdb_id=473642,
            id_source="tvdb",
            season=1,
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        old_season_dir = tmp_path / old_folder_name / "Season 1"
        old_season_dir.mkdir(parents=True)

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_dir), "tags": ""}
            ]
        )
        # Same title/year/tvdb_id as already stored -- this is re-deriving
        # the folder with fixed code, not correcting wrong metadata.
        tmdb_result = (long_title, 1, "2026", None, 473642, "tvdb")

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
            ):
                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_moved == 1
        assert "(2026)" in result.new_folder
        client.move_torrent.assert_awaited_once_with("abc123", result.new_folder)
        # The now-empty old (year-less) folder is cleaned up too.
        assert not old_season_dir.exists()

    async def test_reparse_fixes_restrictive_permissions_left_by_the_move(
        self, tmp_path
    ):
        """Reported: a torrent relocated via move_torrent ends up with its
        destination directory at 0o700 (created by the downloader itself,
        not this process) instead of the 0o775 every normally-downloaded
        show gets -- which Plex, running as a different user, can't even
        list. move_torrent is mocked here (as in every other test in this
        class) so it never touches the filesystem itself; the new folder
        is pre-created with the bad permissions to stand in for whatever
        the real downloader would have produced, and reparse_bangumi must
        normalize it after the (mocked) move succeeds."""
        bangumi = make_bangumi(
            filter="", official_title="Wrong Title", year=None, tvdb_id=None
        )
        async with Database() as db:
            await db.bangumi.add(bangumi)
            bangumi_id = bangumi.id
            await db.torrent.add(
                Torrent(
                    name="ep1.mkv",
                    url="https://example.com/ep1.torrent",
                    qb_hash="abc123",
                    bangumi_id=bangumi_id,
                )
            )

        tmp_path.chmod(0o775)
        old_season_dir = tmp_path / "Old Folder" / "Season 1"
        old_season_dir.mkdir(parents=True)

        client = _client_with_all_torrents(
            all_torrents=[
                {"hash": "abc123", "save_path": str(old_season_dir), "tags": ""}
            ]
        )
        tmdb_result = ("Correct Title", 1, "2019", None, 359274, "tvdb")

        async with Database() as db:
            with (
                patch(
                    "module.manager.collector.settings.downloader.path",
                    str(tmp_path),
                ),
                patch(
                    "module.manager.collector.TitleParser.tmdb_parser",
                    AsyncMock(return_value=tmdb_result),
                ),
            ):
                # Stand in for the downloader having just created the
                # destination with restrictive permissions, since the
                # mocked move_torrent never touches the real filesystem.
                # The bug was reported on the *show* folder itself, one
                # level above the Season N directory reparse computes.
                new_show_dir = tmp_path / "Correct Title (2019) [tvdb-359274]"
                new_season_dir = new_show_dir / "Season 1"
                new_season_dir.mkdir(parents=True)
                new_show_dir.chmod(0o700)
                new_season_dir.chmod(0o700)

                result = await reparse_bangumi(db, client, bangumi_id)

        assert result is not None
        assert result.torrents_moved == 1
        assert stat.S_IMODE(new_show_dir.stat().st_mode) == 0o775
        assert stat.S_IMODE(new_season_dir.stat().st_mode) == 0o775
