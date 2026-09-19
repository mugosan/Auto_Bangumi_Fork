"""Tests for manual_import: listing unmanaged torrents, preview, apply."""

from unittest.mock import AsyncMock, patch

from module.downloader import RenameOutcome, RenameResult
from module.manager.manual_import import (
    FileMapping,
    apply_import,
    list_import_candidates,
    preview_import,
)


def _make_client(**overrides):
    client = AsyncMock()
    for name, value in overrides.items():
        setattr(client, name, value)
    return client


class TestListImportCandidates:
    async def test_excludes_torrents_with_ab_tag(self):
        client = _make_client(
            get_torrent_info=AsyncMock(
                return_value=[
                    {
                        "hash": "managed",
                        "name": "Managed Show",
                        "save_path": "/downloads/Managed",
                        "category": "Bangumi",
                        "size": 100,
                        "progress": 1.0,
                        "state": "uploading",
                        "tags": "ab:renamed",
                    },
                    {
                        "hash": "unmanaged",
                        "name": "Random Batch Torrent",
                        "save_path": "/downloads",
                        "category": "",
                        "size": 200,
                        "progress": 1.0,
                        "state": "uploading",
                        "tags": "",
                    },
                ]
            )
        )

        candidates = await list_import_candidates(client)

        assert [c.hash for c in candidates] == ["unmanaged"]
        client.get_torrent_info.assert_awaited_once_with(
            category=None, status_filter=None, tag=None
        )

    async def test_excludes_torrents_with_any_ab_prefixed_tag(self):
        """Tags are comma-separated; only one needs the ab: prefix."""
        client = _make_client(
            get_torrent_info=AsyncMock(
                return_value=[
                    {
                        "hash": "h1",
                        "name": "x",
                        "save_path": "",
                        "category": "",
                        "size": 0,
                        "progress": 0,
                        "state": "",
                        "tags": "some-other-tag, ab:renamed",
                    }
                ]
            )
        )

        candidates = await list_import_candidates(client)

        assert candidates == []


class TestPreviewImport:
    async def test_returns_none_for_unknown_hash(self):
        client = _make_client(get_torrent_info=AsyncMock(return_value=[]))

        result = await preview_import(client, "missing", "Some Show", 1)

        assert result is None

    async def test_resolves_tvdb_id_and_maps_files(self):
        client = _make_client(
            get_torrent_info=AsyncMock(
                return_value=[
                    {
                        "hash": "abc123",
                        "name": "[Group] My Show 01-12 Batch",
                        "save_path": "/downloads/incoming",
                    }
                ]
            ),
            get_torrent_files=AsyncMock(
                return_value=[
                    {"name": "[Group] My Show - 01.mkv"},
                    {"name": "[Group] My Show - 02.mkv"},
                    {"name": "[Group] My Show - 01.chs.srt"},
                    {"name": "readme.nfo"},
                ]
            ),
        )
        tmdb_result = (
            "冰海战记",  # resolved title
            2,  # tmdb season (ignored -- we use the user's season)
            "2019",  # year
            "poster.jpg",
            359274,  # meta_id (real tvdb id)
            "tvdb",  # id_source
        )
        with patch(
            "module.manager.manual_import.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ):
            preview = await preview_import(client, "abc123", "My Show", 1)

        assert preview is not None
        assert preview.official_title == "冰海战记"
        assert preview.year == "2019"
        assert preview.tvdb_id == 359274
        assert preview.id_source == "tvdb"
        assert "[tvdb-359274]" in preview.target_folder
        assert "冰海战记 (2019)" in preview.target_folder

        media_mappings = [m for m in preview.mappings if m.kind == "media"]
        subtitle_mappings = [m for m in preview.mappings if m.kind == "subtitle"]
        # Default rename_method is "pn": filenames use each file's own
        # embedded title ("My Show", scraped from the filename itself), not
        # the resolved TMDB/TVDB title -- same as the automated flow.
        assert {m.target_path for m in media_mappings} == {
            "My Show S01E01.mkv",
            "My Show S01E02.mkv",
        }
        assert [m.target_path for m in subtitle_mappings] == ["My Show S01E01.zh.srt"]
        # readme.nfo is neither media nor subtitle -- check_files drops it
        # before torrent_parser ever sees it, so it's not "unparsed" either.
        assert preview.unparsed == []

    async def test_unparseable_filename_is_reported_not_silently_dropped(self):
        client = _make_client(
            get_torrent_info=AsyncMock(
                return_value=[{"hash": "h1", "name": "torrent", "save_path": "/x"}]
            ),
            get_torrent_files=AsyncMock(
                return_value=[{"name": "no_episode_marker_at_all.mkv"}]
            ),
        )
        tmdb_result = ("My Show", 1, None, None, None, "tmdb")
        with patch(
            "module.manager.manual_import.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ):
            preview = await preview_import(client, "h1", "My Show", 1)

        assert preview is not None
        assert preview.mappings == []
        assert preview.unparsed == ["no_episode_marker_at_all.mkv"]

    async def test_no_tmdb_match_omits_id_tag_from_folder(self):
        client = _make_client(
            get_torrent_info=AsyncMock(
                return_value=[{"hash": "h1", "name": "torrent", "save_path": "/x"}]
            ),
            get_torrent_files=AsyncMock(return_value=[]),
        )
        tmdb_result = ("My Show", 1, None, None, None, "tmdb")
        with patch(
            "module.manager.manual_import.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ):
            preview = await preview_import(client, "h1", "My Show", 1)

        assert preview is not None
        assert "[" not in preview.target_folder
        assert preview.target_folder.endswith("My Show/Season 1")


class TestApplyImport:
    async def test_moves_then_renames_each_file(self):
        client = _make_client(
            move_torrent=AsyncMock(),
            rename_torrent_file=AsyncMock(
                return_value=RenameResult(RenameOutcome.RENAMED)
            ),
        )
        mappings = [
            FileMapping("a.mkv", "Show S01E01.mkv", 1, True, "media"),
            FileMapping("b.mkv", "Show S01E02.mkv", 2, True, "media"),
        ]

        results = await apply_import(
            client, "hash1", "/downloads/Show/Season 1", mappings
        )

        client.move_torrent.assert_awaited_once_with(
            "hash1", "/downloads/Show/Season 1"
        )
        assert client.rename_torrent_file.await_count == 2
        assert all(succeeded for _mapping, succeeded, _detail in results)

    async def test_skips_rename_call_when_paths_already_match(self):
        client = _make_client(move_torrent=AsyncMock(), rename_torrent_file=AsyncMock())
        mappings = [FileMapping("same.mkv", "same.mkv", 1, True, "media")]

        results = await apply_import(
            client, "hash1", "/downloads/Show/Season 1", mappings
        )

        client.rename_torrent_file.assert_not_awaited()
        assert results == [(mappings[0], True, None)]

    async def test_reports_per_file_failure_without_aborting(self):
        client = _make_client(
            move_torrent=AsyncMock(),
            rename_torrent_file=AsyncMock(
                side_effect=[
                    RenameResult(RenameOutcome.RENAMED),
                    RenameResult(RenameOutcome.DESTINATION_EXISTS, detail="exists"),
                ]
            ),
        )
        mappings = [
            FileMapping("a.mkv", "Show S01E01.mkv", 1, True, "media"),
            FileMapping("b.mkv", "Show S01E02.mkv", 2, True, "media"),
        ]

        results = await apply_import(
            client, "hash1", "/downloads/Show/Season 1", mappings
        )

        assert results[0][1] is True
        assert results[1] == (mappings[1], False, "exists")
