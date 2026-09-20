"""Tests for manual_import: listing unmanaged torrents, preview, apply."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from module.downloader import AddResult, RenameOutcome, RenameResult
from module.manager.manual_import import (
    FileMapping,
    FolderFileMapping,
    apply_folder_import,
    apply_import,
    list_import_candidates,
    preview_folder_import,
    preview_import,
    upload_torrent,
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


class TestUploadTorrent:
    async def test_returns_none_when_add_fails(self):
        client = _make_client(
            add_torrent_file=AsyncMock(return_value=(AddResult.FAILED, None))
        )

        result = await upload_torrent(client, b"not a valid torrent")

        assert result is None
        client.get_torrent_info.assert_not_awaited()

    async def test_returns_candidate_once_it_appears(self):
        client = _make_client(
            add_torrent_file=AsyncMock(return_value=(AddResult.ADDED, "abc123")),
            get_torrent_info=AsyncMock(
                return_value=[
                    {
                        "hash": "abc123",
                        "name": "My Show Batch",
                        "save_path": "/downloads",
                        "category": "",
                        "size": 12345,
                        "progress": 0.0,
                        "state": "metaDL",
                    }
                ]
            ),
        )

        result = await upload_torrent(client, b"...")

        assert result is not None
        assert result.hash == "abc123"
        assert result.name == "My Show Batch"

    async def test_polls_until_the_torrent_appears(self):
        """A raw .torrent file already has full metadata, but the downloader
        still needs a moment to register it -- poll rather than fail on the
        first empty listing."""
        candidate_info = {
            "hash": "abc123",
            "name": "My Show Batch",
            "save_path": "/downloads",
            "category": "",
            "size": 12345,
            "progress": 0.0,
            "state": "metaDL",
        }
        client = _make_client(
            add_torrent_file=AsyncMock(return_value=(AddResult.ADDED, "abc123")),
            get_torrent_info=AsyncMock(side_effect=[[], [], [candidate_info]]),
        )

        with patch("module.manager.manual_import.asyncio.sleep", AsyncMock()):
            result = await upload_torrent(client, b"...")

        assert result is not None
        assert result.hash == "abc123"
        assert client.get_torrent_info.await_count == 3

    async def test_gives_up_after_the_torrent_never_appears(self):
        client = _make_client(
            add_torrent_file=AsyncMock(return_value=(AddResult.ADDED, "abc123")),
            get_torrent_info=AsyncMock(return_value=[]),
        )

        with patch("module.manager.manual_import.asyncio.sleep", AsyncMock()):
            result = await upload_torrent(client, b"...")

        assert result is None
        assert client.get_torrent_info.await_count == 5


class TestPreviewFolderImport:
    async def test_returns_none_for_missing_path(self, tmp_path: Path):
        result = await preview_folder_import(
            str(tmp_path / "does-not-exist"), "Some Show", 1
        )

        assert result is None

    async def test_scans_recursively_and_maps_files(self, tmp_path: Path):
        root = tmp_path / "My Show Batch"
        (root / "sub").mkdir(parents=True)
        (root / "[Group] My Show - 01.mkv").write_bytes(b"")
        (root / "sub" / "[Group] My Show - 02.mkv").write_bytes(b"")
        (root / "[Group] My Show - 01.chs.srt").write_bytes(b"")
        (root / "readme.nfo").write_bytes(b"")

        tmdb_result = ("冰海战记", 2, "2019", "poster.jpg", 359274, "tvdb")
        with patch(
            "module.manager.manual_import.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ):
            preview = await preview_folder_import(str(root), "My Show", 1)

        assert preview is not None
        assert preview.official_title == "冰海战记"
        assert preview.year == "2019"
        assert preview.tvdb_id == 359274
        assert "[tvdb-359274]" in preview.target_folder

        media_mappings = [m for m in preview.mappings if m.kind == "media"]
        subtitle_mappings = [m for m in preview.mappings if m.kind == "subtitle"]
        assert {Path(m.target_path).name for m in media_mappings} == {
            "My Show S01E01.mkv",
            "My Show S01E02.mkv",
        }
        assert [Path(m.target_path).name for m in subtitle_mappings] == [
            "My Show S01E01.zh.srt"
        ]
        # target_path is absolute (target_folder / renamed name), unlike the
        # torrent-based FileMapping which is a bare relative filename.
        assert all(Path(m.target_path).is_absolute() for m in preview.mappings)
        assert preview.unparsed == []

    async def test_unparseable_filename_is_reported(self, tmp_path: Path):
        root = tmp_path / "Batch"
        root.mkdir()
        (root / "no_episode_marker_at_all.mkv").write_bytes(b"")

        tmdb_result = ("My Show", 1, None, None, None, "tmdb")
        with patch(
            "module.manager.manual_import.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ):
            preview = await preview_folder_import(str(root), "My Show", 1)

        assert preview is not None
        assert preview.mappings == []
        assert preview.unparsed == ["no_episode_marker_at_all.mkv"]


class TestApplyFolderImport:
    async def test_moves_and_renames_files_on_disk(self, tmp_path: Path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        (source_dir / "a.mkv").write_bytes(b"episode 1")

        mappings = [
            FolderFileMapping(
                source_path=str(source_dir / "a.mkv"),
                target_path=str(target_dir / "Show S01E01.mkv"),
                episode=1,
                parsed=True,
                kind="media",
            )
        ]

        results = await apply_folder_import(mappings)

        assert results == [(mappings[0], True, None)]
        assert not (source_dir / "a.mkv").exists()
        assert (target_dir / "Show S01E01.mkv").read_bytes() == b"episode 1"

    async def test_refuses_to_overwrite_existing_destination(self, tmp_path: Path):
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"
        source_dir.mkdir()
        target_dir.mkdir()
        (source_dir / "a.mkv").write_bytes(b"new")
        (target_dir / "Show S01E01.mkv").write_bytes(b"existing")

        mappings = [
            FolderFileMapping(
                source_path=str(source_dir / "a.mkv"),
                target_path=str(target_dir / "Show S01E01.mkv"),
                episode=1,
                parsed=True,
                kind="media",
            )
        ]

        results = await apply_folder_import(mappings)

        assert results == [(mappings[0], False, "destination already exists")]
        assert (source_dir / "a.mkv").exists()
        assert (target_dir / "Show S01E01.mkv").read_bytes() == b"existing"

    async def test_reports_missing_source_without_aborting(self, tmp_path: Path):
        mappings = [
            FolderFileMapping(
                source_path=str(tmp_path / "missing.mkv"),
                target_path=str(tmp_path / "target" / "Show S01E01.mkv"),
                episode=1,
                parsed=True,
                kind="media",
            ),
        ]

        results = await apply_folder_import(mappings)

        assert results == [(mappings[0], False, "source file not found")]

    async def test_skips_move_when_paths_already_match(self, tmp_path: Path):
        (tmp_path / "same.mkv").write_bytes(b"x")
        mappings = [
            FolderFileMapping(
                source_path=str(tmp_path / "same.mkv"),
                target_path=str(tmp_path / "same.mkv"),
                episode=1,
                parsed=True,
                kind="media",
            )
        ]

        results = await apply_folder_import(mappings)

        assert results == [(mappings[0], True, None)]
        assert (tmp_path / "same.mkv").exists()
