"""Tests for POST /bangumi/reparse/{bangumi_id}: route wiring only -- the
actual reparse logic (TMDB re-resolution + torrent move) is covered by
test_collector.py's TestReparseBangumi."""

from unittest.mock import AsyncMock, patch

from module.manager.collector import ReparseResult


def _patch_download_client(mock_client):
    patcher = patch("module.api.bangumi.DownloadClient")
    MockClient = patcher.start()
    MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
    return patcher


_MOVED_RESULT = ReparseResult(
    old_official_title="Wrong Title",
    new_official_title="Correct Title",
    old_year=None,
    new_year="2019",
    old_tvdb_id=None,
    new_tvdb_id=359274,
    old_id_source=None,
    new_id_source="tvdb",
    old_folder="/downloads/Wrong Title/Season 1",
    new_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
    metadata_changed=True,
    torrents_found=2,
    torrents_already_correct=0,
    torrents_moved=2,
    torrents_failed=0,
)


class TestReparseRoute:
    def test_reparse_success_reports_moved_torrents(self, authed_client):
        patcher = _patch_download_client(AsyncMock())
        try:
            with patch(
                "module.api.bangumi.reparse_bangumi",
                AsyncMock(return_value=_MOVED_RESULT),
            ) as mock_reparse:
                response = authed_client.post("/api/v1/bangumi/reparse/1")
        finally:
            patcher.stop()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] is True
        assert "Correct Title" in body["msg_en"]
        assert "2" in body["msg_en"]
        mock_reparse.assert_awaited_once()
        assert mock_reparse.await_args is not None
        assert mock_reparse.await_args.args[2] == 1

    def test_reparse_not_found(self, authed_client):
        patcher = _patch_download_client(AsyncMock())
        try:
            with patch(
                "module.api.bangumi.reparse_bangumi",
                AsyncMock(return_value=None),
            ):
                response = authed_client.post("/api/v1/bangumi/reparse/999")
        finally:
            patcher.stop()

        assert response.status_code == 404

    def test_reparse_all_torrents_already_correct(self, authed_client):
        """Regression for #1044: metadata being unchanged is no longer what
        this branch reports on -- it's specifically "every tracked torrent's
        actual save_path already matches", which the route can only know
        from torrents_already_correct/torrents_found, not metadata_changed."""
        result = ReparseResult(
            old_official_title="Correct Title",
            new_official_title="Correct Title",
            old_year="2019",
            new_year="2019",
            old_tvdb_id=359274,
            new_tvdb_id=359274,
            old_id_source="tvdb",
            new_id_source="tvdb",
            old_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
            new_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
            metadata_changed=False,
            torrents_found=1,
            torrents_already_correct=1,
            torrents_moved=0,
            torrents_failed=0,
        )
        patcher = _patch_download_client(AsyncMock())
        try:
            with patch(
                "module.api.bangumi.reparse_bangumi",
                AsyncMock(return_value=result),
            ):
                response = authed_client.post("/api/v1/bangumi/reparse/1")
        finally:
            patcher.stop()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] is True
        assert "already" in body["msg_en"]

    def test_reparse_metadata_unchanged_but_torrent_still_moved(self, authed_client):
        """The exact bug fixed: metadata_changed=False must NOT stop reparse
        from moving a torrent whose actual save_path differs from the target
        folder (the DB's metadata was already correct, but the files never
        were moved to match)."""
        result = ReparseResult(
            old_official_title="Correct Title",
            new_official_title="Correct Title",
            old_year="2019",
            new_year="2019",
            old_tvdb_id=359274,
            new_tvdb_id=359274,
            old_id_source="tvdb",
            new_id_source="tvdb",
            old_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
            new_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
            metadata_changed=False,
            torrents_found=1,
            torrents_already_correct=0,
            torrents_moved=1,
            torrents_failed=0,
        )
        patcher = _patch_download_client(AsyncMock())
        try:
            with patch(
                "module.api.bangumi.reparse_bangumi",
                AsyncMock(return_value=result),
            ):
                response = authed_client.post("/api/v1/bangumi/reparse/1")
        finally:
            patcher.stop()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] is True
        assert "Moved 1 of 1" in body["msg_en"]

    def test_reparse_partial_failure_reports_status_false(self, authed_client):
        result = ReparseResult(
            old_official_title="Wrong Title",
            new_official_title="Correct Title",
            old_year=None,
            new_year="2019",
            old_tvdb_id=None,
            new_tvdb_id=359274,
            old_id_source=None,
            new_id_source="tvdb",
            old_folder="/downloads/Wrong Title/Season 1",
            new_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
            metadata_changed=True,
            torrents_found=2,
            torrents_already_correct=0,
            torrents_moved=1,
            torrents_failed=1,
        )
        patcher = _patch_download_client(AsyncMock())
        try:
            with patch(
                "module.api.bangumi.reparse_bangumi",
                AsyncMock(return_value=result),
            ):
                response = authed_client.post("/api/v1/bangumi/reparse/1")
        finally:
            patcher.stop()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] is False

    def test_reparse_no_torrents_found_explains_why_nothing_moved(self, authed_client):
        """folder metadata resolved fine, but nothing is tracked for this
        bangumi anywhere -- the message must say why instead of looking
        like a silent no-op."""
        result = ReparseResult(
            old_official_title="Wrong Title",
            new_official_title="Correct Title",
            old_year=None,
            new_year="2019",
            old_tvdb_id=None,
            new_tvdb_id=359274,
            old_id_source=None,
            new_id_source="tvdb",
            old_folder="/downloads/Wrong Title/Season 1",
            new_folder="/downloads/Correct Title (2019) [tvdb-359274]/Season 1",
            metadata_changed=True,
            torrents_found=0,
            torrents_already_correct=0,
            torrents_moved=0,
            torrents_failed=0,
        )
        patcher = _patch_download_client(AsyncMock())
        try:
            with patch(
                "module.api.bangumi.reparse_bangumi",
                AsyncMock(return_value=result),
            ):
                response = authed_client.post("/api/v1/bangumi/reparse/1")
        finally:
            patcher.stop()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] is True
        assert "No tracked torrents found" in body["msg_en"]
        assert result.new_folder in body["msg_en"]

    def test_reparse_unauthorized(self, unauthed_client):
        response = unauthed_client.post("/api/v1/bangumi/reparse/1")

        assert response.status_code == 401
