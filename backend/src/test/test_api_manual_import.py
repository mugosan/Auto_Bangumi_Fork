"""Tests for the manual-import API routes: end-to-end wiring through FastAPI,
not just the manager-layer functions test_manual_import.py already covers.

These caught a real bug: ImportCandidate/FileMapping/etc are
`@dataclass(slots=True)`, which have no `__dict__` -- every route that did
`.__dict__` on one (candidates, preview, upload) raised AttributeError on
every real request despite the manager-layer unit tests passing, because
those tests never go through the API layer at all. Fixed with
`dataclasses.asdict()`; these tests exercise the actual routes so a
regression back to `.__dict__` fails immediately instead of silently
shipping broken again."""

from unittest.mock import AsyncMock, patch

from module.downloader import AddResult


def _patch_download_client(mock_client):
    """Match the established pattern (see test_api_downloader.py) for
    mocking `async with DownloadClient() as client:` at the API layer."""
    patcher = patch("module.api.manual_import.DownloadClient")
    MockClient = patcher.start()
    MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
    return patcher


class TestUploadTorrentRoute:
    def test_upload_success_returns_candidate(self, authed_client):
        client = AsyncMock()
        client.add_torrent_file = AsyncMock(return_value=(AddResult.ADDED, "abc123"))
        client.get_torrent_info = AsyncMock(
            return_value=[
                {
                    "hash": "abc123",
                    "name": "My Show Batch",
                    "save_path": "/downloads",
                    "category": "",
                    "size": 100,
                    "progress": 0.0,
                    "state": "",
                }
            ]
        )
        patcher = _patch_download_client(client)
        try:
            response = authed_client.post(
                "/api/v1/manual-import/upload",
                files={
                    "file": (
                        "show.torrent",
                        b"d4:infod4:name3:fooee",
                        "application/x-bittorrent",
                    )
                },
            )
        finally:
            patcher.stop()

        assert response.status_code == 200
        assert response.json()["hash"] == "abc123"
        assert response.json()["name"] == "My Show Batch"

    def test_upload_rejects_empty_file(self, authed_client):
        response = authed_client.post(
            "/api/v1/manual-import/upload",
            files={"file": ("show.torrent", b"", "application/x-bittorrent")},
        )

        assert response.status_code == 400

    def test_upload_rejects_invalid_torrent(self, authed_client):
        client = AsyncMock()
        client.add_torrent_file = AsyncMock(return_value=(AddResult.FAILED, None))
        patcher = _patch_download_client(client)
        try:
            response = authed_client.post(
                "/api/v1/manual-import/upload",
                files={
                    "file": (
                        "bad.torrent",
                        b"not a torrent",
                        "application/x-bittorrent",
                    )
                },
            )
        finally:
            patcher.stop()

        assert response.status_code == 400

    def test_upload_unauthorized(self, unauthed_client):
        response = unauthed_client.post(
            "/api/v1/manual-import/upload",
            files={"file": ("show.torrent", b"x", "application/x-bittorrent")},
        )

        assert response.status_code == 401


class TestCandidatesRoute:
    def test_returns_serialized_candidates(self, authed_client):
        client = AsyncMock()
        client.get_torrent_info = AsyncMock(
            return_value=[
                {
                    "hash": "unmanaged",
                    "name": "Random Batch Torrent",
                    "save_path": "/downloads",
                    "category": "",
                    "size": 200,
                    "progress": 1.0,
                    "state": "uploading",
                    "tags": "",
                }
            ]
        )
        patcher = _patch_download_client(client)
        try:
            response = authed_client.get("/api/v1/manual-import/candidates")
        finally:
            patcher.stop()

        assert response.status_code == 200
        assert response.json() == [
            {
                "hash": "unmanaged",
                "name": "Random Batch Torrent",
                "save_path": "/downloads",
                "category": "",
                "size": 200,
                "progress": 1.0,
                "state": "uploading",
            }
        ]


class TestPreviewRoute:
    def test_returns_serialized_preview(self, authed_client):
        client = AsyncMock()
        client.get_torrent_info = AsyncMock(
            return_value=[
                {
                    "hash": "abc123",
                    "name": "[Group] My Show 01-02 Batch",
                    "save_path": "/downloads/incoming",
                }
            ]
        )
        client.get_torrent_files = AsyncMock(
            return_value=[{"name": "[Group] My Show - 01.mkv"}]
        )
        patcher = _patch_download_client(client)
        tmdb_result = ("My Show", 1, "2024", None, 12345, "tvdb")
        try:
            with patch(
                "module.manager.manual_import.TitleParser.tmdb_parser",
                AsyncMock(return_value=tmdb_result),
            ):
                response = authed_client.post(
                    "/api/v1/manual-import/preview",
                    json={
                        "torrent_hash": "abc123",
                        "official_title": "My Show",
                        "season": 1,
                    },
                )
        finally:
            patcher.stop()

        assert response.status_code == 200
        body = response.json()
        assert body["official_title"] == "My Show"
        assert body["tvdb_id"] == 12345
        assert len(body["mappings"]) == 1
        assert body["mappings"][0]["target_path"] == "My Show S01E01.mkv"


class TestFolderPreviewRoute:
    def test_returns_serialized_preview(self, authed_client, tmp_path):
        root = tmp_path / "Batch"
        root.mkdir()
        (root / "[Group] My Show - 01.mkv").write_bytes(b"")

        tmdb_result = ("My Show", 1, "2024", None, 12345, "tvdb")
        with patch(
            "module.manager.manual_import.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ):
            response = authed_client.post(
                "/api/v1/manual-import/folder/preview",
                json={
                    "path": str(root),
                    "official_title": "My Show",
                    "season": 1,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["official_title"] == "My Show"
        assert body["tvdb_id"] == 12345
        assert len(body["mappings"]) == 1
        assert body["mappings"][0]["target_path"].endswith("My Show S01E01.mkv")
