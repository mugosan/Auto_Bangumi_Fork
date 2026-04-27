"""Tests for the TVDB v4 metadata parser.

Unit tests: all network calls are mocked — no real API key or internet access required.
Integration test (test_tvdb_real_query): requires a configured tvdb.api_key in
config/config_dev.json and a live internet connection; skipped automatically otherwise.
The module-level token and cache are reset before each test via an autouse fixture.
"""

import importlib

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from module.parser.analyser.tvdb_parser import (
    TVDBInfo,
    _get,
    _last_aired_season,
    _official_seasons,
    tvdb_parser,
)

# analyser/__init__.py re-exports `tvdb_parser` as a package attribute, which
# shadows the submodule when accessed via `getattr`.  importlib.import_module
# returns the true module object from sys.modules, bypassing that shadowing.
tvdb_module = importlib.import_module("module.parser.analyser.tvdb_parser")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_response(json_data, status_code=200):
    """Build a minimal mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.content = b"fake-image-data"
    return resp


def make_settings(enable=True, api_key="fake-api-key", language="eng"):
    settings = MagicMock()
    settings.tvdb.enable = enable
    settings.tvdb.api_key = api_key
    settings.tvdb.language = language
    return settings


# ---------------------------------------------------------------------------
# Module-state fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_tvdb_state():
    """Clear module-level token and cache before and after each test."""
    tvdb_module._tvdb_token = None
    tvdb_module._tvdb_cache.clear()
    yield
    tvdb_module._tvdb_token = None
    tvdb_module._tvdb_cache.clear()


# ---------------------------------------------------------------------------
# Shared API response fixtures
# ---------------------------------------------------------------------------

LOGIN_RESP = {"status": "success", "data": {"token": "fake-token"}}

SEARCH_RESP = {
    "status": "success",
    "data": [{"tvdb_id": "12345", "name": "Test Anime"}],
}

EXTENDED_RESP = {
    "status": "success",
    "data": {
        "id": 12345,
        "name": "Test Anime",
        "firstAired": "2020-04-01",
        "status": {"name": "Ended"},
        "image": "https://example.com/poster.jpg",
        "seasons": [
            {"type": {"type": "official"}, "number": 1, "year": "2020", "image": None},
            {"type": {"type": "official"}, "number": 2, "year": "2021", "image": None},
            {"type": {"type": "dvd"}, "number": 1, "year": "2020", "image": None},
        ],
    },
}

TRANSLATION_RESP = {
    "status": "success",
    "data": {"name": "テストアニメ", "language": "jpn"},
}


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestOfficialSeasons:
    def test_keeps_numbered_official_seasons(self):
        seasons = [
            {"type": {"type": "official"}, "number": 1, "year": "2020"},
            {"type": {"type": "official"}, "number": 2, "year": "2021"},
        ]
        assert _official_seasons(seasons) == seasons

    def test_excludes_non_official_types(self):
        seasons = [
            {"type": {"type": "official"}, "number": 1},
            {"type": {"type": "dvd"}, "number": 1},
            {"type": {"type": "absolute"}, "number": 2},
        ]
        result = _official_seasons(seasons)
        assert len(result) == 1
        assert result[0]["number"] == 1

    def test_excludes_season_zero(self):
        """Season 0 is the specials season — should be filtered out."""
        seasons = [
            {"type": {"type": "official"}, "number": 0, "year": "2020"},
            {"type": {"type": "official"}, "number": 1, "year": "2020"},
        ]
        result = _official_seasons(seasons)
        assert len(result) == 1
        assert result[0]["number"] == 1

    def test_empty_input(self):
        assert _official_seasons([]) == []

    def test_all_non_official(self):
        seasons = [{"type": {"type": "dvd"}, "number": 1}]
        assert _official_seasons(seasons) == []


class TestLastAiredSeason:
    def test_returns_highest_past_season(self):
        seasons = [
            {"number": 1, "year": "2020"},
            {"number": 2, "year": "2022"},
            {"number": 3, "year": "2099"},  # future
        ]
        assert _last_aired_season(seasons) == 2

    def test_empty_returns_1(self):
        assert _last_aired_season([]) == 1

    def test_single_past_season(self):
        seasons = [{"number": 1, "year": "2020"}]
        assert _last_aired_season(seasons) == 1

    def test_all_future_falls_back_to_count(self):
        seasons = [{"number": 1, "year": "2099"}, {"number": 2, "year": "2099"}]
        # No valid years → falls back to len(seasons)
        assert _last_aired_season(seasons) == 2

    def test_missing_year_is_skipped(self):
        seasons = [
            {"number": 1, "year": None},
            {"number": 2, "year": "2021"},
        ]
        assert _last_aired_season(seasons) == 2


# ---------------------------------------------------------------------------
# _get() — authenticated GET helper
# ---------------------------------------------------------------------------


class TestGet:
    async def test_returns_json_on_success(self):
        resp = make_response({"status": "success", "data": {}})
        client = AsyncMock()
        client.get.return_value = resp

        with patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await _get("token", "/search", {"query": "test"})

        assert result == {"status": "success", "data": {}}

    async def test_401_clears_token_and_returns_none(self):
        tvdb_module._tvdb_token = "stale-token"
        resp = make_response({}, status_code=401)
        client = AsyncMock()
        client.get.return_value = resp

        with patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await _get("stale-token", "/search")

        assert result is None
        assert tvdb_module._tvdb_token is None

    async def test_exception_returns_none(self):
        client = AsyncMock()
        client.get.side_effect = Exception("Network error")

        with patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await _get("token", "/search")

        assert result is None


# ---------------------------------------------------------------------------
# tvdb_parser() — full parser
# ---------------------------------------------------------------------------


class TestTvdbParser:
    async def test_disabled_returns_none(self):
        settings = make_settings(enable=False)
        with patch("module.conf.settings", settings):
            result = await tvdb_parser("Test Anime", "en")
        assert result is None

    async def test_no_api_key_returns_none(self):
        settings = make_settings(api_key="")
        with patch("module.conf.settings", settings):
            result = await tvdb_parser("Test Anime", "en")
        assert result is None

    async def test_auth_failure_returns_none(self):
        settings = make_settings()
        client = AsyncMock()
        client.post.side_effect = Exception("Connection refused")

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await tvdb_parser("Test Anime", "en")

        assert result is None

    async def test_empty_search_results_returns_none(self):
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.return_value = make_response({"status": "success", "data": []})

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await tvdb_parser("Unknown Title", "en")

        assert result is None

    async def test_missing_tvdb_id_returns_none(self):
        settings = make_settings()
        bad_search = {"status": "success", "data": [{"name": "No ID Here"}]}
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.return_value = make_response(bad_search)

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await tvdb_parser("Test Anime", "en")

        assert result is None

    async def test_successful_english_lookup(self):
        """Full happy-path with English title — no translation endpoint called."""
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [
            make_response(SEARCH_RESP),
            make_response(EXTENDED_RESP),
            make_response({}),  # poster download
        ]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)), \
             patch("module.parser.analyser.tvdb_parser.save_image", return_value="/posters/test.jpg"):
            result = await tvdb_parser("Test Anime", "en")

        assert isinstance(result, TVDBInfo)
        assert result.id == 12345
        assert result.title == "Test Anime"
        assert result.original_title == "Test Anime"
        assert result.year == "2020"
        assert result.last_season == 2
        assert result.series_status == "Ended"
        assert result.poster_link == "/posters/test.jpg"

    async def test_official_seasons_only(self):
        """DVDs and other alternate orderings are excluded from season_list."""
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [
            make_response(SEARCH_RESP),
            make_response(EXTENDED_RESP),
            make_response({}),  # poster
        ]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)), \
             patch("module.parser.analyser.tvdb_parser.save_image", return_value=None):
            result = await tvdb_parser("Test Anime", "en")

        # EXTENDED_RESP has 2 official + 1 dvd season
        assert len(result.season) == 2
        assert all(s["season"] in (1, 2) for s in result.season)

    async def test_translated_title_for_non_english(self):
        """Japanese lookup fetches translation endpoint and uses translated name."""
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [
            make_response(SEARCH_RESP),
            make_response(EXTENDED_RESP),
            make_response(TRANSLATION_RESP),
            make_response({}),  # poster
        ]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)), \
             patch("module.parser.analyser.tvdb_parser.save_image", return_value=None):
            result = await tvdb_parser("Test Anime", "jp")

        assert result.title == "テストアニメ"
        assert result.original_title == "Test Anime"

    async def test_fallback_to_original_when_no_translation(self):
        """When translation endpoint returns no name, fall back to series name."""
        settings = make_settings()
        no_name_translation = {"status": "success", "data": {"language": "jpn"}}
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [
            make_response(SEARCH_RESP),
            make_response(EXTENDED_RESP),
            make_response(no_name_translation),
            make_response({}),  # poster
        ]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)), \
             patch("module.parser.analyser.tvdb_parser.save_image", return_value=None):
            result = await tvdb_parser("Test Anime", "jp")

        assert result.title == "Test Anime"

    async def test_test_mode_returns_raw_poster_url(self):
        """test=True skips downloading the poster and stores the URL directly."""
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [make_response(SEARCH_RESP), make_response(EXTENDED_RESP)]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await tvdb_parser("Test Anime", "en", test=True)

        assert result.poster_link == "https://example.com/poster.jpg"

    async def test_cache_hit_skips_all_network_calls(self):
        """Second call for the same title+language is served from the cache."""
        cached = TVDBInfo(
            id=99, title="Cached", original_title="Cached",
            season=[], last_season=1, year="2020",
        )
        tvdb_module._tvdb_cache["Test Anime:en"] = cached

        settings = make_settings()
        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client") as mock_client:
            result = await tvdb_parser("Test Anime", "en")

        mock_client.assert_not_called()
        assert result is cached

    async def test_token_reused_on_second_call(self):
        """An existing token is not refreshed unless a 401 is received."""
        tvdb_module._tvdb_token = "existing-token"
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [make_response(SEARCH_RESP), make_response(EXTENDED_RESP)]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            await tvdb_parser("Test Anime", "en", test=True)

        client.post.assert_not_called()

    async def test_no_poster_url_leaves_poster_link_none(self):
        """When the series has no image, poster_link stays None."""
        no_image = {
            "status": "success",
            "data": {
                **EXTENDED_RESP["data"],
                "image": None,
                "seasons": [],
            },
        }
        settings = make_settings()
        client = AsyncMock()
        client.post.return_value = make_response(LOGIN_RESP)
        client.get.side_effect = [make_response(SEARCH_RESP), make_response(no_image)]

        with patch("module.conf.settings", settings), \
             patch("module.parser.analyser.tvdb_parser.get_shared_client", AsyncMock(return_value=client)):
            result = await tvdb_parser("Test Anime", "en", test=True)

        assert result.poster_link is None


# ---------------------------------------------------------------------------
# Integration test — requires a real TVDB API key
# ---------------------------------------------------------------------------


async def test_tvdb_real_query():
    """Live query against api4.thetvdb.com for 進撃の巨人 (Attack on Titan).

    Skipped automatically when tvdb.api_key is not set in config.
    To run: set tvdb.enable=true and tvdb.api_key in config/config_dev.json,
    then: uv run pytest src/test/test_tvdb.py::test_tvdb_real_query -v
    """
    from module.conf import settings as real_settings

    if not real_settings.tvdb.api_key:
        pytest.skip("tvdb.api_key not configured in config/config_dev.json")

    # Clear any cached state from the unit tests above
    tvdb_module._tvdb_token = None
    tvdb_module._tvdb_cache.clear()

    # Force-enable TVDB for this test regardless of the config flag
    real_settings.tvdb.enable = True

    result = await tvdb_parser("進撃の巨人", "jp", test=True)

    assert result is not None, "TVDB returned no match for 進撃の巨人"
    assert result.id > 0
    assert result.title == "進撃の巨人"
    assert result.year == "2013"
    assert result.last_season >= 1
    assert result.series_status is not None
    # Poster URL should be a real TVDB image URL
    assert result.poster_link and result.poster_link.startswith("https://")
