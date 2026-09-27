"""Tests for season_resolver: TMDB-air-date-based season disambiguation.

Mirrors test_tmdb.py's pattern -- the async lookup test patches
RequestContent.get_json with fixture data so the suite stays deterministic
and offline; pick_season_by_air_date itself is pure and needs no mocking.
"""

import datetime
import importlib

import pytest

from module.parser.analyser.season_resolver import (
    pick_season_by_air_date,
    reset_cache,
    resolve_episode_air_dates_by_season,
)

tmdb_parser_module = importlib.import_module("module.parser.analyser.tmdb_parser")


@pytest.fixture(autouse=True)
def _clear_tmdb_caches():
    """Both tmdb_parser's show-level cache and season_resolver's per-season
    episode-list cache are bare module globals with no test-scoped
    isolation -- clear them before each test so one test's fake fixture data
    can't leak into the next (same tmdb id/season keys recur across tests)."""
    tmdb_parser_module._tmdb_cache.clear()
    reset_cache()


# ---------------------------------------------------------------------------
# pick_season_by_air_date
# ---------------------------------------------------------------------------


class TestPickSeasonByAirDate:
    def test_single_close_candidate_wins(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2022, 1, 10),
                air_dates_by_season={1: datetime.date(2022, 1, 5)},
            )
            == 1
        )

    def test_can_confirm_the_current_season_too(self):
        """The whole point: this must be able to say "yes, current season"
        just as readily as "no, an earlier one" -- otherwise a genuine
        duplicate would still get relocated."""
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2026, 1, 6),
                air_dates_by_season={
                    1: datetime.date(2022, 1, 5),
                    2: datetime.date(2026, 1, 5),
                },
            )
            == 2
        )

    def test_correctly_picks_earlier_season_over_current(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2022, 1, 6),
                air_dates_by_season={
                    1: datetime.date(2022, 1, 5),
                    2: datetime.date(2026, 1, 5),
                },
            )
            == 1
        )

    def test_nothing_within_range_is_inconclusive(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2024, 1, 1),
                air_dates_by_season={
                    1: datetime.date(2022, 1, 5),
                    2: datetime.date(2026, 1, 5),
                },
            )
            is None
        )

    def test_exact_tie_is_inconclusive(self):
        """Never guess when two seasons are equally close -- e.g. a split
        cour that aired again exactly N days apart in two different years."""
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2024, 1, 5),
                air_dates_by_season={
                    1: datetime.date(2024, 1, 1),
                    2: datetime.date(2024, 1, 9),
                },
                max_gap_days=10,
            )
            is None
        )

    def test_missing_air_date_is_ignored_not_treated_as_a_match(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2022, 1, 10),
                air_dates_by_season={1: datetime.date(2022, 1, 5), 2: None},
            )
            == 1
        )

    def test_empty_map_is_inconclusive(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2022, 1, 10), air_dates_by_season={}
            )
            is None
        )

    def test_boundary_at_max_gap_days_is_included(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2022, 2, 19),  # exactly 45 days later
                air_dates_by_season={1: datetime.date(2022, 1, 5)},
                max_gap_days=45,
            )
            == 1
        )

    def test_one_day_past_max_gap_is_inconclusive(self):
        assert (
            pick_season_by_air_date(
                pub_date=datetime.date(2022, 2, 20),  # 46 days later
                air_dates_by_season={1: datetime.date(2022, 1, 5)},
                max_gap_days=45,
            )
            is None
        )


# ---------------------------------------------------------------------------
# resolve_episode_air_dates_by_season
# ---------------------------------------------------------------------------

_SHOW_INFO = {
    "genres": [{"id": 16, "name": "Animation"}],
    "name": "尼古喵喵",
    "original_name": "Nikogami",
    "first_air_date": "2022-01-05",
    "status": "Ended",
    "poster_path": "/poster.jpg",
    "seasons": [
        {
            "name": "第 1 季",
            "air_date": "2022-01-05",
            "poster_path": "/s1.jpg",
            "season_number": 1,
            "episode_count": 12,
        },
        {
            "name": "第 2 季",
            "air_date": "2026-01-05",
            "poster_path": "/s2.jpg",
            "season_number": 2,
            "episode_count": 12,
        },
    ],
}


async def _fake_get_json(url: str):
    if "/search/tv" in url:
        return {"results": [{"id": 82684}]}
    if "/external_ids" in url:
        return {"tvdb_id": None}
    if "/season/1?" in url:
        return {"episodes": [{"episode_number": 1, "air_date": "2022-01-05"}]}
    if "/season/2?" in url:
        return {"episodes": [{"episode_number": 1, "air_date": "2026-01-05"}]}
    return _SHOW_INFO


class TestResolveEpisodeAirDatesBySeason:
    async def test_returns_air_date_per_candidate_season(self, mocker):
        mocker.patch.object(
            tmdb_parser_module.RequestContent, "get_json", side_effect=_fake_get_json
        )
        result = await resolve_episode_air_dates_by_season(
            official_title="尼古喵喵", language="zh", episode=1, seasons=[1, 2]
        )

        assert result == {
            1: datetime.date(2022, 1, 5),
            2: datetime.date(2026, 1, 5),
        }

    async def test_episode_absent_from_season_maps_to_none(self, mocker):
        async def fake_get_json(url: str):
            if "/season/2?" in url:
                return {"episodes": []}
            return await _fake_get_json(url)

        mocker.patch.object(
            tmdb_parser_module.RequestContent, "get_json", side_effect=fake_get_json
        )
        result = await resolve_episode_air_dates_by_season(
            official_title="尼古喵喵", language="zh", episode=1, seasons=[1, 2]
        )

        assert result[1] == datetime.date(2022, 1, 5)
        assert result[2] is None

    async def test_no_tmdb_match_returns_empty_map(self, mocker):
        async def fake_get_json_no_match(url: str):
            if "/search/tv" in url:
                return {"results": []}
            return None

        mocker.patch.object(
            tmdb_parser_module.RequestContent,
            "get_json",
            side_effect=fake_get_json_no_match,
        )
        result = await resolve_episode_air_dates_by_season(
            official_title="Unknown Show", language="zh", episode=1, seasons=[1, 2]
        )

        assert result == {}
