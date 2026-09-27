"""Resolve which season an episode actually belongs to, using TMDB air dates.

torrent_parser() defaults an untagged release's season to whichever season
the bangumi is currently tracking, since RSS matching is by show name alone
and can't tell a genuine new episode from an old season's rerun leaking into
the same feed. When that guess collides with an already-organized episode,
this module gives the renamer a real, evidence-based tiebreaker: TMDB knows
when each season's episode N actually aired, and the RSS feed's own
<pubDate> says when this specific release showed up. Comparing the two tells
apart "this really is the current season" (pub_date close to the current
season's air date) from "this is an old season" (pub_date close to an
earlier one) -- something no amount of guessing from occupancy alone can do.
"""

from __future__ import annotations

import datetime
from collections import OrderedDict

from module.network import RequestContent

from .tmdb_parser import get_season_episode_air_dates, tmdb_parser

# This now runs unconditionally for every not-yet-renamed episode of a
# multi-season bangumi (see Renamer._try_correct_season_before_claim), not
# just ones that happen to collide -- a whole batch of episodes for the same
# show+season would otherwise each re-fetch the identical episode list.
# tmdb_parser() already caches the show-level lookup; this caches the
# per-season episode list the same way.
_SEASON_AIR_DATES_CACHE_MAX = 256
_season_air_dates_cache: "OrderedDict[tuple[int, int, str], list[dict]]" = OrderedDict()


def reset_cache() -> None:
    """Clear the per-season air-date cache. Call after a config reload that
    could change the TMDB endpoint/key, same as tmdb_parser.reset_cache()."""
    _season_air_dates_cache.clear()


async def _cached_season_episode_air_dates(
    tv_id: int, season_number: int, language: str, req: RequestContent
) -> list[dict]:
    key = (tv_id, season_number, language)
    if key in _season_air_dates_cache:
        _season_air_dates_cache.move_to_end(key)
        return _season_air_dates_cache[key]
    episodes = await get_season_episode_air_dates(tv_id, season_number, language, req)
    _season_air_dates_cache[key] = episodes
    if len(_season_air_dates_cache) > _SEASON_AIR_DATES_CACHE_MAX:
        _season_air_dates_cache.popitem(last=False)
    return episodes


async def resolve_episode_air_dates_by_season(
    *,
    official_title: str,
    language: str,
    episode: int,
    seasons: list[int],
) -> dict[int, datetime.date | None]:
    """TMDB's official air date for `episode` in each of `seasons`.

    Returns {} when the show itself can't be matched on TMDB at all (no api
    key, network error, no match) -- callers treat that the same as
    "inconclusive" and fall back to a weaker signal. A season present in the
    result with a None value means TMDB matched the show but has no such
    episode number in that season (e.g. it's shorter than others).
    """
    tmdb_info = await tmdb_parser(official_title, language)
    if tmdb_info is None:
        return {}
    result: dict[int, datetime.date | None] = {}
    async with RequestContent() as req:
        for season in seasons:
            episodes = await _cached_season_episode_air_dates(
                tmdb_info.id, season, language, req
            )
            match = next((e for e in episodes if e["episode_number"] == episode), None)
            result[season] = match["air_date"] if match else None
    return result


def pick_season_by_air_date(
    *,
    pub_date: datetime.date,
    air_dates_by_season: dict[int, datetime.date | None],
    max_gap_days: int = 45,
) -> int | None:
    """Pick whichever season's expected air date for this episode is
    closest to when the release actually showed up in the RSS feed.

    Only returns a season when there's a clear, close match: closest
    candidate must be within `max_gap_days` of `pub_date`, and strictly
    closer than every other candidate (an exact tie is left unresolved
    rather than picked arbitrarily). Returns None -- "inconclusive" -- when
    nothing is close enough or two seasons are equally close, so the caller
    can fall back to a weaker signal instead of guessing.

    `max_gap_days` defaults to a season-scale window: generous enough to
    absorb simulcast delays and re-encodes landing weeks after the original
    broadcast, but far short of the usual gap between two different
    seasons' premieres.
    """
    gaps = {
        season: abs((air_date - pub_date).days)
        for season, air_date in air_dates_by_season.items()
        if air_date is not None
    }
    within_range = {s: gap for s, gap in gaps.items() if gap <= max_gap_days}
    if not within_range:
        return None
    best_season = min(within_range, key=within_range.get)
    best_gap = within_range[best_season]
    if any(
        season != best_season and gap == best_gap
        for season, gap in within_range.items()
    ):
        return None
    return best_season
