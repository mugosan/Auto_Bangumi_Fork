import logging
from collections import OrderedDict
from dataclasses import dataclass

from module.network.request_url import get_shared_client
from module.utils import save_image

logger = logging.getLogger(__name__)

TVDB_URL = "https://api4.thetvdb.com/v4"

_TVDB_CACHE_MAX = 512
_tvdb_cache: OrderedDict[str, "TVDBInfo | None"] = OrderedDict()
_tvdb_token: str | None = None

# Maps the app's language keys to TVDB ISO 639-3 language codes
TVDB_LANGUAGE = {"zh": "zho", "jp": "jpn", "en": "eng"}


@dataclass
class TVDBInfo:
    id: int
    title: str
    original_title: str
    season: list[dict]
    last_season: int
    year: str
    poster_link: str = None
    series_status: str = None
    season_episode_counts: dict[int, int] = None


async def _authenticate(api_key: str) -> str | None:
    """POST to TVDB /login and return a bearer token. Caches the token module-wide."""
    global _tvdb_token
    if _tvdb_token:
        return _tvdb_token
    client = await get_shared_client()
    try:
        resp = await client.post(
            f"{TVDB_URL}/login",
            json={"apikey": api_key},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            _tvdb_token = data["data"]["token"]
            logger.debug("[TVDB] Authentication successful")
            return _tvdb_token
        logger.warning("[TVDB] Unexpected login response: %s", data.get("status"))
    except Exception as e:
        logger.warning("[TVDB] Authentication failed: %s", e)
    return None


async def _get(token: str, path: str, params: dict = None) -> dict | None:
    """Authenticated GET against the TVDB v4 API.

    Returns the parsed JSON dict, or None on failure. Clears the cached token
    on 401 so the next call will re-authenticate.
    """
    global _tvdb_token
    client = await get_shared_client()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        resp = await client.get(f"{TVDB_URL}{path}", headers=headers, params=params)
        if resp.status_code == 401:
            logger.warning("[TVDB] Token expired — clearing for re-authentication")
            _tvdb_token = None
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("[TVDB] Request failed for %s: %s", path, e)
        return None


def _official_seasons(seasons: list[dict]) -> list[dict]:
    """Return only numbered official seasons (excludes specials and alt orderings)."""
    return [
        s for s in seasons
        if s.get("type", {}).get("type") == "official" and s.get("number", 0) > 0
    ]


def _last_aired_season(seasons: list[dict]) -> int:
    """Return the highest official season number that has already started airing."""
    import datetime

    current_year = datetime.date.today().year
    valid = [
        s for s in seasons
        if s.get("year") and int(s.get("year") or 0) <= current_year
    ]
    if not valid:
        return len(seasons) or 1
    return max(int(s["number"]) for s in valid)


async def tvdb_parser(title: str, language: str, test: bool = False) -> TVDBInfo | None:
    from module.conf import settings

    if not settings.tvdb.enable:
        logger.debug("[TVDB] TVDB lookup is disabled")
        return None

    api_key = settings.tvdb.api_key
    if not api_key:
        logger.warning("[TVDB] No API key configured — set tvdb.api_key in settings")
        return None

    cache_key = f"{title}:{language}"
    if cache_key in _tvdb_cache:
        return _tvdb_cache[cache_key]

    lang_code = TVDB_LANGUAGE.get(language, "eng")

    token = await _authenticate(api_key)
    if not token:
        return None

    # Search for the series
    search_data = await _get(token, "/search", {"query": title, "type": "series"})
    if not search_data or not search_data.get("data"):
        logger.warning("[TVDB] No results for '%s'", title)
        _tvdb_cache[cache_key] = None
        return None

    results = search_data["data"]
    first = results[0]
    tvdb_id = int(first.get("tvdb_id") or 0)
    if not tvdb_id:
        logger.warning("[TVDB] Could not extract tvdb_id from search results for '%s'", title)
        _tvdb_cache[cache_key] = None
        return None

    logger.debug("[TVDB] Matched '%s' → TVDB id %s", title, tvdb_id)

    # Fetch extended series info (includes seasons list)
    ext_data = await _get(token, f"/series/{tvdb_id}/extended", {"meta": "translations", "short": "false"})
    if not ext_data or not ext_data.get("data"):
        _tvdb_cache[cache_key] = None
        return None

    series = ext_data["data"]

    # Resolve title: try the requested language translation first, fall back to series name
    official_title = series.get("name", title)
    if lang_code != "eng":
        trans_data = await _get(token, f"/series/{tvdb_id}/translations/{lang_code}")
        if trans_data and trans_data.get("data"):
            translated = trans_data["data"].get("name")
            if translated:
                official_title = translated

    original_title = series.get("name", title)
    year = (series.get("firstAired") or "").split("-")[0]
    series_status = (series.get("status") or {}).get("name")

    # Build season list from official (numbered) seasons
    all_seasons = series.get("seasons") or []
    official = _official_seasons(all_seasons)
    season_list = [
        {
            "season": s.get("number"),
            "year": s.get("year"),
            "image": s.get("image"),
        }
        for s in official
    ]
    last_season = _last_aired_season(official) if official else 1

    # Download poster image
    poster_url = series.get("image") or (official[0].get("image") if official else None)
    poster_link = None
    if poster_url:
        if not test:
            client = await get_shared_client()
            try:
                img_resp = await client.get(poster_url)
                img_resp.raise_for_status()
                poster_link = save_image(img_resp.content, "jpg")
            except Exception as e:
                logger.warning("[TVDB] Failed to download poster: %s", e)
        else:
            poster_link = poster_url

    result = TVDBInfo(
        id=tvdb_id,
        title=official_title,
        original_title=original_title,
        season=season_list,
        last_season=last_season,
        year=year,
        poster_link=poster_link,
        series_status=series_status,
    )

    if len(_tvdb_cache) >= _TVDB_CACHE_MAX:
        _tvdb_cache.popitem(last=False)
    _tvdb_cache[cache_key] = result
    return result


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(tvdb_parser("魔法禁书目录", "zh")))
