"""Mikan-sourced subscriptions never got a year/tvdb_id/id_source, unlike
tmdb-sourced ones -- Mikan's homepage scrape only gives title/poster. This
cross-references the resolved title against TMDB for those three fields
only, keeping Mikan's own title/season (see collector.py's mirrored change
for the equivalent search-subscribe path)."""

from unittest.mock import AsyncMock, patch

from module.models import Torrent
from module.rss.analyser import RSSAnalyser
from test.factories import make_bangumi, make_rss_item


async def test_mikan_branch_fills_year_and_tvdb_but_keeps_title_and_season():
    bangumi = make_bangumi(official_title="Mikan Title", season=1, year=None)
    rss = make_rss_item(parser="mikan")
    torrent = Torrent(name="test", homepage="https://mikanani.me/Home/Episode/abc")

    tmdb_result = ("TMDB Title", 2, "2019", "tmdb-poster.jpg", 359274, "tvdb")
    with (
        patch(
            "module.rss.analyser.TitleParser.mikan_parser",
            AsyncMock(return_value=("mikan-poster.jpg", "Mikan Title")),
        ),
        patch(
            "module.rss.analyser.TitleParser.tmdb_parser",
            AsyncMock(return_value=tmdb_result),
        ) as mock_tmdb,
    ):
        await RSSAnalyser().official_title_parser(bangumi, rss, torrent)

    mock_tmdb.assert_awaited_once()
    assert bangumi.official_title == "Mikan Title"
    assert bangumi.season == 1
    assert bangumi.poster_link == "mikan-poster.jpg"
    assert bangumi.year == "2019"
    assert bangumi.tvdb_id == 359274
    assert bangumi.id_source == "tvdb"


async def test_mikan_branch_tolerates_tmdb_failure():
    """A TMDB hiccup must not break an otherwise-successful Mikan resolution."""
    bangumi = make_bangumi(official_title="Mikan Title", season=1, year=None)
    rss = make_rss_item(parser="mikan")
    torrent = Torrent(name="test", homepage="https://mikanani.me/Home/Episode/abc")

    with (
        patch(
            "module.rss.analyser.TitleParser.mikan_parser",
            AsyncMock(return_value=("mikan-poster.jpg", "Mikan Title")),
        ),
        patch(
            "module.rss.analyser.TitleParser.tmdb_parser",
            AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        await RSSAnalyser().official_title_parser(bangumi, rss, torrent)

    assert bangumi.official_title == "Mikan Title"
    assert bangumi.poster_link == "mikan-poster.jpg"
    assert bangumi.year is None
    assert bangumi.tvdb_id is None


async def test_fetch_poster_false_skips_mikan_and_tmdb_entirely():
    bangumi = make_bangumi(official_title="Mikan Title", season=1, year=None)
    rss = make_rss_item(parser="mikan")
    torrent = Torrent(name="test", homepage="https://mikanani.me/Home/Episode/abc")

    with (
        patch(
            "module.rss.analyser.TitleParser.mikan_parser", AsyncMock()
        ) as mock_mikan,
        patch("module.rss.analyser.TitleParser.tmdb_parser", AsyncMock()) as mock_tmdb,
    ):
        await RSSAnalyser().official_title_parser(
            bangumi, rss, torrent, fetch_poster=False
        )

    mock_mikan.assert_not_awaited()
    mock_tmdb.assert_not_awaited()
