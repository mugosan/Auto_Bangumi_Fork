from module.parser.analyser.tmdb_parser import tmdb_parser


async def test_tmdb_parser():
    bangumi_title = "海盗战记"
    bangumi_year = "2019"
    bangumi_season = 2

    tmdb_info = await tmdb_parser(bangumi_title, "zh", test=True)

    assert tmdb_info.title == "冰海战记"
    assert tmdb_info.year == bangumi_year
    assert tmdb_info.last_season == bangumi_season


async def test_tmdb_parser_cross_references_real_tvdb_id():
    """tmdb_parser also resolves the real TheTVDB series id via TMDB's own
    external_ids endpoint -- no separate TVDB API key/subscription needed."""
    tmdb_info = await tmdb_parser("海盗战记", "zh", test=True)

    assert tmdb_info.tvdb_id == 359274
