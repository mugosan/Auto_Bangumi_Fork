from unittest.mock import AsyncMock, patch

import pytest

from module.conf import settings
from module.parser.analyser.tmdb_parser import TMDBInfo
from module.parser.title_parser import TitleParser


class TestTitleParser:
    async def test_tmdb_parser_prefers_real_tvdb_id(self):
        """When TMDB's external_ids cross-reference has a tvdb_id, that's
        the id used, and id_source records it came from TVDB."""
        info = TMDBInfo(
            id=88803,
            title="冰海战记",
            original_title="Vinland Saga",
            season=[],
            last_season=2,
            year="2019",
            poster_link="poster.jpg",
            tvdb_id=359274,
        )
        with patch(
            "module.parser.title_parser.tmdb_parser", AsyncMock(return_value=info)
        ):
            meta_id, id_source, title, season, year, poster_link = (
                await TitleParser.tmdb_parser("Vinland Saga", 1, "zh")
            )

        assert meta_id == 359274
        assert id_source == "tvdb"
        assert title == "冰海战记"

    async def test_tmdb_parser_falls_back_to_tmdb_id_without_cross_reference(self):
        """When TMDB has no tvdb_id for the show, fall back to the TMDB id
        itself, with id_source recording that it's not a real TVDB id."""
        info = TMDBInfo(
            id=88803,
            title="冰海战记",
            original_title="Vinland Saga",
            season=[],
            last_season=2,
            year="2019",
            poster_link="poster.jpg",
            tvdb_id=None,
        )
        with patch(
            "module.parser.title_parser.tmdb_parser", AsyncMock(return_value=info)
        ):
            meta_id, id_source, title, season, year, poster_link = (
                await TitleParser.tmdb_parser("Vinland Saga", 1, "zh")
            )

        assert meta_id == 88803
        assert id_source == "tmdb"

    async def test_tmdb_parser_no_match_returns_tmdb_source(self):
        with patch(
            "module.parser.title_parser.tmdb_parser", AsyncMock(return_value=None)
        ):
            meta_id, id_source, title, season, year, poster_link = (
                await TitleParser.tmdb_parser("Unknown Show", 1, "zh")
            )

        assert meta_id is None
        assert id_source == "tmdb"
        assert title == "Unknown Show"

    async def test_parse_without_openai(self):
        text = "[梦蓝字幕组]New Doraemon 哆啦A梦新番[747][2023.02.25][AVC][1080P][GB_JP][MP4]"
        result = await TitleParser.raw_parser(text)
        assert result.group_name == "梦蓝字幕组"
        assert result.title_raw == "New Doraemon"
        assert result.dpi == "1080P"
        assert result.season == 1
        assert result.subtitle == "GB_JP"

    @pytest.mark.skipif(
        not settings.experimental_openai.enable,
        reason="OpenAI is not enabled in settings",
    )
    async def test_parse_with_openai(self):
        text = "[梦蓝字幕组]New Doraemon 哆啦A梦新番[747][2023.02.25][AVC][1080P][GB_JP][MP4]"
        result = await TitleParser.raw_parser(text)
        assert result.group_name == "梦蓝字幕组"
        assert result.title_raw == "New Doraemon"
        assert result.dpi == "1080P"
        assert result.season == 1
        assert result.subtitle == "GB_JP"
