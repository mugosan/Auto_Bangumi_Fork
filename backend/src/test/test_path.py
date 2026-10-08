"""Tests for path helpers: save path generation, file classification, parsing."""

import os
from unittest.mock import patch

from module.downloader.path import (
    _truncate_to_byte_budget,
    check_files,
    file_depth,
    gen_save_path,
    is_ep,
    path_to_bangumi,
    rule_name,
    sanitize_path_fragment,
    sibling_season_save_path,
)
from test.factories import make_bangumi

# ---------------------------------------------------------------------------
# sanitize_path_fragment
# ---------------------------------------------------------------------------


class TestSanitizePathFragment:
    def test_replaces_reserved_characters(self):
        assert sanitize_path_fragment('A<B>C:D"E/F\\G|H?I*J') == "A B C D E F G H I J"

    def test_collapses_whitespace_and_strips_trailing_dots(self):
        assert sanitize_path_fragment("Name  ...") == "Name"

    def test_preserves_cjk_and_brackets(self):
        assert (
            sanitize_path_fragment("[Sub] 孤独摇滚！(2022)") == "[Sub] 孤独摇滚！(2022)"
        )

    def test_idempotent(self):
        once = sanitize_path_fragment("Fate/Zero: Part?2")
        assert sanitize_path_fragment(once) == once

    def test_all_reserved_title_falls_back_in_save_path(self):
        """全保留字符的标题不能让保存路径坍缩出空目录层。"""
        bangumi = make_bangumi(official_title="??", year=None)
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "//" not in result
        assert "Unknown Bangumi" in result


# ---------------------------------------------------------------------------
# gen_save_path
# ---------------------------------------------------------------------------


class TestGenSavePath:
    def test_with_year(self):
        """Save path includes (year) when year is set."""
        bangumi = make_bangumi(official_title="My Anime", year="2024", season=2)
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "My Anime (2024)" in result
        assert "Season 2" in result

    def test_reserved_characters_sanitized_in_folder(self):
        """标题里的保留字符不能把保存路径拆成多级目录 (#721)。"""
        bangumi = make_bangumi(official_title="Fate/Zero: Part?2", year="2024")
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "Fate Zero Part 2 (2024)" in result
        assert result.count("/") == 4  # /downloads/Bangumi/<folder>/Season 1

    def test_without_year(self):
        """Save path omits year parentheses when year is None."""
        bangumi = make_bangumi(official_title="My Anime", year=None, season=1)
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "My Anime" in result
        assert "()" not in result
        assert "Season 1" in result

    def test_season_formatting(self):
        """Season is a plain integer, not zero-padded in path."""
        bangumi = make_bangumi(season=10)
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "Season 10" in result

    def test_with_different_base_path(self):
        """Works with different base download path."""
        bangumi = make_bangumi(official_title="Test", year="2025", season=3)
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/mnt/media/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result.startswith("/mnt/media/Bangumi")
        assert "Test (2025)" in result
        assert "Season 3" in result

    def test_folder_tags_real_tvdb_id(self):
        """Folder is tagged [tvdb-<id>] when tvdb_id came from TVDB (via
        TMDB's external_ids cross-reference, id_source='tvdb')."""
        bangumi = make_bangumi(
            official_title="My Anime", year="2024", tvdb_id=267440, id_source="tvdb"
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "My Anime (2024) [tvdb-267440]" in result

    def test_folder_tags_tmdb_fallback_id(self):
        """Folder is tagged [tmdb-<id>] when TMDB had no tvdb cross-reference
        (id_source='tmdb', tvdb_id holds TMDB's own id)."""
        bangumi = make_bangumi(
            official_title="My Anime", year="2024", tvdb_id=12345, id_source="tmdb"
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "My Anime (2024) [tmdb-12345]" in result

    def test_folder_omits_tag_when_no_id(self):
        """No tag at all when tvdb_id is unset (e.g. mikan parser, or no
        TMDB match) -- unlike always showing a bare '[tmdb-None]'."""
        bangumi = make_bangumi(official_title="My Anime", year="2024", tvdb_id=None)
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result == "/downloads/Bangumi/My Anime (2024)/Season 1"
        assert "[" not in result

    def test_movie_layout_omits_season_folder(self):
        """Movies use a flat 'Title (Year)' layout with no Season subfolder."""
        bangumi = make_bangumi(
            official_title="天气之子", year="2019", season=1, episode_type="movie"
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result == "/downloads/Bangumi/天气之子 (2019)"
        assert "Season" not in result

    def test_special_uses_season_zero(self):
        """Specials/OVA/OAD (season=0) land in Season 0, Jellyfin/Plex convention."""
        bangumi = make_bangumi(
            official_title="My Anime", year="2024", season=0, episode_type="special"
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result == "/downloads/Bangumi/My Anime (2024)/Season 0"

    def test_gen_save_path_regular_offset_to_zero_reverts_to_original_season(self):
        """普通剧集 season+offset 落到 0 时回退原季号：Season 0 会被
        Plex/Jellyfin 当作特别篇，只有 special 类型才允许落入。"""
        bangumi = make_bangumi(
            official_title="My Anime",
            year="2024",
            season=1,
            season_offset=-1,
            episode_type="episode",
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result == "/downloads/Bangumi/My Anime (2024)/Season 1"

    def test_gen_save_path_special_offset_to_zero_lands_in_season_zero(self):
        """特别篇（special）经偏移落到第 0 季是合法的（Jellyfin/Plex 惯例）。"""
        bangumi = make_bangumi(
            official_title="My Anime",
            year="2024",
            season=1,
            season_offset=-1,
            episode_type="special",
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result == "/downloads/Bangumi/My Anime (2024)/Season 0"

    def test_gen_save_path_special_offset_below_zero_reverts_to_original_season(self):
        """特别篇偏移到负季号仍属非法配置，回退原季号。"""
        bangumi = make_bangumi(
            official_title="My Anime",
            year="2024",
            season=0,
            season_offset=-1,
            episode_type="special",
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result == "/downloads/Bangumi/My Anime (2024)/Season 0"


# ---------------------------------------------------------------------------
# _truncate_to_byte_budget
# ---------------------------------------------------------------------------


class TestTruncateToByteBudget:
    def test_under_budget_is_unchanged(self):
        assert _truncate_to_byte_budget("My Anime", 150) == "My Anime"

    def test_ascii_truncated_to_exact_byte_count(self):
        result = _truncate_to_byte_budget("A" * 200, 10)
        assert result == "A" * 10
        assert len(result.encode("utf-8")) == 10

    def test_never_splits_a_multibyte_character(self):
        """Each CJK character below is 3 UTF-8 bytes -- a budget that isn't
        a multiple of 3 must not produce a half-decoded character or raise."""
        name = "追放されたチート付与魔術師" * 5
        result = _truncate_to_byte_budget(name, 100)
        assert len(result.encode("utf-8")) <= 100
        # Must still be valid, round-trippable UTF-8 -- a split multi-byte
        # sequence would have been silently dropped, not mangled in place.
        assert result.encode("utf-8").decode("utf-8") == result

    def test_exact_boundary_is_kept_whole(self):
        name = "あ" * 10  # 3 bytes each -> 30 bytes total
        assert _truncate_to_byte_budget(name, 30) == name
        assert _truncate_to_byte_budget(name, 29) == "あ" * 9


# ---------------------------------------------------------------------------
# _media_folder / gen_save_path long-title truncation
# ---------------------------------------------------------------------------


class TestGenSavePathLongTitleTruncation:
    LONG_TITLE = (
        "追放されたチート付与魔術師は気ままなセカンドライフを謳歌する。"
        "～俺は武器だけじゃなく、あらゆるものに『強化ポイント』を付与できるし、"
        "俺の意思でいつでも効果を解除できるけど、残った人たち大丈夫？～"
    )

    def test_long_cjk_title_exceeds_filesystem_limit_before_the_fix(self):
        """Establishes the bug is real before testing the fix: this is the
        exact title that produced a 313-byte folder name and a torrent
        stuck at 0% because qBittorrent's mkdir was rejected outright."""
        folder_name = f"{self.LONG_TITLE} (2026) [tvdb-473642]"
        assert len(folder_name.encode("utf-8")) > 255  # ext4's NAME_MAX

    def test_long_title_is_truncated_to_fit_the_configured_budget(self):
        bangumi = make_bangumi(
            official_title=self.LONG_TITLE,
            year="2026",
            tvdb_id=473642,
            id_source="tvdb",
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        folder_name = result.split("/downloads/Bangumi/", 1)[1].split("/Season")[0]
        assert len(folder_name.encode("utf-8")) <= 150

    def test_id_tag_always_survives_truncation_intact(self):
        """The [tvdb-N]/[tmdb-N] tag is short and load-bearing (Plex/HAMA
        matching, and path_to_bangumi's reverse-parse) -- truncation must
        always shorten the title, never clip into the tag."""
        bangumi = make_bangumi(
            official_title=self.LONG_TITLE,
            year="2026",
            tvdb_id=473642,
            id_source="tvdb",
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert result.endswith("[tvdb-473642]/Season 1")

    def test_truncated_folder_still_round_trips_through_path_to_bangumi(self):
        """The id tag survives truncation well enough that path_to_bangumi
        (used to reverse-derive the show name for "advance" rename mode)
        still strips it correctly."""
        bangumi = make_bangumi(
            official_title=self.LONG_TITLE,
            year="2026",
            tvdb_id=473642,
            id_source="tvdb",
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            save_path = gen_save_path(bangumi)

            name, season = path_to_bangumi(save_path)

        assert season == 1
        assert "[tvdb-473642]" not in name
        assert len(name.encode("utf-8")) <= 150

    def test_short_title_with_tag_is_never_truncated(self):
        """A normal-length title must be completely unaffected."""
        bangumi = make_bangumi(
            official_title="My Anime", year="2024", tvdb_id=267440, id_source="tvdb"
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "My Anime (2024) [tvdb-267440]" in result


class TestEffectiveMaxFolderNameBytes:
    """The real per-mount limit (`os.pathconf`, same as `getconf NAME_MAX
    <path>`) takes precedence over the configured default whenever it can
    be detected -- a plain filesystem reporting the standard 255 should
    not be truncated down to some arbitrary lower default "just in case",
    and a genuinely restrictive one (e.g. an encrypted-home seedbox, which
    reports its own reduced limit through this same mechanism) should be
    respected even if the configured default is set higher."""

    def test_detected_limit_caps_an_overly_generous_configured_value(self, tmp_path):
        """tmp_path is a real directory on this machine's real filesystem,
        so os.pathconf against it returns the real OS-reported NAME_MAX --
        exercising actual detection, not a mocked stand-in for it."""
        from module.downloader import path as path_module

        real_limit = os.pathconf(str(tmp_path), "PC_NAME_MAX")
        long_title = "あ" * 200  # 600 bytes -- certainly past any real limit

        bangumi = make_bangumi(official_title=long_title, year=None, tvdb_id=None)
        with (
            patch.object(path_module, "_name_max_cache", {}),
            patch("module.downloader.path.settings") as mock_settings,
        ):
            mock_settings.downloader.path = str(tmp_path)
            # Deliberately far above what any real filesystem allows, to
            # prove detection -- not this config value -- is what bites.
            mock_settings.downloader.max_folder_name_bytes = 10_000
            result = gen_save_path(bangumi)

        folder_name = result[len(str(tmp_path)) + 1 :].split("/Season")[0]
        assert len(folder_name.encode("utf-8")) <= real_limit

    def test_falls_back_to_configured_value_when_root_does_not_exist(self):
        """A download root that hasn't been created yet (or isn't
        reachable) can't be queried -- must fall back to the configured
        value instead of raising."""
        from module.downloader import path as path_module

        bangumi = make_bangumi(official_title="My Anime", year="2024", tvdb_id=None)
        with (
            patch.object(path_module, "_name_max_cache", {}),
            patch("module.downloader.path.settings") as mock_settings,
        ):
            mock_settings.downloader.path = "/this/path/does/not/exist/anywhere"
            mock_settings.downloader.max_folder_name_bytes = 150
            result = gen_save_path(bangumi)

        assert "My Anime (2024)" in result


# ---------------------------------------------------------------------------
# rule_name
# ---------------------------------------------------------------------------


class TestRuleName:
    def test_without_group_tag(self):
        """Rule name without group tag is just title and season."""
        bangumi = make_bangumi(official_title="My Anime", season=1, group_name="Sub")
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.bangumi_manage.group_tag = False
            result = rule_name(bangumi)

        assert result == "My Anime S1"

    def test_with_group_tag(self):
        """Rule name with group tag includes [group] prefix."""
        bangumi = make_bangumi(
            official_title="My Anime", season=2, group_name="SubGroup"
        )
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.bangumi_manage.group_tag = True
            result = rule_name(bangumi)

        assert result == "[SubGroup] My Anime S2"


# ---------------------------------------------------------------------------
# check_files
# ---------------------------------------------------------------------------


class TestCheckFiles:
    def test_separates_media_and_subtitles(self):
        """Media files (.mp4/.mkv) and subtitle files (.ass/.srt) are separated."""
        files = [
            {"name": "episode01.mkv"},
            {"name": "episode01.ass"},
            {"name": "episode02.mp4"},
            {"name": "episode02.srt"},
        ]
        media, subs = check_files(files)

        assert len(media) == 2
        assert "episode01.mkv" in media
        assert "episode02.mp4" in media
        assert len(subs) == 2
        assert "episode01.ass" in subs
        assert "episode02.srt" in subs

    def test_ignores_other_extensions(self):
        """Files with non-media, non-subtitle extensions are ignored."""
        files = [
            {"name": "episode.mkv"},
            {"name": "readme.txt"},
            {"name": "info.nfo"},
            {"name": "cover.jpg"},
        ]
        media, subs = check_files(files)

        assert len(media) == 1
        assert len(subs) == 0

    def test_case_insensitive_extensions(self):
        """Extension matching is case-insensitive."""
        files = [
            {"name": "episode.MKV"},
            {"name": "episode.MP4"},
            {"name": "sub.ASS"},
            {"name": "sub.SRT"},
        ]
        media, subs = check_files(files)

        assert len(media) == 2
        assert len(subs) == 2

    def test_empty_file_list(self):
        """Empty file list returns empty lists."""
        media, subs = check_files([])
        assert media == []
        assert subs == []

    def test_nested_paths(self):
        """Files in subdirectories are handled correctly."""
        files = [
            {"name": "Season 1/episode01.mkv"},
            {"name": "Subs/episode01.ass"},
        ]
        media, subs = check_files(files)

        assert len(media) == 1
        assert len(subs) == 1


# ---------------------------------------------------------------------------
# path_to_bangumi
# ---------------------------------------------------------------------------


class TestPathToBangumi:
    def test_extracts_name_and_season(self):
        """Parses save_path to extract bangumi name and season number."""
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            name, season = path_to_bangumi(
                "/downloads/Bangumi/My Anime (2024)/Season 2"
            )

        assert name == "My Anime (2024)"
        assert season == 2

    def test_season_1_default(self):
        """When no Season pattern found, defaults to season 1."""
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            name, season = path_to_bangumi("/downloads/Bangumi/My Anime (2024)")

        assert name == "My Anime (2024)"
        assert season == 1

    def test_s_prefix_pattern(self):
        """Recognizes S01 style season naming."""
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            name, season = path_to_bangumi("/downloads/Bangumi/Anime/S03")

        assert season == 3

    def test_strips_tvdb_id_tag_from_folder_name(self):
        """Regression for #1042: _media_folder() appends "[tvdb-N]"/"[tmdb-N]"
        to the folder for Plex/HAMA matching. "advance" rename mode feeds
        this name straight into every episode's filename, so the tag must
        not leak into bangumi_name or it ends up baked into every file."""
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            name, season = path_to_bangumi(
                "/downloads/Bangumi/My Anime (2024) [tvdb-457532]/Season 2"
            )

        assert name == "My Anime (2024)"
        assert season == 2

    def test_strips_tmdb_id_tag_from_folder_name(self):
        """Same as above for the tmdb-id fallback tag (no tvdb match)."""
        with patch("module.downloader.path.settings") as mock_settings:
            mock_settings.downloader.path = "/downloads/Bangumi"
            mock_settings.downloader.max_folder_name_bytes = 150
            name, season = path_to_bangumi(
                "/downloads/Bangumi/My Anime (2024) [tmdb-12345]/Season 1"
            )

        assert name == "My Anime (2024)"


# ---------------------------------------------------------------------------
# sibling_season_save_path
# ---------------------------------------------------------------------------


class TestSiblingSeasonSavePath:
    def test_swaps_season_component(self):
        assert (
            sibling_season_save_path("/downloads/Bangumi/My Anime (2024)/Season 2", 1)
            == "/downloads/Bangumi/My Anime (2024)/Season 1"
        )

    def test_swaps_s_prefix_component(self):
        assert (
            sibling_season_save_path("/downloads/Bangumi/Anime/S03", 1)
            == "/downloads/Bangumi/Anime/Season 1"
        )

    def test_keeps_id_tagged_folder_name_intact(self):
        assert (
            sibling_season_save_path(
                "/downloads/Bangumi/My Anime (2024) [tvdb-457532]/Season 2", 1
            )
            == "/downloads/Bangumi/My Anime (2024) [tvdb-457532]/Season 1"
        )

    def test_no_season_component_returns_unchanged(self):
        """Movie layout has no "Season N" folder -- nothing to swap."""
        path = "/downloads/Bangumi/天气之子 (2019)"
        assert sibling_season_save_path(path, 1) == path

    def test_same_season_is_a_no_op(self):
        path = "/downloads/Bangumi/My Anime (2024)/Season 2"
        assert sibling_season_save_path(path, 2) == path


# ---------------------------------------------------------------------------
# is_ep / file_depth
# ---------------------------------------------------------------------------


class TestIsEp:
    def test_shallow_file(self):
        """File at depth 1 (just filename) is considered an episode."""
        assert is_ep("episode.mkv") is True

    def test_one_folder_deep(self):
        """File at depth 2 (one folder) is still an episode."""
        assert is_ep("Season 1/episode.mkv") is True

    def test_too_deep(self):
        """File at depth 3+ is NOT considered an episode."""
        assert is_ep("a/b/episode.mkv") is False

    def test_file_depth(self):
        """file_depth returns correct part count."""
        assert file_depth("file.mkv") == 1
        assert file_depth("a/file.mkv") == 2
        assert file_depth("a/b/c/file.mkv") == 4
