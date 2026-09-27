from module.manager.revision_policy import (
    find_fallback_season,
    is_strict_upgrade,
    parse_revision_identity,
    replacement_staged_path,
    same_release_identity,
)

V1 = "[ANi] 尼古喵喵 - 01 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4"
V2 = "[ANi] 尼古喵喵 - 01 [V2][1080P][Baha][WEB-DL][AAC AVC][CHT].mp4"


def test_mikan_classic_v2_is_a_strict_upgrade():
    old = parse_revision_identity(V1, bangumi_id=42, default_season=1)
    new = parse_revision_identity(V2, bangumi_id=42, default_season=1)

    assert old is not None
    assert new is not None
    assert old.revision == 1
    assert new.revision == 2
    assert same_release_identity(old, new)
    assert is_strict_upgrade(old, new)


def test_cross_group_or_resolution_is_not_a_strict_upgrade():
    old = parse_revision_identity(V1, bangumi_id=42, default_season=1)
    other_group = parse_revision_identity(
        V2.replace("[ANi]", "[Other]"), bangumi_id=42, default_season=1
    )
    other_resolution = parse_revision_identity(
        V2.replace("1080P", "720P"), bangumi_id=42, default_season=1
    )

    assert old is not None
    assert other_group is not None
    assert other_resolution is not None
    assert not is_strict_upgrade(old, other_group)
    assert not is_strict_upgrade(old, other_resolution)


def test_missing_bangumi_group_or_resolution_is_ineligible():
    assert parse_revision_identity(V2, bangumi_id=None, default_season=1) is None
    assert (
        parse_revision_identity(
            "尼古喵喵 - 01 [V2][1080P].mp4",
            bangumi_id=42,
            default_season=1,
        )
        is None
    )


def test_staged_path_is_deterministic_and_keeps_extension():
    assert (
        replacement_staged_path(
            "subdir/尼古喵喵 S01E01.mp4", old_task_id="abcdef123456", old_revision=1
        )
        == "subdir/尼古喵喵 S01E01.ab-replaced-v1-abcdef12.mp4"
    )


# ---------------------------------------------------------------------------
# find_fallback_season
# ---------------------------------------------------------------------------


class TestFindFallbackSeason:
    def test_never_created_earlier_season_is_treated_as_free(self):
        """AutoBangumi never downloaded Season 1 itself -- absence from the
        occupancy map still counts as a free slot, since the bangumi already
        tracking season 2 is itself evidence season 1 exists in canon."""
        assert (
            find_fallback_season(current_season=2, episode=1, occupied_by_season={})
            == 1
        )

    def test_returns_none_when_earlier_season_already_has_the_episode(self):
        """Season 1 already has this exact episode too -- genuinely
        ambiguous, so don't guess; leave it held for manual review."""
        assert (
            find_fallback_season(
                current_season=2, episode=1, occupied_by_season={1: {1, 2, 3}}
            )
            is None
        )

    def test_prefers_closest_earlier_season_with_a_free_slot(self):
        """Season 2 is occupied but Season 1 (further back) is free --
        skip past the occupied one instead of stopping there."""
        assert (
            find_fallback_season(
                current_season=3,
                episode=5,
                occupied_by_season={2: {5}, 1: set()},
            )
            == 1
        )

    def test_closest_free_season_wins_over_a_farther_one(self):
        assert (
            find_fallback_season(
                current_season=3,
                episode=5,
                occupied_by_season={2: set(), 1: set()},
            )
            == 2
        )

    def test_current_season_one_has_no_candidates(self):
        """A bangumi tracking season 1 itself has nothing earlier to fall
        back to -- must never touch season 0 by accident."""
        assert (
            find_fallback_season(current_season=1, episode=1, occupied_by_season={})
            is None
        )

    def test_every_earlier_season_occupied_returns_none(self):
        assert (
            find_fallback_season(
                current_season=3,
                episode=1,
                occupied_by_season={1: {1}, 2: {1}},
            )
            is None
        )

    def test_fractional_episode_matches_across_int_and_float(self):
        """12.5 (a recap/半集) must match whether stored as int-like float or
        plain float -- set membership relies on Python's numeric equality."""
        assert (
            find_fallback_season(
                current_season=2,
                episode=12.5,
                occupied_by_season={1: {1.0, 2.0}},
            )
            == 1
        )
        assert (
            find_fallback_season(
                current_season=2,
                episode=1,
                occupied_by_season={1: {1.0}},
            )
            is None
        )

    def test_respects_custom_min_season(self):
        """A special/OVA bangumi tracking season 1 with min_season=0 can
        still fall back to Season 0, but never below it."""
        assert (
            find_fallback_season(
                current_season=1,
                episode=1,
                occupied_by_season={},
                min_season=0,
            )
            == 0
        )
        assert (
            find_fallback_season(
                current_season=0,
                episode=1,
                occupied_by_season={},
                min_season=0,
            )
            is None
        )
