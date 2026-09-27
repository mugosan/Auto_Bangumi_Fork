"""Admission helpers for destructive higher-revision replacement (#1078)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

from module.parser.analyser.selector import parse_configured_release_title
from module.parser.analyser.tokenizer import MediaType
from module.parser.release_policy import preference_identity, preference_revision


@dataclass(frozen=True, slots=True)
class RevisionIdentity:
    bangumi_id: int
    media_type: MediaType
    season: int
    episode: int | float
    group: str
    resolution: str
    revision: int


def _normalize_label(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", "", normalized)


def _adjust_episode(episode: int | float, offset: int) -> int | float:
    if episode == 0 and offset:
        return 0
    adjusted = episode + offset
    if adjusted < 0 or (adjusted == 0 and episode > 0):
        return episode
    return adjusted


def parse_revision_identity(
    torrent_name: str,
    *,
    bangumi_id: int | None,
    default_season: int,
    episode_offset: int = 0,
) -> RevisionIdentity | None:
    """Return the strict identity required before an old file may be deleted."""

    if bangumi_id is None:
        return None
    release = parse_configured_release_title(torrent_name)
    if release is None:
        return None
    identity = preference_identity(release, default_season=default_season)
    if identity is None:
        return None
    media_type, season, episode = identity
    group = _normalize_label(release.group)
    resolution = _normalize_label(release.resolution)
    if not group or not resolution:
        return None
    return RevisionIdentity(
        bangumi_id=bangumi_id,
        media_type=media_type,
        season=season,
        episode=_adjust_episode(episode, episode_offset),
        group=group,
        resolution=resolution,
        revision=preference_revision(release),
    )


def is_strict_upgrade(old: RevisionIdentity, new: RevisionIdentity) -> bool:
    """Whether *new* may destructively replace *old*."""

    return bool(same_release_identity(old, new) and new.revision > old.revision)


def same_release_identity(old: RevisionIdentity, new: RevisionIdentity) -> bool:
    """Whether two revisions describe the same strictly matched resource."""

    return bool(
        old.bangumi_id == new.bangumi_id
        and old.media_type is new.media_type
        and old.season == new.season
        and old.episode == new.episode
        and old.group == new.group
        and old.resolution == new.resolution
    )


def find_fallback_season(
    *,
    current_season: int,
    episode: int | float,
    occupied_by_season: dict[int, set[int | float]],
    min_season: int = 1,
) -> int | None:
    """Guess which earlier season an untagged, colliding release actually
    belongs to, given what's already known to occupy each earlier season.

    Only ever consulted once a same-season conflict already exists (the
    current season's slot for `episode` is taken by another release that
    isn't a strict upgrade of it) -- it never changes behaviour for a
    torrent that has nowhere to collide with in the first place. RSS
    matching is by show name only, so a fansub release with no season
    marker matches every season's subscription equally well; when the
    currently tracked season already has that episode and an earlier
    season doesn't, the earlier season is the more likely home for it.

    Walks `current_season - 1` down to `min_season` and returns the first
    (closest to current) season whose slot for `episode` is free. A season
    absent from `occupied_by_season` is treated as free -- this is what
    lets a show's Season 1 folder be created for the first time even
    though AutoBangumi never downloaded anything there itself; the show
    already being tracked at `current_season > 1` is itself evidence an
    earlier season exists in canon.

    Returns None when `current_season <= min_season` (nothing earlier to
    consider) or every earlier season already has this exact episode.
    """
    for candidate in range(current_season - 1, min_season - 1, -1):
        if episode not in occupied_by_season.get(candidate, set()):
            return candidate
    return None


def replacement_staged_path(
    target_path: str, *, old_task_id: str, old_revision: int
) -> str:
    """Build a deterministic same-directory temporary path for saga recovery."""

    normalized = target_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    staged_name = f"{stem}.ab-replaced-v{old_revision}-{old_task_id[:8]}{suffix}"
    parent = str(path.parent)
    return staged_name if parent == "." else f"{parent}/{staged_name}"
