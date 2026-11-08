"""Near-duplicate detection based on shared source footage."""
from __future__ import annotations

from core.timestamp_utils import overlap_seconds
from models.schemas import PromoCandidate


def shared_footage_seconds(a: PromoCandidate, b: PromoCandidate) -> float:
    total = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode_id == sb.episode_id:
                total += overlap_seconds((sa.start_seconds, sa.end_seconds), (sb.start_seconds, sb.end_seconds))
    return total


def overlap_ratio(a: PromoCandidate, b: PromoCandidate) -> float:
    shortest = min(a.total_duration_seconds, b.total_duration_seconds)
    if shortest <= 0:
        return 0.0
    return min(1.0, shared_footage_seconds(a, b) / shortest)


def mark_near_duplicates(ordered: list[PromoCandidate], threshold: float) -> list[tuple[PromoCandidate, PromoCandidate, float]]:
    """``ordered`` must be best-first. Later candidates that reuse more than
    ``threshold`` of their footage from a kept candidate are marked duplicate."""
    kept: list[PromoCandidate] = []
    dups: list[tuple[PromoCandidate, PromoCandidate, float]] = []
    for c in ordered:
        match = None
        for k in kept:
            r = overlap_ratio(c, k)
            if r >= threshold:
                match = (k, r)
                break
        if match:
            k, r = match
            c.duplicate_of = k.promo_id
            c.status = "duplicate"
            note = f"near-duplicate of {k.promo_id} ({r:.0%} shared footage)"
            if note not in c.warnings:
                c.warnings.append(note)
            if "duplicate_moment" not in c.failure_cases:
                c.failure_cases.append("duplicate_moment")
            dups.append((c, k, r))
        else:
            kept.append(c)
    return dups
