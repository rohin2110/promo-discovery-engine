"""Step 3: cross-episode promo discovery over the global candidate pool.

Episode windows are chosen deterministically (episodes within the configured
gap whose single-episode candidates scored best). The model proposes combined
edits for each window; Python then checks that the combination is supported
by the scripts (shared characters, episode gap, continuity references).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from config.settings import ScoringConfig
from core.candidate_generator import validate_with_repair
from core.validator import EpisodeIndex
from llm.prompts import cross_episode_messages
from llm.structured_output import StructuredLLMClient
from models.schemas import Episode, EventType, PromoCandidate, TimelineEvent

log = logging.getLogger(__name__)

# Generic English honorifics/titles; removed so "LADY X" and "X" match. No story knowledge.
HONORIFICS = {
    "the", "a", "an", "of", "lady", "lord", "sir", "dame", "mr", "mrs", "ms", "miss", "dr", "doctor", "madam",
    "king", "queen", "prince", "princess", "duke", "duchess", "earl", "count", "countess", "baron", "baroness",
    "father", "mother", "brother", "sister", "uncle", "aunt", "old", "young", "elder", "little", "officer",
    "captain", "general", "detective", "agent", "professor", "master", "mistress", "dowager", "o.s", "v.o", "cont'd",
}


def name_tokens(speaker: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]+", speaker or "")
    return {w.lower().strip("'’") for w in words if w.lower() not in HONORIFICS and len(w) >= 3}


class CharacterLexicon:
    """Character names learned from the uploaded dataset's speaker labels."""

    def __init__(self, episodes: list[Episode]) -> None:
        self.tokens: set[str] = set()
        for ep in episodes:
            for ev in ep.timeline_events:
                if ev.event_type == EventType.dialogue:
                    self.tokens |= name_tokens(ev.speaker)
        # A "name" used by many speaker labels is usually a role word ("guard", "nobleman").
        self.tokens = {t for t in self.tokens if t.isalpha()}

    def characters_in(self, events: list[TimelineEvent]) -> set[str]:
        found: set[str] = set()
        for ev in events:
            if ev.event_type == EventType.dialogue:
                found |= name_tokens(ev.speaker) & self.tokens
            for w in re.findall(r"\b[A-Z][a-z'’]+\b", ev.text or ""):
                lw = w.lower().strip("'’")
                if lw in self.tokens and lw not in HONORIFICS:
                    found.add(lw)
        return found


def episode_gap(a: Episode, b: Episode) -> int:
    if a.episode_number is not None and b.episode_number is not None:
        return abs(b.episode_number - a.episode_number)
    return abs(b.order_index - a.order_index)


def continuity_links(earlier: Episode, later: Episode) -> bool:
    if earlier.episode_number is None or not later.continuity_notes:
        return False
    return re.search(rf"\bepisode\s*{earlier.episode_number}\b", later.continuity_notes, re.I) is not None


def select_windows(episodes: list[Episode], single: list[PromoCandidate], cfg: ScoringConfig) -> list[list[Episode]]:
    best: dict[str, float] = {}
    for c in single:
        if c.scores and c.validation.eligible and len(c.episode_ids) == 1:
            best[c.episode_ids[0]] = max(best.get(c.episode_ids[0], 0.0), c.scores.total)
    usable = [e for e in episodes if e.usable_for_promos]
    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            gap = episode_gap(a, b)
            if 1 <= gap <= cfg.cross_episode_max_episode_gap:
                score = best.get(a.episode_id, 0.0) + best.get(b.episode_id, 0.0)
                pairs.append((score, a.order_index, [a, b]))
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [p[2] for p in pairs[: cfg.cross_episode_max_windows]]


def coherence_check(cand: PromoCandidate, index: EpisodeIndex, lexicon: CharacterLexicon,
                    cfg: ScoringConfig) -> tuple[bool, list[str]]:
    """Deterministic support for a cross-episode combination."""
    notes: list[str] = []
    per_seg = []
    for s in cand.segments:
        evs = [index.events[e] for e in s.source_event_ids]
        per_seg.append(lexicon.characters_in(evs))
    shared = set.intersection(*per_seg) if per_seg else set()
    linked_pairs = 0
    for a, b in zip(per_seg, per_seg[1:]):
        if a & b:
            linked_pairs += 1
    eps = [index.episodes[e] for e in cand.episode_ids]
    gaps_ok = all(episode_gap(x, y) <= cfg.cross_episode_max_episode_gap for x, y in zip(eps, eps[1:]))
    cont = any(continuity_links(x, y) for x, y in zip(eps, eps[1:]))
    if shared:
        notes.append(f"characters present in every segment: {sorted(shared)}")
    if cont:
        notes.append("continuity notes of the later episode reference the earlier one")
    ok = gaps_ok and (linked_pairs == len(per_seg) - 1 or not cfg.cross_episode_require_shared_character)
    if not gaps_ok:
        notes.append(f"episodes are more than {cfg.cross_episode_max_episode_gap} apart")
    if linked_pairs < len(per_seg) - 1:
        notes.append("at least one cut joins segments that share no character")
    return ok, notes


class CrossEpisodeAnalyser:
    def __init__(self, client: StructuredLLMClient, index: EpisodeIndex, cfg: ScoringConfig, episodes: list[Episode]) -> None:
        self.client = client
        self.index = index
        self.cfg = cfg
        self.lexicon = CharacterLexicon(episodes)

    @property
    def enabled(self) -> bool:
        return self.cfg.allow_cross_episode_segments and self.cfg.effective_max_segments >= 2 and self.cfg.cross_episode_max_windows > 0

    def propose(self, window: list[Episode]) -> list[PromoCandidate]:
        subject = "+".join(e.episode_id for e in window)
        cands = validate_with_repair(
            self.client, self.index, self.cfg, task="cross_episode_generation", subject=subject,
            messages=cross_episode_messages(window, self.cfg),
            context={"episodes": window, "config": self.cfg, "lexicon": self.lexicon},
            default_episode=None, limit=self.cfg.cross_episode_candidates_per_window,
        )
        out = []
        for c in cands:
            if c.rejection_reasons:
                out.append(c)
                continue
            if c.origin != "cross_episode":
                c.rejection_reasons.append("cross-episode request returned a single-episode clip (duplicate of per-episode analysis)")
                c.status = "rejected"
                c.validation.eligible = False
                out.append(c)
                continue
            ok, notes = coherence_check(c, self.index, self.lexicon, self.cfg)
            c.deterministic_notes.extend(notes)
            if not ok:
                c.warnings.append("cross-episode connection not supported by shared characters/episode proximity")
                c.failure_cases.append("incoherent_cross_episode")
            out.append(c)
        return out
