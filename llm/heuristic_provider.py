"""Deterministic offline provider (no LLM).

Used when no model is reachable, for demos and for fast tests. It answers the
same tasks with the same JSON contracts as a real model, using a generic
English drama lexicon and timeline features. Because the output flows through
the identical validation, scoring and ranking code, swapping in a real LLM
needs no pipeline changes.

This is a transparent baseline, not a substitute for semantic judgement: it
reports a low self-confidence (0.35) so its promos are flagged for review.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm.provider import BaseLLMProvider, LLMRequest, LLMResponse
from models.schemas import Episode, EventType, TimelineEvent

HEURISTIC_CONFIDENCE = 0.35
MAX_RATING = 8.5  # a lexical baseline never claims a perfect promo
STORY = {EventType.dialogue, EventType.action, EventType.description, EventType.other}

_LEXICON = re.compile(
    r"\b(betray\w*|lie|lies|liar|lying|kill\w*|murder\w*|dead|death|die[sd]?|dying|blood\w*|poison\w*|secret\w*|"
    r"truth|never|dare|guard\w*|arrest\w*|prison\w*|jail|cell|hang\w*|execut\w*|revenge|steal\w*|stole|thief|"
    r"threat\w*|danger\w*|help|stop|run|divorc\w*|pregnan\w*|baby|marr\w*|wedding|love|hate|curse\w*|trial|"
    r"guilty|accus\w*|enemy|monster|destroy\w*|ruin\w*|shame\w*|humiliat\w*|disgrace\w*|abandon\w*|traitor|"
    r"fraud|fake|impostor|hid(e|den|ing)|discover\w*|reveal\w*|confess\w*|burn\w*|fire|sword|knife|gun|weapon\w*|"
    r"attack\w*|fight\w*|scream\w*|tears?|slap\w*|witness\w*|proof|evidence|own|mine|lose|lost|alive|gone|"
    r"forbid\w*|liar|whore|bastard|coward|banish\w*|disown\w*|inherit\w*|fortune|debt|sold|ransom|kidnap\w*)\b",
    re.I,
)
_EMOTION = re.compile(
    r"(angr|furi|shock|desperat|scream|terrif|cry|crying|sob|outrag|accus|threat|menac|panic|horrif|devastat|"
    r"cold|venom|cruel|rage|enrag|hiss|snarl|gloat|smug|defian|broken|tremb|chok|frantic|icy|deadly|fierce|"
    r"sinister|gasp|stunn|disbelie|incredul|seeth|wrath|bitter|vicious|mock)",
    re.I,
)
_ACTION = re.compile(
    r"\b(slap\w*|grab\w*|strik\w*|struck|shov\w*|collaps\w*|falls?|fell|scream\w*|burst\w*|slam\w*|shatter\w*|"
    r"seiz\w*|drag\w*|stab\w*|chok\w*|kneel\w*|faint\w*|bleed\w*|push\w*|throw\w*|threw|lung\w*|punch\w*|"
    r"draws? (a|his|her) (sword|knife|gun|blade)|kiss\w*|storms?|bursts? in|barg\w*|crash\w*|tears? (up|open))\b",
    re.I,
)
_SOUND = re.compile(r"(cliffhanger|ominous|sting|crash|boom|thunder|scream|gunshot|shatter|gasp|slam|swell)", re.I)
_CONFLICT = re.compile(r"\b(you|your|why|how could|never|won't|will not|must|dare|lie|truth|betray\w*|take|stop)\b", re.I)


def intensity(ev: TimelineEvent) -> float:
    text = ev.text or ""
    score = min(2.0, 0.7 * len(_LEXICON.findall(text)))
    if ev.event_type == EventType.dialogue:
        score += 0.4
        score += 0.6 if "!" in text else 0.0
        score += 0.5 if "?" in text else 0.0
        if _EMOTION.search(ev.parenthetical or ""):
            score += 0.7
    elif ev.event_type == EventType.action:
        score += 0.9 if _ACTION.search(text) else 0.2
        score += 0.3 if _EMOTION.search(text) else 0.0
    elif ev.event_type == EventType.camera:
        score = 0.3 * score + (0.2 if re.search(r"close-?up|push-?in|snap|whip|crash zoom", text, re.I) else 0.0)
    elif ev.event_type == EventType.other:
        score += 0.8 if _SOUND.search(text) else 0.1
    return round(score, 3)


def _overlap(ev: TimelineEvent, a: float, b: float) -> bool:
    return ev.is_timed and min(ev.end_seconds, b) - max(ev.start_seconds, a) > 0.05  # type: ignore[type-var,operator]


def window_intensity(events: list[TimelineEvent], a: float, b: float) -> float:
    return sum(intensity(e) * (0.3 if e.event_type == EventType.camera else 1.0) for e in events if _overlap(e, a, b))


def _open_ending(ev: Optional[TimelineEvent]) -> float:
    if ev is None:
        return 0.0
    t = (ev.text or "").strip().strip('"”’\'').strip()
    bonus = 0.0
    if t.endswith("?"):
        bonus += 1.2
    if t.endswith(("...", "…", "—", "-")):
        bonus += 0.9
    if ev.event_type == EventType.other and _SOUND.search(ev.text or ""):
        bonus += 0.8
    return bonus


def _quote(ev: TimelineEvent, limit: int = 90) -> str:
    t = (ev.text or "").strip()
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0]
    return cut


def _speaker_tokens(events: list[TimelineEvent]) -> set[str]:
    out: set[str] = set()
    for e in events:
        if e.event_type == EventType.dialogue and e.speaker:
            out.add(e.speaker.split("(")[0].strip().split()[-1].lower())
    return out


class HeuristicProvider(BaseLLMProvider):
    name = "heuristic"

    @property
    def model(self) -> str:
        return "heuristic-baseline-v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        ctx = request.context or {}
        task = request.task
        if task.startswith("candidate_generation"):
            payload = self._generate(ctx["episode"], ctx["config"], ctx.get("k", 3))
        elif task.startswith("cross_episode_generation"):
            payload = self._cross(ctx["episodes"], ctx["config"], ctx.get("lexicon"))
        elif task.startswith("promo_evaluation"):
            payload = self._evaluate(ctx["clip"], ctx["config"])
        else:
            payload = {"error": f"heuristic provider does not support task '{task}'"}
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), provider=self.name, model=self.model)

    # ------------------------------------------------------------------
    def _scored_windows(self, ep: Episode, cfg, lo: float, hi: float):
        """All (score, start_event, end_event) windows with lo <= duration <= hi, best first."""
        timed = sorted(ep.timed_events, key=lambda e: (e.start_seconds, e.order))
        story = [e for e in timed if e.event_type in STORY]
        scene_last = {e.scene_id: e for e in story}
        hook = {s.event_id: window_intensity(timed, s.start_seconds, s.start_seconds + cfg.hook_window_seconds)  # type: ignore[operator]
                for s in story}
        cliff: dict[str, float] = {}
        for e in story:
            t1 = e.end_seconds
            c = window_intensity(timed, t1 - cfg.cliffhanger_window_seconds, t1) + _open_ending(e)  # type: ignore[operator]
            if scene_last.get(e.scene_id) is e:
                c += 0.6
            if any(x.event_type == EventType.dialogue and x.start_seconds < t1 - 0.2 < x.end_seconds - 0.4  # type: ignore[operator]
                   for x in timed if x is not e):
                c -= 1.5
            cliff[e.event_id] = c
        base = []
        for s in story:
            for e in story:
                if e.order < s.order:
                    continue
                dur = e.end_seconds - s.start_seconds  # type: ignore[operator]
                if lo <= dur <= hi:
                    base.append((hook[s.event_id] + cliff[e.event_id], s, e))
        base.sort(key=lambda t: (-t[0], t[1].order, t[2].order))
        out = []
        for score, s, e in base[:300]:  # speaker bonus only for the strongest windows
            inside = [x for x in timed if _overlap(x, s.start_seconds, e.end_seconds)]  # type: ignore[arg-type]
            out.append((round(score + 0.3 * min(len(_speaker_tokens(inside)), 3), 3), s, e))
        out += [(round(sc, 3), s, e) for sc, s, e in base[300:]]
        out.sort(key=lambda t: (-t[0], t[1].order, t[2].order))
        return out

    def _generate(self, ep: Episode, cfg, k: int) -> dict[str, Any]:
        windows = self._scored_windows(ep, cfg, cfg.min_duration_seconds, cfg.max_duration_seconds)
        chosen: list[tuple[float, TimelineEvent, TimelineEvent]] = []
        for w in windows:
            _, s, e = w
            a, b = s.start_seconds, e.end_seconds
            if any(min(b, c[2].end_seconds) - max(a, c[1].start_seconds) > 0.3 * min(b - a, c[2].end_seconds - c[1].start_seconds)  # type: ignore[operator]
                   for c in chosen):
                continue
            chosen.append(w)
            if len(chosen) >= k:
                break
        cands = []
        for score, s, e in chosen:
            cands.append({
                "title": f"{ep.episode_title}: {_quote(s, 40)}",
                "segments": [{"start_event_id": s.event_id, "end_event_id": e.event_id}],
                "opening_hook": f"Opens on {s.event_type.value}: {_quote(s, 80)}",
                "central_conflict": "Selected by lexical drama intensity (heuristic baseline; no semantic analysis).",
                "ending_cliffhanger": f"Ends on {e.event_type.value}: {_quote(e, 80)}",
                "promotional_reason": f"High heuristic intensity at both ends of the clip (window score {score:.2f}).",
            })
        return {"candidates": cands}

    def _cross(self, episodes: list[Episode], cfg, lexicon) -> dict[str, Any]:
        if len(episodes) < 2:
            return {"candidates": []}
        a_ep, b_ep = episodes[0], episodes[-1]
        hook_lo = max(cfg.min_segment_seconds, 6.0)
        hook_hi = max(hook_lo, min(20.0, cfg.max_duration_seconds / 2))
        hooks = self._scored_windows(a_ep, cfg, hook_lo, hook_hi)
        ends = self._scored_windows(b_ep, cfg, max(cfg.min_segment_seconds, 8.0), cfg.max_duration_seconds - hook_lo)
        for _, hs, he in hooks[:15]:
            seg_a = he.end_seconds - hs.start_seconds  # type: ignore[operator]
            a_events = [x for x in a_ep.timed_events if _overlap(x, hs.start_seconds, he.end_seconds)]  # type: ignore[arg-type]
            for _, es, ee in ends[:25]:
                total = seg_a + ee.end_seconds - es.start_seconds  # type: ignore[operator]
                if not cfg.min_duration_seconds <= total <= cfg.max_duration_seconds:
                    continue
                b_events = [x for x in b_ep.timed_events if _overlap(x, es.start_seconds, ee.end_seconds)]  # type: ignore[arg-type]
                if lexicon is not None:
                    shared = lexicon.characters_in(a_events) & lexicon.characters_in(b_events)
                else:
                    shared = _speaker_tokens(a_events) & _speaker_tokens(b_events)
                if not shared:
                    continue
                return {"candidates": [{
                    "title": f"{a_ep.episode_title} to {b_ep.episode_title}",
                    "segments": [{"start_event_id": hs.event_id, "end_event_id": he.event_id},
                                 {"start_event_id": es.event_id, "end_event_id": ee.event_id}],
                    "opening_hook": f"Opens on: {_quote(hs, 80)}",
                    "central_conflict": "Heuristic pairing of a high-intensity opening with a later open ending.",
                    "ending_cliffhanger": f"Ends on: {_quote(ee, 80)}",
                    "promotional_reason": "Strongest lexical hook and ending across consecutive episodes.",
                    "narrative_connection": f"Both segments feature: {', '.join(sorted(shared))} (speaker/name overlap).",
                }]}
        return {"candidates": []}

    def _evaluate(self, clip, cfg) -> dict[str, Any]:
        events = clip.all_events
        hook_ev = [e for e in clip.hook_events if e.event_type in STORY]
        cliff_ev = [e for e in clip.cliff_events if e.event_type in STORY]
        H = sum(intensity(e) * (0.3 if e.event_type == EventType.camera else 1.0) for e in clip.hook_events)
        C = sum(intensity(e) * (0.3 if e.event_type == EventType.camera else 1.0) for e in clip.cliff_events)
        last_story = cliff_ev[-1] if cliff_ev else None
        hook_rating = min(MAX_RATING, 1.0 + 1.6 * H)
        if not hook_ev:
            hook_rating = min(hook_rating, 3.0)
        cliff_rating = 1.0 + 1.3 * C + _open_ending(last_story)
        if any("mid-line" in n and "cuts dialogue" in n for n in clip.notes):
            cliff_rating -= 2.5
        cliff_rating = max(0.0, min(MAX_RATING, cliff_rating))
        if not cliff_ev:
            cliff_rating = min(cliff_rating, 3.0)

        dialogue = [e for e in events if e.event_type == EventType.dialogue]
        speakers = _speaker_tokens(events)
        named = sum(1 for e in dialogue if any(re.search(rf"\b{re.escape(s)}\b", e.text, re.I) for s in speakers))
        conflict_hits = sum(len(_CONFLICT.findall(e.text)) for e in dialogue)
        ctx = 1.0 + (2.5 if len(speakers) >= 2 else 1.0 if speakers else 0.0) + min(2.0, named) \
            + (2.0 if len(dialogue) >= 4 else 1.0 if dialogue else 0.0) + min(2.5, 0.25 * conflict_hits)
        ctx_rating = max(0.0, min(MAX_RATING, ctx))

        story_secs = 0.0
        for sv in clip.segments:
            ivs = sorted((max(e.start_seconds, sv.segment.start_seconds), min(e.end_seconds, sv.segment.end_seconds))
                         for e in sv.events if e.event_type in STORY)
            cur_a = cur_b = None
            for a, b in ivs:
                if cur_b is None or a > cur_b:
                    if cur_b is not None:
                        story_secs += cur_b - cur_a
                    cur_a, cur_b = a, b
                else:
                    cur_b = max(cur_b, b)
            if cur_b is not None:
                story_secs += cur_b - cur_a
        density = story_secs / clip.total_duration if clip.total_duration else 0.0
        pace = 3.0 + 7.0 * min(1.0, density / 0.6) - 1.5 * sum(1 for n in clip.notes if "without dialogue" in n)
        pace_rating = max(0.0, min(MAX_RATING, pace))

        def ev_list(evs: list[TimelineEvent], n: int) -> list[dict[str, str]]:
            top = sorted(evs, key=lambda e: -intensity(e))[:n]
            return [{"event_id": e.event_id, "quote": _quote(e)} for e in top if e.text]

        coherent = None
        if clip.cross_episode:
            per = [_speaker_tokens(sv.events) for sv in clip.segments]
            coherent = all(a & b for a, b in zip(per, per[1:]))
        weaknesses = []
        if hook_rating < 5:
            weaknesses.append("opening window has little dramatic language")
        if cliff_rating < 5:
            weaknesses.append("ending does not clearly leave an open question")
        if ctx_rating < 5:
            weaknesses.append("few named characters or conflict cues for a new viewer")
        if pace_rating < 5:
            weaknesses.append("long stretches without dialogue or action")
        weaknesses.append("heuristic baseline: ratings are lexical estimates, not semantic judgement; human review advised")
        opening = hook_ev[0] if hook_ev else (clip.hook_events[0] if clip.hook_events else None)
        return {
            "title": clip.title,
            "hook": {"rating": round(hook_rating, 1), "description": _quote(opening) if opening else "",
                     "evidence": ev_list(hook_ev, 1), "reasoning": f"lexical intensity {H:.2f} in hook window"},
            "cliffhanger": {"rating": round(cliff_rating, 1), "description": _quote(last_story) if last_story else "",
                            "evidence": ev_list(cliff_ev[-2:], 1), "reasoning": f"lexical intensity {C:.2f} in final window"},
            "context": {"rating": round(ctx_rating, 1), "summary": f"{len(speakers)} speaking character(s), {len(dialogue)} line(s)",
                        "evidence": ev_list(dialogue, 2), "reasoning": f"{named} line(s) name a character; {conflict_hits} conflict cue(s)"},
            "pacing": {"rating": round(pace_rating, 1), "reasoning": f"dialogue/action covers {density:.0%} of runtime"},
            "confidence": HEURISTIC_CONFIDENCE,
            "weaknesses": weaknesses,
            "why_it_works": "Heuristic: the opening and closing windows contain the densest dramatic language in the clip"
                            + (f", ending on '{_quote(last_story, 60)}'" if last_story else "") + ".",
            "cross_episode_coherent": coherent,
        }
