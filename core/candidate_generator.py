"""Step 1: per-episode candidate promo detection with validation-driven repair."""
from __future__ import annotations

import logging

from config.settings import ScoringConfig
from core.validator import EpisodeIndex, resolve_candidate
from llm.prompts import candidate_generation_messages, candidate_repair_message
from llm.structured_output import StructuredLLMClient
from models.schemas import Episode, GenerationResponse, ProposedCandidate, PromoCandidate

log = logging.getLogger(__name__)


def _proposal_key(p: ProposedCandidate) -> tuple:
    return tuple((s.start_event_id.strip().upper(), s.end_event_id.strip().upper()) for s in p.segments)


def validate_with_repair(
    client: StructuredLLMClient, index: EpisodeIndex, cfg: ScoringConfig, *, task: str, subject: str,
    messages: list[dict[str, str]], context: dict, default_episode: str | None, limit: int,
) -> list[PromoCandidate]:
    """Run a generation request, validate every proposal deterministically and
    send failed proposals back to the model with concrete errors and hints."""
    result = client.call(task, messages, GenerationResponse, context=context, subject=subject)
    if not result.ok:
        log.error("%s: generation failed: %s", subject, result.error)
        return [PromoCandidate(
            title=f"(generation failed for {subject})", status="rejected", rejection_reasons=[f"generation failed: {result.error}"],
            generation_call_ids=result.call_ids, episode_ids=[default_episode] if default_episode else [],
        )]
    call_ids = list(result.call_ids)
    proposals = result.parsed.candidates[:limit]  # type: ignore[union-attr]
    valid: list[PromoCandidate] = []
    rejected: list[PromoCandidate] = []
    pending: list[tuple[ProposedCandidate, PromoCandidate, list[str], list[str]]] = []
    seen: set[tuple] = set()

    def _take(props: list[ProposedCandidate], note: str = "") -> list[tuple[ProposedCandidate, PromoCandidate, list[str], list[str]]]:
        bad = []
        for p in props:
            key = _proposal_key(p)
            if key in seen:
                continue
            seen.add(key)
            cand, errors, hints = resolve_candidate(p, index, cfg, default_episode)
            cand.generation_call_ids = list(call_ids)
            if note:
                cand.warnings.append(note)
            if errors:
                bad.append((p, cand, errors, hints))
            else:
                valid.append(cand)
        return bad

    pending = _take(proposals)
    convo = result.messages
    for attempt in range(1, cfg.generation_repair_attempts + 1):
        if not pending:
            break
        problems = [{"title": p.title, "segments": [s.model_dump() for s in p.segments], "errors": errs, "hints": hints}
                    for p, _, errs, hints in pending]
        repair_msgs = convo + [{"role": "user", "content": candidate_repair_message(problems, cfg)}]
        rep = client.call(f"{task}_repair", repair_msgs, GenerationResponse,
                          context={**context, "repair": problems}, subject=f"{subject} repair {attempt}")
        call_ids += rep.call_ids
        if not rep.ok:
            log.warning("%s: repair attempt %d failed: %s", subject, attempt, rep.error)
            break
        convo = rep.messages
        # The failed originals stay on record as rejected; repaired versions are re-validated.
        for _, cand, _, _ in pending:
            cand.warnings.append(f"sent back for repair (attempt {attempt})")
            rejected.append(cand)
        pending = _take(rep.parsed.candidates[:len(pending)], f"repaired after validation feedback (attempt {attempt})")  # type: ignore[union-attr]
    rejected.extend(cand for _, cand, _, _ in pending)
    for c in rejected:
        c.status = "rejected"
        c.generation_call_ids = list(dict.fromkeys(c.generation_call_ids + call_ids))
    return valid + rejected


class CandidateGenerator:
    def __init__(self, client: StructuredLLMClient, index: EpisodeIndex, cfg: ScoringConfig) -> None:
        self.client = client
        self.index = index
        self.cfg = cfg

    def generate_for_episode(self, ep: Episode) -> list[PromoCandidate]:
        if not ep.usable_for_promos:
            return []
        if ep.timeline_end is not None and ep.timeline_start is not None and \
                ep.timeline_end - ep.timeline_start < self.cfg.min_duration_seconds:
            log.info("%s shorter than minimum promo duration; skipped", ep.episode_id)
            return []
        k = self.cfg.candidates_per_episode
        return validate_with_repair(
            self.client, self.index, self.cfg, task="candidate_generation", subject=ep.episode_id,
            messages=candidate_generation_messages(ep, self.cfg, k),
            context={"episode": ep, "config": self.cfg, "k": k}, default_episode=ep.episode_id, limit=k,
        )
