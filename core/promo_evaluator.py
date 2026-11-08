"""Step 2: evaluate a validated candidate as a complete standalone clip."""
from __future__ import annotations

import logging

from config.settings import ScoringConfig
from core.clip_builder import build_clip_view
from core.scoring_engine import apply_eligibility, build_criterion, compute_scores
from core.validator import EpisodeIndex, verify_evidence
from llm.prompts import evaluation_messages, evidence_repair_message
from llm.structured_output import StructuredLLMClient
from models.schemas import EvaluationResponse, PromoCandidate

log = logging.getLogger(__name__)


class PromoEvaluator:
    def __init__(self, client: StructuredLLMClient, index: EpisodeIndex, cfg: ScoringConfig) -> None:
        self.client = client
        self.index = index
        self.cfg = cfg

    def evaluate(self, cand: PromoCandidate) -> PromoCandidate:
        cfg = self.cfg
        clip = build_clip_view(cand, self.index, cfg)
        cand.deterministic_notes = list(dict.fromkeys(cand.deterministic_notes + clip.notes))
        messages = evaluation_messages(clip.to_prompt_text(cfg), cfg, cross_episode=clip.cross_episode)
        context = {"clip": clip, "config": cfg, "candidate": cand}
        subject = cand.promo_id or cand.title
        res = self.client.call("promo_evaluation", messages, EvaluationResponse, context=context, subject=subject)
        cand.evaluation_call_ids += res.call_ids
        if not res.ok:
            cand.status = "unevaluated"
            cand.review_reasons.append(f"evaluation failed: {res.error}")
            cand.human_review_required = True
            apply_eligibility(cand, cfg)
            return cand

        ev: EvaluationResponse = res.parsed  # type: ignore[assignment]
        verified = self._verify(ev, clip.event_ids)
        bad = [{"criterion": k, **e.model_dump()} for k, items in verified.items() for e in items if not e.verified]
        attempts = 0
        while bad and attempts < cfg.evaluation_retry_attempts:
            attempts += 1
            retry = self.client.call(
                "promo_evaluation_evidence_repair",
                res.messages + [{"role": "user", "content": evidence_repair_message(bad)}],
                EvaluationResponse, context=context, subject=f"{subject} evidence repair {attempts}",
            )
            cand.evaluation_call_ids += retry.call_ids
            if not retry.ok:
                break
            res = retry
            ev = retry.parsed  # type: ignore[assignment]
            verified = self._verify(ev, clip.event_ids)
            bad = [{"criterion": k, **e.model_dump()} for k, items in verified.items() for e in items if not e.verified]

        w = cfg.weights
        cand.hook = build_criterion(ev.hook, w.hook, verified["hook"])
        cand.cliffhanger = build_criterion(ev.cliffhanger, w.cliffhanger, verified["cliffhanger"])
        cand.context = build_criterion(ev.context, w.context, verified["context"])
        cand.pacing = build_criterion(ev.pacing, w.pacing, verified["pacing"])
        cand.scores = compute_scores(cand)
        if ev.title.strip() and cand.title in ("", "Untitled promo"):
            cand.title = ev.title.strip()
        cand.model_confidence = ev.confidence
        cand.weaknesses = [w_ for w_ in ev.weaknesses if str(w_).strip()]
        cand.why_it_works = ev.why_it_works
        cand.cross_episode_coherent = ev.cross_episode_coherent if clip.cross_episode else None
        if bad:
            cand.warnings.append(f"{len(bad)} evidence quote(s) unverified after {attempts} retry(ies)")
        cand.validation.evidence_verified = not bad
        cand.status = "ranked"
        apply_eligibility(cand, cfg)
        return cand

    def _verify(self, ev: EvaluationResponse, clip_ids: list[str]) -> dict:
        return {k: verify_evidence(getattr(ev, k).evidence, clip_ids, self.index)
                for k in ("hook", "cliffhanger", "context", "pacing")}
