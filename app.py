"""Streamlit dashboard for the AI promo discovery and ranking engine.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from config.settings import LLMSettings, ScoringConfig, load_app_config, load_llm_settings, load_scoring_config
from core.ingestion import ARCHIVE_EXTENSIONS, SUPPORTED_EXTENSIONS
from core.normalizer import load_dataset
from core.pipeline import STAGES, PromoPipeline, list_runs, load_run, rescore_candidates
from core.quality_assurance import FAILURE_CASES, FeedbackEntry, FeedbackStore, build_qa_report, qa_report_markdown, snapshot
from models.schemas import PromoCandidate
from utils.export import build_zip, candidates_frame, export_run, segments_frame, validation_report

st.set_page_config(page_title="Promo Discovery Engine", layout="wide")

APP_CFG = load_app_config()
FEEDBACK = FeedbackStore(APP_CFG.resolve(APP_CFG.feedback_path))
STATUS_COLORS = {"recommended": "green", "ranked": "blue", "duplicate": "orange", "rejected": "red", "unevaluated": "gray"}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.small-note {color: var(--text-color); opacity: .7; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
def sidebar_config() -> tuple[Optional[ScoringConfig], Optional[LLMSettings], list[str]]:
    errors: list[str] = []
    base = load_scoring_config()
    env_llm = load_llm_settings()
    st.sidebar.header("Configuration")

    with st.sidebar.expander("Duration and windows", expanded=False):
        min_d = st.number_input("Minimum duration (s)", 1.0, 600.0, float(base.min_duration_seconds), 1.0)
        max_d = st.number_input("Maximum duration (s)", 2.0, 900.0, float(base.max_duration_seconds), 1.0)
        tgt_d = st.number_input("Target duration (s)", 1.0, 900.0, float(base.target_duration_seconds), 1.0)
        hook_w = st.number_input("Hook window (s)", 0.5, 30.0, float(base.hook_window_seconds), 0.5)
        cliff_w = st.number_input("Cliffhanger window (s)", 0.5, 30.0, float(base.cliffhanger_window_seconds), 0.5)

    with st.sidebar.expander("Scoring weights (must sum to 100)", expanded=False):
        w_hook = st.number_input("Hook", 0.0, 100.0, float(base.weights.hook), 1.0)
        w_cliff = st.number_input("Cliffhanger", 0.0, 100.0, float(base.weights.cliffhanger), 1.0)
        w_ctx = st.number_input("Standalone context", 0.0, 100.0, float(base.weights.context), 1.0)
        w_pace = st.number_input("Pacing and editability", 0.0, 100.0, float(base.weights.pacing), 1.0)
        total_w = w_hook + w_cliff + w_ctx + w_pace
        (st.success if abs(total_w - 100) < 1e-6 else st.error)(f"Sum of weights: {total_w:g}")

    with st.sidebar.expander("Eligibility thresholds", expanded=False):
        m_hook = st.number_input("Minimum hook score", 0.0, 100.0, float(base.minimum_scores.hook), 1.0)
        m_cliff = st.number_input("Minimum cliffhanger score", 0.0, 100.0, float(base.minimum_scores.cliffhanger), 1.0)
        m_ctx = st.number_input("Minimum context score", 0.0, 100.0, float(base.minimum_scores.context), 1.0)
        m_total = st.number_input("Minimum total score", 0.0, 100.0, float(base.minimum_scores.total), 1.0)
        min_conf = st.slider("Confidence needed to skip review", 0.0, 1.0, float(base.min_confidence_for_auto_accept), 0.05)

    with st.sidebar.expander("Segments and selection", expanded=False):
        allow_multi = st.checkbox("Allow multiple segments", base.allow_multiple_segments)
        allow_cross = st.checkbox("Allow cross-episode promos", base.allow_cross_episode_segments)
        max_seg = st.number_input("Maximum segments per promo", 1, 10, int(base.max_segments_per_promo))
        per_ep = st.number_input("Candidates per episode", 1, 10, int(base.candidates_per_episode))
        cross_windows = st.number_input("Cross-episode windows to analyse", 0, 100, int(base.cross_episode_max_windows))
        top_k = st.number_input("Top recommendations", 1, 200, int(base.top_k_recommendations))
        prefer_cont = st.checkbox("Prefer continuous segments", base.prefer_continuous_segments)
        dedupe = st.checkbox("Remove near-duplicates", base.remove_near_duplicates)
        dup_thr = st.slider("Near-duplicate overlap threshold", 0.1, 1.0, float(base.near_duplicate_overlap_threshold), 0.05)

    with st.sidebar.expander("AI provider and model", expanded=True):
        providers = ["openai_compatible", "ollama", "heuristic"]
        provider = st.selectbox("Provider", providers, index=providers.index(env_llm.provider) if env_llm.provider in providers else 2,
                                help="heuristic = deterministic offline baseline (no LLM, no key).")
        model = st.text_input("Model", env_llm.model if provider == env_llm.provider else "")
        api_base = st.text_input("API base URL", env_llm.api_base if provider == env_llm.provider else
                                 {"ollama": "http://localhost:11434", "openai_compatible": "https://api.openai.com/v1"}.get(provider, ""))
        key_in = st.text_input("API key", "", type="password",
                               help="Leave empty to use LLM_API_KEY from the environment. Never written to disk.")
        temperature = st.slider("Temperature", 0.0, 1.0, float(env_llm.temperature), 0.05)
        max_tokens = st.number_input("Max tokens", 256, 32768, int(env_llm.max_tokens), 256)
        timeout = st.number_input("Timeout (s)", 5, 1200, int(env_llm.timeout), 5)
        retries = st.number_input("Max retries", 0, 10, int(env_llm.max_retries))
        concurrency = st.number_input("Parallel requests", 1, 32, int(env_llm.concurrency))
        if provider == "heuristic":
            st.info("Offline baseline: lexical scoring with no semantic understanding. All promos are flagged for review.")

    try:
        scoring = ScoringConfig(
            min_duration_seconds=min_d, max_duration_seconds=max_d, target_duration_seconds=tgt_d,
            hook_window_seconds=hook_w, cliffhanger_window_seconds=cliff_w,
            weights={"hook": w_hook, "cliffhanger": w_cliff, "context": w_ctx, "pacing": w_pace},
            minimum_scores={"hook": m_hook, "cliffhanger": m_cliff, "context": m_ctx, "total": m_total},
            allow_multiple_segments=allow_multi, allow_cross_episode_segments=allow_cross,
            max_segments_per_promo=int(max_seg), prefer_continuous_segments=prefer_cont,
            remove_near_duplicates=dedupe, top_k_recommendations=int(top_k), candidates_per_episode=int(per_ep),
            cross_episode_max_windows=int(cross_windows), near_duplicate_overlap_threshold=dup_thr,
            min_confidence_for_auto_accept=min_conf,
            **base.model_dump(include={
                "min_segment_seconds", "generation_repair_attempts", "evaluation_retry_attempts",
                "cross_episode_max_episode_gap", "cross_episode_candidates_per_window",
                "cross_episode_require_shared_character", "multi_segment_penalty", "cross_episode_penalty",
                "failure_case_strong_ratio", "failure_case_weak_ratio",
            }),
        )
    except ValidationError as exc:
        scoring = None
        errors += [e["msg"].replace("Value error, ", "") for e in exc.errors()]
    try:
        llm = LLMSettings(**{
            **env_llm.model_dump(exclude={"api_key"}),
            "api_key": key_in or env_llm.api_key.get_secret_value(),
            "provider": provider, "model": model, "api_base": api_base, "temperature": temperature,
            "max_tokens": int(max_tokens), "timeout": float(timeout), "max_retries": int(retries),
            "concurrency": int(concurrency),
        })
        errors += llm.missing_requirements()
    except ValidationError as exc:
        llm = None
        errors += [e["msg"] for e in exc.errors()]
    for e in errors:
        st.sidebar.error(e)
    if scoring is not None:
        st.sidebar.download_button("Download scoring config", json.dumps(scoring.model_dump(), indent=2),
                                   "scoring_config.json", "application/json", width="stretch")
    return scoring, llm, errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def uploads_digest(uploads: list[tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for name, data in uploads:
        h.update(name.encode())
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def status_badge(status: str) -> str:
    return f":{STATUS_COLORS.get(status, 'gray')}-badge[{status.replace('_', ' ').title()}]"


def set_result(data: dict[str, Any]) -> None:
    st.session_state["result"] = data


def result() -> Optional[dict[str, Any]]:
    return st.session_state.get("result")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_upload() -> list[tuple[str, bytes]]:
    st.subheader("Upload screenplays")
    exts = sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS | ARCHIVE_EXTENSIONS)
    files = st.file_uploader(
        f"Screenplay files or ZIP datasets ({', '.join(exts)})", accept_multiple_files=True, type=exts,
        help=f"Limits: {APP_CFG.max_upload_mb:g} MB per upload, {APP_CFG.max_file_mb:g} MB per file, nested ZIP depth {APP_CFG.max_zip_depth}.",
    )
    uploads = [(f.name, f.getvalue()) for f in files or []]
    if not uploads:
        st.info("Upload a ZIP of screenplay files (or individual TXT, MD, PDF, DOCX, JSON, CSV or SRT files) to begin.")
        return []
    digest = uploads_digest(uploads)
    if st.session_state.get("preview_digest") != digest:
        with st.spinner("Parsing and normalising..."):
            episodes, report, ingestion = load_dataset(uploads, APP_CFG)
        st.session_state["preview"] = (episodes, report)
        st.session_state["preview_digest"] = digest
    episodes, report = st.session_state["preview"]
    parsed_files = sum(1 for f in report.files if f.status == "parsed")
    c = st.columns(6)
    c[0].metric("Files parsed", parsed_files)
    c[1].metric("Episodes detected", report.episodes_detected)
    c[2].metric("Usable episodes", sum(1 for e in episodes if e.usable_for_promos))
    c[3].metric("Timeline events", report.events_total)
    c[4].metric("Errors", len(report.errors))
    c[5].metric("Warnings", len(report.warnings))
    for i in report.errors[:20]:
        st.error(f"{i.source_filename}: {i.message}")
    if len(report.errors) > 20:
        st.error(f"... and {len(report.errors) - 20} more errors (see the parse report).")
    if report.warnings:
        with st.expander(f"Warnings ({len(report.warnings)})"):
            st.dataframe(pd.DataFrame([w.model_dump(mode="json") for w in report.warnings]), width="stretch", hide_index=True)
    with st.expander("File inventory"):
        st.dataframe(pd.DataFrame([f.model_dump() for f in report.files]), width="stretch", hide_index=True)
    with st.expander("Episodes", expanded=True):
        st.dataframe(pd.DataFrame(report.episode_summaries), width="stretch", hide_index=True)
    if episodes:
        with st.expander("Inspect normalised events"):
            ep_id = st.selectbox("Episode", [e.episode_id for e in episodes], key="inspect_ep")
            ep = next(e for e in episodes if e.episode_id == ep_id)
            st.caption(f"{ep.episode_title} | {ep.source_filename}")
            if ep.logline:
                st.write(f"**Logline:** {ep.logline}")
            st.dataframe(pd.DataFrame([{
                "event_id": e.event_id, "scene_id": e.scene_id, "type": e.event_type.value, "speaker": e.speaker,
                "start": e.start_timestamp, "end": e.end_timestamp, "start_s": e.start_seconds, "end_s": e.end_seconds,
                "status": e.timestamp_status, "flags": ", ".join(e.flags), "text": e.text,
            } for e in ep.timeline_events]), width="stretch", hide_index=True, height=360)
    return uploads


def tab_run(uploads: list[tuple[str, bytes]], scoring: Optional[ScoringConfig], llm: Optional[LLMSettings], errors: list[str]) -> None:
    st.subheader("Run promo discovery")
    if not uploads:
        st.info("Upload a dataset first.")
        return
    if errors or scoring is None or llm is None:
        st.warning("Fix the configuration errors in the sidebar before running.")
        return
    episodes, _ = st.session_state.get("preview", ([], None))
    usable = sum(1 for e in episodes if e.usable_for_promos)
    limit = st.number_input("Limit to the first N usable episodes (0 = all)", 0, max(usable, 1), 0,
                            help="Useful for a quick trial with a slow or paid model.")
    st.caption(f"Backend: **{llm.provider}** {llm.model} | {usable} usable episode(s) | "
               f"{scoring.candidates_per_episode} candidate(s) per episode | cross-episode "
               f"{'on' if scoring.allow_cross_episode_segments else 'off'}")
    if not st.button("Run promo discovery", type="primary", disabled=usable == 0):
        return
    overall = st.progress(0.0, text="Starting...")
    rows = {s: st.empty() for s in STAGES}
    for s in STAGES:
        rows[s].markdown(f":gray[pending] **{s}**")

    def progress(stage: str, frac: float, msg: str) -> None:
        idx = STAGES.index(stage) if stage in STAGES else 0
        overall.progress(min(1.0, (idx + frac) / len(STAGES)), text=f"{stage}: {msg}")
        label = ":green[done]" if frac >= 1.0 else ":blue[running]"
        rows.get(stage, st.empty()).markdown(f"{label} **{stage}** {msg}")

    try:
        res = PromoPipeline(scoring, llm, APP_CFG, progress=progress).run(uploads, max_episodes=int(limit) or None)
    except Exception as exc:  # surface any unexpected crash clearly
        st.exception(exc)
        return
    overall.progress(1.0, text=f"Finished in {res.elapsed_seconds:.1f}s")
    set_result({
        "run_id": res.run_id, "output_dir": str(res.output_dir), "raw_dir": str(res.raw_dir), "backend": res.backend,
        "candidates": res.candidates, "episodes": res.episodes, "parse_report": res.parse_report,
        "qa_report": res.qa_report, "errors": res.errors, "scoring": scoring,
    })
    st.success(f"Run {res.run_id} complete: {len(res.ranked)} promos ranked from {len(res.candidates)} candidates. "
               "Open the Results tab.")
    for e in res.errors:
        st.error(e)


def render_card(c: PromoCandidate, run_id: str, scoring: ScoringConfig) -> None:
    with st.container(border=True):
        head = f"#{c.rank} " if c.rank else ""
        st.markdown(f"#### {head}{c.title}")
        badges = [status_badge(c.status), f":violet-badge[{c.origin.replace('_', ' ')}]"]
        if c.human_review_required:
            badges.append(":orange-badge[Human review]")
        if c.validation.eligible:
            badges.append(":green-badge[Validated]")
        st.markdown(" ".join(badges) + f"  `{c.promo_id}`")
        w = scoring.weights
        if c.scores:
            m = st.columns(6)
            m[0].metric("Total", f"{c.scores.total:.1f}/100")
            m[1].metric("Hook", f"{c.scores.hook:.1f}/{w.hook:g}")
            m[2].metric("Cliffhanger", f"{c.scores.cliffhanger:.1f}/{w.cliffhanger:g}")
            m[3].metric("Context", f"{c.scores.context:.1f}/{w.context:g}")
            m[4].metric("Pacing", f"{c.scores.pacing:.1f}/{w.pacing:g}")
            m[5].metric("Duration", f"{c.total_duration_seconds:.1f}s")
        else:
            st.write(f"Duration: {c.total_duration_seconds:.1f}s (not scored)")
        if c.segments:
            st.dataframe(pd.DataFrame([{
                "episode": s.episode_id, "start": s.start_timestamp, "end": s.end_timestamp,
                "duration_s": s.duration_seconds, "start_event": s.start_event_id, "end_event": s.end_event_id,
                "file": s.source_filename.rsplit("/", 1)[-1],
            } for s in c.segments]), width="stretch", hide_index=True)
        col1, col2 = st.columns(2)
        for col, name in ((col1, "hook"), (col2, "cliffhanger")):
            crit = getattr(c, name)
            with col:
                st.markdown(f"**{name.title()}**")
                if crit is None:
                    st.caption("not evaluated")
                    continue
                if crit.description:
                    st.write(crit.description)
                for e in crit.evidence:
                    mark = ":green[verified]" if e.verified else f":red[unverified: {e.note}]"
                    st.markdown(f"> {e.quote}  \n`{e.event_id}` {mark}")
                if crit.reasoning:
                    st.caption(crit.reasoning)
        if c.context:
            st.markdown(f"**Context:** {c.context.summary or c.context.description} ")
            if c.context.reasoning:
                st.caption(c.context.reasoning)
        if c.pacing and c.pacing.reasoning:
            st.markdown(f"**Pacing:** {c.pacing.reasoning}")
        if c.why_it_works:
            st.markdown(f"**Why it works:** {c.why_it_works}")
        if c.weaknesses:
            st.markdown("**Potential weaknesses:**\n" + "\n".join(f"- {x}" for x in c.weaknesses))
        flags = c.rejection_reasons + c.review_reasons + c.warnings
        if flags or c.failure_cases:
            with st.expander("Validation, review reasons and failure cases"):
                for r in c.rejection_reasons:
                    st.error(r)
                for r in c.review_reasons:
                    st.warning(r)
                for r in c.warnings:
                    st.info(r)
                if c.failure_cases:
                    st.write("Failure cases: " + ", ".join(FAILURE_CASES.get(f, f) for f in c.failure_cases))
                for n in c.deterministic_notes:
                    st.caption(f"Note: {n}")
                st.json(c.validation.model_dump(), expanded=False)
        with st.expander("Source script lines"):
            for s in c.segments:
                st.caption(f"{s.episode_id} {s.start_timestamp}-{s.end_timestamp}")
                st.code("\n".join(s.source_evidence) or "(camera only)", language=None, wrap_lines=True)
            st.caption(f"Model calls: generation {', '.join(c.generation_call_ids) or '-'}; evaluation {', '.join(c.evaluation_call_ids) or '-'}")
        with st.form(f"review_{run_id}_{c.promo_id}", border=False):
            latest = st.session_state.get("feedback_latest", {}).get(c.promo_id)
            cols = st.columns([2, 3, 2, 1])
            decision = cols[0].radio("Reviewer decision", ["accepted", "rejected", "needs_editing"], horizontal=True,
                                     index=["accepted", "rejected", "needs_editing"].index(latest.decision) if latest else 0,
                                     format_func=lambda x: x.replace("_", " ").title(), key=f"d_{c.promo_id}")
            notes = cols[1].text_input("Notes", latest.notes if latest else "", key=f"n_{c.promo_id}")
            reviewer = cols[2].text_input("Reviewer", latest.reviewer if latest else "", key=f"r_{c.promo_id}")
            if cols[3].form_submit_button("Save"):
                FEEDBACK.add(FeedbackEntry(run_id=run_id, promo_id=c.promo_id, decision=decision, notes=notes,
                                           reviewer=reviewer, snapshot=snapshot(c)))
                st.session_state["feedback_latest"] = FEEDBACK.latest(run_id)
                st.toast(f"Saved review for {c.promo_id}")
            if latest:
                st.caption(f"Last review: {latest.decision.replace('_', ' ')} by {latest.reviewer or 'anonymous'} at {latest.timestamp[:19]}")


def tab_results(scoring: Optional[ScoringConfig]) -> None:
    res = result()
    if not res:
        st.info("Run the pipeline or load a past run to see results.")
        return
    cands: list[PromoCandidate] = res["candidates"]
    run_scoring: ScoringConfig = res.get("scoring") or scoring or load_scoring_config()
    st.session_state.setdefault("feedback_latest", FEEDBACK.latest(res["run_id"]))
    qa = res["qa_report"]
    st.subheader(f"Results for run {res['run_id']}")
    st.caption(f"Backend: {res['backend']}")
    if res["backend"].startswith("heuristic"):
        st.warning("These results come from the offline heuristic baseline, not an LLM. Treat them as a starting point for editors.")
    ranked = [c for c in cands if c.rank is not None]
    totals = [c.scores.total for c in cands if c.scores]
    m = st.columns(7)
    m[0].metric("Episodes processed", qa.get("episodes_processed", 0))
    m[1].metric("Candidates discovered", len(cands))
    m[2].metric("Eligible promos", len(ranked))
    m[3].metric("Passing all thresholds", sum(1 for c in ranked if c.passes_thresholds))
    m[4].metric("Need human review", sum(1 for c in cands if c.human_review_required))
    m[5].metric("Average score", f"{sum(totals) / len(totals):.1f}" if totals else "-")
    m[6].metric("Processing errors", len(res.get("errors") or qa.get("errors", [])))
    for e in res.get("errors") or []:
        st.error(e)

    if scoring is not None and st.button("Re-score this run with the sidebar weights and thresholds (no new AI calls)"):
        rescore_candidates(cands, scoring, res["episodes"])
        res["scoring"] = scoring
        res["qa_report"] = build_qa_report(cands, scoring, backend=res["backend"], episodes=qa.get("episodes_processed", 0))
        res["qa_report"]["rescored"] = True
        out_dir, raw_dir = Path(res["output_dir"]), Path(res["raw_dir"])
        prev = json.loads((out_dir / "run_config.json").read_text()) if (out_dir / "run_config.json").exists() else {}
        prev.update({"run_id": res["run_id"], "backend": res["backend"], "rescored_scoring_config": scoring.model_dump()})
        export_run(out_dir, raw_dir, run_meta=prev, episodes=res["episodes"], parse_report=res["parse_report"],
                   inventory=[f.model_dump() for f in res["parse_report"].files], candidates=cands,
                   qa_report=res["qa_report"], qa_markdown=qa_report_markdown(res["qa_report"]))
        st.rerun()

    episodes = sorted({e for c in cands for e in c.episode_ids})
    f = st.columns([3, 2, 2, 2, 2])
    sel_eps = f[0].multiselect("Episodes", episodes)
    statuses = ["recommended", "ranked", "duplicate", "rejected", "unevaluated"]
    sel_status = f[1].multiselect("Status", statuses, default=["recommended", "ranked"])
    min_score = f[2].slider("Minimum total score", 0, 100, 0)
    sort_by = f[3].selectbox("Sort by", ["rank", "total", "hook", "cliffhanger", "context", "pacing", "duration", "confidence"])
    eligibility = f[4].selectbox("Eligibility", ["all", "eligible", "not eligible", "needs review", "passes thresholds"])
    view = [c for c in cands
            if (not sel_eps or set(c.episode_ids) & set(sel_eps))
            and (not sel_status or c.status in sel_status)
            and ((c.scores.total if c.scores else 0) >= min_score)
            and (eligibility == "all"
                 or (eligibility == "eligible" and c.validation.eligible)
                 or (eligibility == "not eligible" and not c.validation.eligible)
                 or (eligibility == "needs review" and c.human_review_required)
                 or (eligibility == "passes thresholds" and c.passes_thresholds))]
    keyf = {
        "rank": lambda c: (c.rank is None, c.rank or 0),
        "duration": lambda c: -c.total_duration_seconds,
        "confidence": lambda c: -c.confidence,
    }.get(sort_by, lambda c: -(getattr(c.scores, sort_by) if c.scores else -1))
    view.sort(key=keyf)
    st.caption(f"{len(view)} promo(s) match the filters.")
    with st.expander("Table view"):
        st.dataframe(candidates_frame(view), width="stretch", hide_index=True)
    n = st.number_input("Cards to show", 1, max(1, len(view)), min(10, max(1, len(view))))
    for c in view[: int(n)]:
        render_card(c, res["run_id"], run_scoring)


def tab_quality() -> None:
    res = result()
    if not res:
        st.info("No run loaded.")
        return
    qa, cands = res["qa_report"], res["candidates"]
    st.subheader("Quality assurance")
    st.caption(qa.get("disclaimer", ""))
    c = st.columns(4)
    c[0].metric("Evaluated", qa.get("candidates_evaluated", 0))
    c[1].metric("Rejected", qa.get("status_counts", {}).get("rejected", 0))
    c[2].metric("Duplicates", qa.get("status_counts", {}).get("duplicate", 0))
    c[3].metric("LLM calls", qa.get("llm_calls", "-"))
    fc = qa.get("failure_case_counts", {})
    if fc:
        st.markdown("**Failure cases detected**")
        st.bar_chart(pd.DataFrame({"count": fc}), horizontal=True)
    st.markdown("**Selection and rejection rationale**")
    st.markdown(qa_report_markdown(qa))
    st.markdown("**Reviewer feedback for this run**")
    entries = FEEDBACK.load(res["run_id"])
    if entries:
        st.json(FEEDBACK.agreement_summary(res["run_id"], cands))
        st.dataframe(pd.DataFrame([e.model_dump(exclude={"snapshot"}) for e in entries]), width="stretch", hide_index=True)
    else:
        st.caption("No reviews saved yet. Use the form on each promo card.")


def tab_export() -> None:
    res = result()
    if not res:
        st.info("No run loaded.")
        return
    out_dir, raw_dir = Path(res["output_dir"]), Path(res["raw_dir"])
    cands = res["candidates"]
    st.subheader(f"Export run {res['run_id']}")
    st.caption(f"Artifacts on disk: {out_dir}")
    cols = st.columns(3)
    cols[0].download_button("Complete JSON results", json.dumps({"run_id": res["run_id"], "backend": res["backend"],
                            "candidates": [c.model_dump(mode="json") for c in cands]}, indent=2),
                            f"{res['run_id']}_candidates_all.json", "application/json", width="stretch")
    ranked = [c for c in cands if c.rank is not None]
    cols[1].download_button("Ranked promos CSV", candidates_frame(ranked).to_csv(index=False),
                            f"{res['run_id']}_ranked_promos.csv", "text/csv", width="stretch")
    cols[2].download_button("Promo segments CSV", segments_frame(cands).to_csv(index=False),
                            f"{res['run_id']}_promo_segments.csv", "text/csv", width="stretch")
    cols = st.columns(3)
    raw = raw_dir / "raw_responses.jsonl"
    cols[0].download_button("Raw model outputs (JSONL)", raw.read_bytes() if raw.exists() else b"",
                            f"{res['run_id']}_raw_responses.jsonl", "application/jsonl", width="stretch",
                            disabled=not raw.exists())
    cols[1].download_button("Validation report", json.dumps(validation_report(cands), indent=2),
                            f"{res['run_id']}_validation_report.json", "application/json", width="stretch")
    cols[2].download_button("Parse report", res["parse_report"].model_dump_json(indent=2),
                            f"{res['run_id']}_parse_report.json", "application/json", width="stretch")
    st.download_button("ZIP of the entire run", build_zip(out_dir, raw_dir), f"{res['run_id']}_artifacts.zip",
                       "application/zip", type="primary", width="stretch")
    fb = FEEDBACK.load(res["run_id"])
    if fb:
        st.download_button("Reviewer feedback (JSONL)", "\n".join(e.model_dump_json() for e in fb),
                           f"{res['run_id']}_reviewer_feedback.jsonl", width="stretch")


def tab_past_runs() -> None:
    st.subheader("Past runs")
    runs = list_runs(APP_CFG)
    if not runs:
        st.info("No completed runs found yet.")
        return
    sel = st.selectbox("Run", [p.name for p in runs])
    run_dir = next(p for p in runs if p.name == sel)
    cfg = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
    st.caption(f"Backend: {cfg.get('backend', '?')} | created {cfg.get('created_at', '?')} | uploads: "
               f"{', '.join(u['name'] for u in cfg.get('uploads', []))}")
    if st.button("Load this run"):
        data = load_run(run_dir)
        sc = cfg.get("rescored_scoring_config") or cfg.get("scoring_config")
        set_result({
            "run_id": data["run_id"], "output_dir": str(run_dir),
            "raw_dir": str(APP_CFG.resolve(APP_CFG.raw_output_dir) / data["run_id"]), "backend": data["backend"],
            "candidates": data["candidates"], "episodes": data["episodes"], "parse_report": data["parse_report"],
            "qa_report": data["qa_report"], "errors": data["qa_report"].get("errors", []),
            "scoring": ScoringConfig(**sc) if sc else None,
        })
        st.session_state.pop("feedback_latest", None)
        st.success(f"Loaded {data['run_id']}. Open the Results tab.")


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Microdrama Promo Discovery Engine")
    st.caption("Finds, validates and ranks promo clips from screenplay timelines. Timestamps are always taken from the source; "
               "the AI only judges story quality.")
    scoring, llm, errors = sidebar_config()
    tabs = st.tabs(["Upload", "Run", "Results", "Quality and review", "Export", "Past runs"])
    with tabs[0]:
        uploads = tab_upload()
    with tabs[1]:
        tab_run(uploads, scoring, llm, errors)
    with tabs[2]:
        tab_results(scoring)
    with tabs[3]:
        tab_quality()
    with tabs[4]:
        tab_export()
    with tabs[5]:
        tab_past_runs()


main()
