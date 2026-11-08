"""Command-line runner (headless alternative to the Streamlit UI).

Examples
  python cli.py run dataset.zip
  python cli.py run ep1.txt ep2.pdf --provider heuristic
  python cli.py run dataset.zip --provider ollama --model qwen3:4b --max-episodes 5
  python cli.py parse dataset.zip          # parsing/normalisation report only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.settings import load_app_config, load_llm_settings, load_scoring_config
from core.normalizer import load_dataset
from core.pipeline import PromoPipeline


def _read(paths: list[str]) -> list[tuple[str, bytes]]:
    out = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            sys.exit(f"not a file: {p}")
        out.append((path.name, path.read_bytes()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI promo discovery and ranking engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "parse"):
        sp = sub.add_parser(name)
        sp.add_argument("files", nargs="+")
        sp.add_argument("--scoring-config")
        sp.add_argument("--app-config")
        if name == "run":
            sp.add_argument("--provider")
            sp.add_argument("--model")
            sp.add_argument("--api-base")
            sp.add_argument("--concurrency", type=int)
            sp.add_argument("--max-episodes", type=int)
    args = ap.parse_args(argv)
    app = load_app_config(args.app_config)
    uploads = _read(args.files)

    if args.cmd == "parse":
        episodes, report, _ = load_dataset(uploads, app)
        print(json.dumps({
            "episodes": report.episodes_detected, "events": report.events_total, "untimed_events": report.events_untimed,
            "files": [f.model_dump(include={"display_name", "status", "message", "parser", "episodes_found"}) for f in report.files],
            "issues": [i.model_dump(mode="json") for i in report.issues],
        }, indent=2))
        return 0 if episodes else 1

    scoring = load_scoring_config(args.scoring_config)
    llm = load_llm_settings(provider=args.provider, model=args.model, api_base=args.api_base, concurrency=args.concurrency)
    problems = llm.missing_requirements()
    if problems:
        sys.exit("LLM configuration incomplete: " + "; ".join(problems) + " (or use --provider heuristic)")

    def progress(stage: str, frac: float, msg: str) -> None:
        print(f"[{stage:<22}] {frac:>4.0%} {msg}", flush=True)

    result = PromoPipeline(scoring, llm, app, progress=progress).run(uploads, max_episodes=args.max_episodes)
    qa = result.qa_report
    print(f"\nRun {result.run_id} ({result.backend}) finished in {result.elapsed_seconds:.1f}s")
    print(f"Episodes: {qa['episodes_processed']}  candidates: {qa['candidates_total']}  ranked: {qa['eligible_ranked']}  "
          f"review: {qa['human_review_required']}  mean score: {qa['score_stats']['mean']}")
    for c in result.ranked[: scoring.top_k_recommendations]:
        segs = "; ".join(f"{s.episode_id} {s.start_timestamp}-{s.end_timestamp}" for s in c.segments)
        flag = " [review]" if c.human_review_required else ""
        print(f"  #{c.rank:<2} {c.scores.total:5.1f} {c.total_duration_seconds:5.1f}s {c.promo_id} {segs} :: {c.title}{flag}")
    for e in result.errors:
        print("ERROR:", e)
    print(f"Artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
