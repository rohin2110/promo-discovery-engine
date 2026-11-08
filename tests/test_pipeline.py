import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from config.settings import LLMSettings, ScoringConfig
from core.pipeline import PromoPipeline, load_run, rescore_candidates
from llm.provider import LLMProviderError, LLMRequest, MockProvider, OpenAICompatibleProvider
from llm.structured_output import RawResponseRecorder, StructuredLLMClient, extract_json
from models.schemas import EventType, GenerationResponse
from tests.conftest import make_dataset, make_screenplay, zip_bytes

STORY = {EventType.dialogue, EventType.action, EventType.other, EventType.description}


# ---------------------------------------------------------------------------
# Scripted model helpers
# ---------------------------------------------------------------------------
def story_events(ep):
    return [e for e in ep.timed_events if e.event_type in STORY]


def window(ep, start_idx, seconds):
    st = story_events(ep)
    s = st[start_idx]
    e = min((x for x in st if x.order > s.order), key=lambda x: abs(x.end_seconds - s.start_seconds - seconds))
    return {"start_event_id": s.event_id, "end_event_id": e.event_id}


def single_gen(ep, req):
    return {"candidates": [{"title": f"{ep.episode_id} clash", "segments": [window(ep, 1, 30)],
                            "opening_hook": "accusation", "central_conflict": "a lie", "ending_cliffhanger": "who killed her",
                            "promotional_reason": "open question"}]}


def evaluation(ratings=(8, 8, 8, 8), quote_ok=True, coherent=None):
    def plan(clip, req):
        hook_ev = next((e for e in clip.hook_events if e.event_type in STORY), clip.hook_events[0])
        cliff_ev = [e for e in clip.cliff_events if e.event_type in STORY][-1]
        hq = hook_ev.text if quote_ok else "An invented line nobody ever said."
        return {
            "title": "Scripted", "confidence": 0.8, "weaknesses": ["scripted"], "why_it_works": "scripted rationale",
            "hook": {"rating": ratings[0], "description": "d", "evidence": [{"event_id": hook_ev.event_id, "quote": hq}], "reasoning": "r"},
            "cliffhanger": {"rating": ratings[1], "evidence": [{"event_id": cliff_ev.event_id, "quote": cliff_ev.text}], "reasoning": "r"},
            "context": {"rating": ratings[2], "summary": "two people argue", "evidence": [], "reasoning": "r"},
            "pacing": {"rating": ratings[3], "reasoning": "r"},
            "cross_episode_coherent": coherent if clip.cross_episode else None,
        }
    return plan


def responder(gen=single_gen, ev=None, cross=None, log=None):
    ev = ev or evaluation()

    def respond(req: LLMRequest) -> str:
        if log is not None:
            log.append(req)
        ctx = req.context
        if req.task.startswith("candidate_generation"):
            return json.dumps(gen(ctx["episode"], req))
        if req.task.startswith("cross_episode"):
            return json.dumps(cross(ctx["episodes"], req) if cross else {"candidates": []})
        if req.task.startswith("promo_evaluation"):
            return json.dumps(ev(ctx["clip"], req))
        raise AssertionError(req.task)
    return respond


def run(app_cfg, uploads, llm, respond=None, cfg=None):
    provider = MockProvider(llm, respond) if respond else None
    return PromoPipeline(cfg or ScoringConfig(), llm, app_cfg, provider=provider).run(uploads)


def raw_calls(res):
    p = res.raw_dir / "raw_responses.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


# ---------------------------------------------------------------------------
# End-to-end with a scripted model
# ---------------------------------------------------------------------------
def test_valid_multi_segment_promo(app_cfg, mock_llm):
    def gen(ep, req):
        st = story_events(ep)
        a = window(ep, 1, 12)
        b_start = next(i for i, e in enumerate(st) if e.start_seconds > 30)
        return {"candidates": [{"title": "two cuts", "segments": [a, window(ep, b_start, 12)]}]}
    res = run(app_cfg, make_dataset(2), mock_llm, responder(gen=gen), ScoringConfig(allow_cross_episode_segments=False))
    c = res.ranked[0]
    assert c.origin == "multi_segment_same_episode" and len(c.segments) == 2
    assert c.total_duration_seconds == pytest.approx(sum(s.duration_seconds for s in c.segments))
    assert c.validation.eligible and c.scores.total == pytest.approx(80.0)
    assert c.ranking_score == pytest.approx(78.0)  # one extra segment penalty
    calls = {r["call_id"]: r for r in raw_calls(res)}
    assert set(c.generation_call_ids + c.evaluation_call_ids) <= set(calls)
    assert all(r["status"] == "ok" for r in calls.values())


def test_repair_loop_fixes_hallucinated_events(app_cfg, mock_llm):
    def gen(ep, req):
        if req.task.endswith("_repair"):
            assert "do not exist" in req.messages[-1]["content"]
            return single_gen(ep, req)
        return {"candidates": [{"title": "ghost", "segments": [{"start_event_id": f"{ep.episode_id}-E900", "end_event_id": f"{ep.episode_id}-E950"}]}]}
    res = run(app_cfg, make_dataset(1), mock_llm, responder(gen=gen))
    rejected = [c for c in res.candidates if c.status == "rejected"]
    assert len(rejected) == 1 and "hallucinated_source" in rejected[0].failure_cases
    assert len(res.ranked) == 1 and "repaired after validation feedback" in " ".join(res.ranked[0].warnings)
    assert any(r["task"] == "candidate_generation_repair" for r in raw_calls(res))


def test_duration_repair_gets_hints(app_cfg, mock_llm):
    def gen(ep, req):
        if req.task.endswith("_repair"):
            assert "valid end events range from" in req.messages[-1]["content"]
            return single_gen(ep, req)
        return {"candidates": [{"title": "too short", "segments": [window(ep, 1, 8)]}]}
    res = run(app_cfg, make_dataset(1), mock_llm, responder(gen=gen))
    assert len(res.ranked) == 1
    assert any("duration_out_of_range" in c.failure_cases for c in res.candidates if c.status == "rejected")


def test_invalid_json_is_retried(app_cfg, mock_llm):
    state = {"n": 0}
    inner = responder()

    def respond(req):
        if req.task == "candidate_generation" and state["n"] == 0:
            state["n"] += 1
            return "Sure! Here are some ideas: (not JSON)"
        return inner(req)
    res = run(app_cfg, make_dataset(1), mock_llm, respond)
    statuses = [r["status"] for r in raw_calls(res)]
    assert statuses[0] == "invalid_json" and "ok" in statuses
    assert len(res.ranked) == 1


def test_strong_hook_with_no_cliffhanger(app_cfg, mock_llm):
    res = run(app_cfg, make_dataset(1), mock_llm, responder(ev=evaluation(ratings=(9, 1, 8, 8))))
    c = res.ranked[0]
    assert "strong_hook_weak_ending" in c.failure_cases
    assert not c.passes_thresholds and c.human_review_required
    assert any("cliffhanger score" in r for r in c.review_reasons)


def test_hallucinated_evidence_is_retried_then_flagged(app_cfg, mock_llm):
    res = run(app_cfg, make_dataset(1), mock_llm, responder(ev=evaluation(quote_ok=False)))
    c = res.ranked[0]
    assert len(c.evaluation_call_ids) == 2  # original + one evidence-repair retry
    assert "unverified_evidence" in c.failure_cases and c.human_review_required
    assert not c.validation.evidence_verified
    assert any(not e.verified for e in c.hook.evidence)


def test_no_suitable_promotional_moments(app_cfg, mock_llm):
    res = run(app_cfg, make_dataset(2, dramatic=False), mock_llm, responder(gen=lambda ep, req: {"candidates": []}))
    assert res.candidates == [] and res.ranked == []
    assert res.qa_report["candidates_total"] == 0
    assert (res.output_dir / "ranked_promos.csv").exists()


def test_cross_episode_promo(app_cfg, mock_llm):
    def cross(episodes, req):
        a, b = episodes
        return {"candidates": [{"title": "across", "segments": [window(a, 1, 12), window(b, 1, 15)],
                                "narrative_connection": "same two characters, same accusation"}]}
    res = run(app_cfg, make_dataset(3), mock_llm, responder(cross=cross, ev=evaluation(coherent=True)),
              ScoringConfig(cross_episode_max_windows=2))
    cross_c = [c for c in res.candidates if c.origin == "cross_episode"]
    assert cross_c and all(c.validation.eligible for c in cross_c)
    c = cross_c[0]
    assert c.cross_episode_coherent is True and len(c.episode_ids) == 2
    assert c.ranking_score == pytest.approx(c.scores.total - 4.0)  # segment + cross penalties
    assert any("characters present in every segment" in n for n in c.deterministic_notes)


def test_incoherent_cross_episode_flagged(app_cfg, mock_llm):
    files = {"e1.txt": make_screenplay(1), "e2.txt": make_screenplay(2, names=("CASEY", "DREW"))}

    def cross(episodes, req):
        a, b = episodes
        return {"candidates": [{"title": "forced", "segments": [window(a, 1, 12), window(b, 1, 15)]}]}
    res = run(app_cfg, [("d.zip", zip_bytes(files))], mock_llm, responder(cross=cross, ev=evaluation(coherent=False)))
    c = next(c for c in res.candidates if c.origin == "cross_episode")
    assert "incoherent_cross_episode" in c.failure_cases and c.human_review_required


# ---------------------------------------------------------------------------
# Heuristic backend: real end-to-end runs without network
# ---------------------------------------------------------------------------
def test_heuristic_small_dataset_end_to_end(app_cfg, heuristic_llm):
    res = run(app_cfg, make_dataset(3), heuristic_llm)
    assert res.qa_report["episodes_processed"] == 3
    assert res.ranked, "heuristic should find promos in dramatic episodes"
    for c in res.ranked:
        assert 20 <= c.total_duration_seconds <= 60
        assert c.validation.source_verified and c.human_review_required  # baseline is always reviewed
    expected = {"run_config.json", "input_inventory.json", "parse_report.json", "normalized_episodes.json",
                "candidates_all.json", "ranked_promos.csv", "promo_segments.csv", "validation_report.json",
                "qa_report.json", "qa_report.md", "run.log", "candidates_summary.csv"}
    assert expected <= {p.name for p in res.output_dir.iterdir()}
    with zipfile.ZipFile(res.files["artifacts_zip"]) as zf:
        names = set(zf.namelist())
    assert "outputs/ranked_promos.csv" in names and "raw_outputs/raw_responses.jsonl" in names
    cfg = json.loads((res.output_dir / "run_config.json").read_text())
    assert "api_key" not in cfg["llm_settings"] and cfg["run_id"] == res.run_id
    loaded = load_run(res.output_dir)
    assert len(loaded["candidates"]) == len(res.candidates)


def test_dataset_with_over_100_episodes(app_cfg, heuristic_llm):
    cfg = ScoringConfig(candidates_per_episode=1, cross_episode_max_windows=3)
    res = run(app_cfg, make_dataset(120, n_lines=10), heuristic_llm, cfg=cfg)
    assert res.qa_report["episodes_processed"] == 120
    assert {e.episode_id for e in res.episodes} >= {"EP001", "EP100", "EP120"}
    assert len([c for c in res.candidates if c.origin == "single_segment"]) == 120


def test_repeat_processing_is_isolated(app_cfg, mock_llm):
    log_a, log_b = [], []
    res_a = run(app_cfg, make_dataset(2), mock_llm, responder(log=log_a))
    data_b = [("other.zip", zip_bytes({f"x/ep{i}.txt": make_screenplay(i + 10, names=("CASEY", "DREW")) for i in range(3)}))]
    res_b = run(app_cfg, data_b, mock_llm, responder(log=log_b))
    assert res_a.run_id != res_b.run_id and res_a.output_dir != res_b.output_dir
    assert {e.episode_id for e in res_b.episodes} == {"EP010", "EP011", "EP012"}
    assert all("MORGAN" not in m["content"] for r in log_b for m in r.messages)
    assert all(eid.startswith(("EP010", "EP011", "EP012")) for c in res_b.candidates for s in c.segments for eid in s.source_event_ids)
    assert (res_a.output_dir / "candidates_all.json").exists()  # first run untouched


def test_rescore_without_new_calls(app_cfg, heuristic_llm):
    res = run(app_cfg, make_dataset(3), heuristic_llm)
    n_calls = len(raw_calls(res))
    before = {c.promo_id: c.scores.total for c in res.candidates if c.scores}
    cfg = ScoringConfig(weights={"hook": 10, "cliffhanger": 10, "context": 40, "pacing": 40},
                        minimum_scores={"hook": 5, "cliffhanger": 5, "context": 10, "total": 50})
    rescore_candidates(res.candidates, cfg, res.episodes)
    after = {c.promo_id: c.scores.total for c in res.candidates if c.scores}
    assert before.keys() == after.keys() and before != after
    c = next(c for c in res.candidates if c.scores)
    assert c.scores.hook == pytest.approx(c.hook.rating)  # weight 10 -> rating == points
    assert len(raw_calls(res)) == n_calls


def test_dataset_without_usable_timestamps(app_cfg, heuristic_llm):
    text = "EPISODE NUMBER: Episode 1\nSCENE 01\nScene heading: INT. A – DAY\n\n**ALEX:**\n\"No times at all.\"\n"
    res = run(app_cfg, [("e1.txt", text.encode())], heuristic_llm)
    assert res.candidates == [] and any("no episodes with usable timestamps" in e for e in res.errors)


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP client against a local stub server
# ---------------------------------------------------------------------------
class _Stub(BaseHTTPRequestHandler):
    script: list = []
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
        status, payload = _Stub.script.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Stub.seen = []
    yield srv
    srv.shutdown()


def _ok(content):
    return 200, {"model": "stub-1", "choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"total_tokens": 5}}


def test_openai_compatible_retries_and_downgrades(stub_server, tmp_path):
    _Stub.script = [
        (500, {"error": "overloaded"}),
        (400, {"error": {"message": "response_format json_object is not supported by this model"}}),
        _ok('```json\n{"candidates": []}\n```'),
    ]
    s = LLMSettings(provider="openai_compatible", model="stub-1", api_key="sk-test",
                    api_base=f"http://127.0.0.1:{stub_server.server_port}/v1", max_retries=2, seed=7)
    prov = OpenAICompatibleProvider(s)
    prov.backoff_base = 0
    rec = RawResponseRecorder(tmp_path / "raw.jsonl")
    res = StructuredLLMClient(prov, s, rec).call("candidate_generation", [{"role": "user", "content": "hi"}], GenerationResponse)
    assert res.ok and res.parsed.candidates == []
    assert [x["path"] for x in _Stub.seen] == ["/v1/chat/completions"] * 3
    assert _Stub.seen[0]["auth"] == "Bearer sk-test" and _Stub.seen[0]["body"]["seed"] == 7
    assert "response_format" in _Stub.seen[1]["body"] and "response_format" not in _Stub.seen[2]["body"]
    line = json.loads((tmp_path / "raw.jsonl").read_text().splitlines()[0])
    assert line["status"] == "ok" and line["model"] == "stub-1" and "sk-test" not in json.dumps(line)


def test_provider_errors_are_recorded_and_flagged(stub_server, app_cfg):
    _Stub.script = [(401, {"error": "bad key"})] * 10
    s = LLMSettings(provider="openai_compatible", model="stub-1", api_key="sk-x",
                    api_base=f"http://127.0.0.1:{stub_server.server_port}/v1", max_retries=0)
    prov = OpenAICompatibleProvider(s)
    with pytest.raises(LLMProviderError):
        prov.complete(LLMRequest(task="t", messages=[]))
    res = PromoPipeline(ScoringConfig(), s, app_cfg, provider=prov).run(make_dataset(1))
    assert res.ranked == [] and res.candidates[0].status == "rejected"
    assert "generation failed" in res.candidates[0].rejection_reasons[0]
    assert raw_calls(res)[0]["status"] == "provider_error"


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('<think>hmm</think>\n{"a": 2}') == {"a": 2}
    assert extract_json('Here you go:\n```json\n{"a": 3}\n```') == {"a": 3}
    assert extract_json('prefix {"a": 4} suffix') == {"a": 4}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_ollama_native_provider_sets_context_window(stub_server, tmp_path):
    from llm.provider import OllamaProvider
    _Stub.script = [(200, {"model": "local-1", "message": {"role": "assistant", "content": '{"candidates": []}'},
                           "done_reason": "stop", "prompt_eval_count": 1200, "eval_count": 9})]
    s = LLMSettings(provider="ollama", model="local-1", api_base=f"http://127.0.0.1:{stub_server.server_port}/v1",
                    context_window=12000, seed=3, extra_body={"think": False})
    rec = RawResponseRecorder(tmp_path / "raw.jsonl")
    res = StructuredLLMClient(OllamaProvider(s), s, rec).call("candidate_generation", [{"role": "user", "content": "hi"}], GenerationResponse)
    assert res.ok
    req = _Stub.seen[0]
    assert req["path"] == "/api/chat" and req["auth"] is None
    assert req["body"]["options"] == {"temperature": 0.1, "num_predict": 4096, "num_ctx": 12000, "seed": 3}
    assert req["body"]["format"] == "json" and req["body"]["stream"] is False and req["body"]["think"] is False


def test_parallel_requests_retry_after_capability_downgrade(stub_server):
    """A request sent with `seed` before another thread disabled it must still be retried."""
    _Stub.script = [(400, {"error": {"message": 'Unknown name "seed": Cannot find field.'}}), _ok('{"candidates": []}')]
    s = LLMSettings(provider="openai_compatible", model="m", api_key="k",
                    api_base=f"http://127.0.0.1:{stub_server.server_port}/v1", max_retries=0, seed=1)
    prov = OpenAICompatibleProvider(s)
    prov._seed_supported = False  # simulate: another thread already downgraded...
    body_with_seed = {"seed": 1}  # ...but this request was built before that
    assert prov._downgrade('Unknown name "seed"', body_with_seed) is True
    assert prov._downgrade('Unknown name "seed"', {}) is False
    prov2 = OpenAICompatibleProvider(s)
    assert prov2.complete(LLMRequest(task="t", messages=[])).text == '{"candidates": []}'
    assert "seed" in _Stub.seen[0]["body"] and "seed" not in _Stub.seen[1]["body"]


def test_daily_quota_stops_further_calls(stub_server, app_cfg):
    from llm.provider import is_hard_quota
    msg = {"error": {"message": "Quota exceeded for metric: free_tier_requests, limit: 20, description: requests per day. Please retry in 22h38m42s."}}
    _Stub.script = [(429, msg)] * 50
    s = LLMSettings(provider="openai_compatible", model="m", api_key="k",
                    api_base=f"http://127.0.0.1:{stub_server.server_port}/v1", max_retries=5)
    prov = OpenAICompatibleProvider(s)
    prov.backoff_base = 0
    res = PromoPipeline(ScoringConfig(), s, app_cfg, provider=prov).run(make_dataset(3))
    assert len(_Stub.seen) == 1  # no retries, no further calls
    assert any("quota exhausted" in e for e in res.errors)
    assert all(c.status == "rejected" for c in res.candidates) and len(res.candidates) == 3
    assert not is_hard_quota('{"error": "rate limit, retryDelay": "20s"}')
    assert is_hard_quota('"retryDelay": "81522s"')
