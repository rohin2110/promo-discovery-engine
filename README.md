# Microdrama Promo Discovery and Ranking Engine

Finds, validates and ranks 20 to 60 second promo clips from screenplay timelines. An LLM judges story quality. Python handles everything that can be computed: parsing, timestamps, durations, weighted scores, eligibility, de-duplication and ranking.

The engine is data-independent. Nothing about a show, its characters, episode numbers, scene names, file names or timestamps is hardcoded. A new dataset needs no code changes.

## Contents

1. [How it works](#how-it-works)
2. [Install](#install)
3. [Configure the LLM](#configure-the-llm)
4. [Run the app](#run-the-app)
5. [Upload a new dataset](#upload-a-new-dataset)
6. [Outputs](#outputs)
7. [Scoring, eligibility and ranking](#scoring-eligibility-and-ranking)
8. [Hallucination prevention](#hallucination-prevention)
9. [Tests](#tests)
10. [Results on the provided dataset](#results-on-the-provided-dataset)
11. [Known limitations and failure cases](#known-limitations-and-failure-cases)
12. [Free deployment](#free-deployment)
13. [Project structure](#project-structure)

## How it works

```
upload ──► ingestion (safe ZIP/size checks)
       ──► parsers (TXT/MD/PDF/DOCX/JSON/CSV/SRT) ──► normaliser (common schema, seconds, IDs, flags)
       ──► Step 1  per-episode candidate generation (LLM picks EVENT ID ranges)
       ──► deterministic validation ──► repair prompt with concrete errors and hints (retry)
       ──► Step 2  standalone evaluation (LLM sees ONLY the clip; 0-10 rubric per criterion)
       ──► evidence verification (quotes must exist verbatim) ──► retry, then flag
       ──► Step 3  cross-episode analysis (adjacent episodes, shared-character check)
       ──► Python scoring, thresholds, near-duplicate removal, ranking, QA report ──► exports
```

Key design choices:

* **The model never writes a timestamp.** It proposes segments as `start_event_id` and `end_event_id` from the normalised timeline. Python looks the events up and takes their start and end times from the source. A clip cannot cite footage that does not exist. If the model names an unknown event, the clip is rejected or sent back for repair.
* **Context is judged on the clip alone.** The evaluator receives only the events inside the edit, plus the hook window (first 3 s) and cliffhanger window (final 5 s) computed by Python. It never sees the logline, continuity notes or other episodes, so it cannot use prior knowledge to rate a confusing clip as clear.
* **Ratings, not points.** The model rates each criterion from 0 to 10. Python converts ratings to points with the configured weights. Changing weights or thresholds re-scores a finished run without new model calls (the "Re-score" button in the UI).
* **Interchangeable backends.** The same pipeline runs with any OpenAI-compatible API, a local Ollama model, or the built-in offline `heuristic` baseline. A scripted `mock` provider exists for tests.

## Install

Requires Python 3.11 or newer.

```bash
cd promo_discovery_engine
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env
```

## Configure the LLM

All model settings live in `.env` (or real environment variables). Business logic never reads them directly.

| Variable | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `openai_compatible`, `ollama` or `heuristic` | `heuristic` |
| `LLM_MODEL` | Model name at the provider | (empty) |
| `LLM_API_KEY` | API key. Never written to run artifacts. | (empty) |
| `LLM_API_BASE` | Base URL ending in `/v1` | provider default |
| `LLM_TEMPERATURE` | Sampling temperature | `0.1` |
| `LLM_MAX_TOKENS` | Max output tokens | `4096` |
| `LLM_TIMEOUT` | Seconds per request | `90` |
| `LLM_MAX_RETRIES` | Transport retries and JSON repair attempts | `2` |
| `LLM_SEED` | Fixed seed, sent when supported | `42` |
| `LLM_RESPONSE_FORMAT` | `json_object`, `json_schema` or `none` | `json_object` |
| `LLM_CONCURRENCY` | Parallel requests | `1` |
| `LLM_CONTEXT_WINDOW` | Context tokens requested from Ollama (`num_ctx`) | `16384` |
| `LLM_EXTRA_BODY` | JSON merged into each request body | (empty) |
| `LLM_SAVE_RAW_RESPONSES` | Store every request and response | `true` |

If a provider rejects `response_format` or `seed`, the client drops that field and retries automatically. If it reports a long-term quota limit (for example a daily cap), the run stops calling the model, finishes with what it has, and lists the error.

Examples:

```bash
# OpenAI
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=sk-...

# Groq (free tier available)
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=gsk_...

# Google Gemini (OpenAI-compatible endpoint, free tier available).
# Free-tier quotas are per model and per day; Flash-Lite allowed a full
# 50-episode run (~210 calls), Flash allowed only 20 requests per day.
# Gemini rejects `seed`; the client drops it automatically.
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-3.1-flash-lite
LLM_API_KEY=...
LLM_MAX_RETRIES=5
LLM_CONCURRENCY=2

# Anthropic Claude (OpenAI SDK compatibility endpoint)
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://api.anthropic.com/v1
LLM_MODEL=claude-sonnet-5-5
LLM_API_KEY=sk-ant-...

# Local Ollama (no key). Pick a model that fits in RAM.
# Uses Ollama's native /api/chat so the context window can be raised;
# the OpenAI-compatible endpoint would silently truncate long timelines.
LLM_PROVIDER=ollama
LLM_API_BASE=http://localhost:11434
LLM_MODEL=qwen2.5:7b
LLM_CONTEXT_WINDOW=16384
LLM_TIMEOUT=600

# Offline baseline, no model at all
LLM_PROVIDER=heuristic
```

Model names change often. Check your provider's current list. Every setting can also be changed in the app sidebar for a single session. A key typed into the sidebar is used in memory only.

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. The tabs are:

* **Upload**: drop files or ZIPs. Shows files parsed, episodes detected, usable episodes, events, and every parsing error and warning. You can inspect any episode's normalised events.
* **Run**: starts the pipeline with live progress for file processing, episode extraction, event normalisation, candidate generation, candidate validation, promo evaluation, cross-episode analysis, final ranking and export. It can limit the run to the first N episodes for a quick trial.
* **Results**: headline metrics, filters by episode, status, score and eligibility, sorting, and a detailed card per promo. Each card shows rank, segment timestamps, duration, all four scores, verified hook and cliffhanger evidence, why it works, weaknesses, validation details and the source script lines. Reviewers can mark a promo Accepted, Rejected or Needs editing.
* **Quality and review**: failure-case counts, the selection and rejection rationale, and reviewer feedback compared with the system's recommendations.
* **Export**: JSON, CSVs, raw model outputs, validation and parse reports, and a ZIP of the whole run.
* **Past runs**: reload any earlier run.

Headless alternative:

```bash
python cli.py parse dataset.zip                     # parsing report only
python cli.py run dataset.zip                       # uses .env
python cli.py run dataset.zip --provider heuristic  # offline
python cli.py run dataset.zip --provider ollama --model qwen2.5:7b --max-episodes 5
```

## Upload a new dataset

Upload a ZIP (nested folders and nested ZIPs are fine) or individual files. Supported formats are TXT, MD, PDF (text-based), DOCX, JSON, CSV and SRT.

What the parser recognises without configuration:

* **Metadata** lines such as `EPISODE NUMBER:`, `Episode:`, `Title:`, `Logline:`/`Synopsis:`, `Continuity notes:`, or a heading like `# Episode 5: The Letter`.
* **Scenes** such as `SCENE 01`, `## Scene 3`, `INT./EXT.` sluglines, `Scene heading:`, `Scene start:`/`Scene end:`.
* **Event blocks**: a label line (`**CAMERA:**`, `ACTION:`, `**NAME (emotion):**`, `NAME:`), then a time range, then text. Labels become `camera`, `action`, `description`, `other` (sound, music, captions) or `dialogue` with speaker and parenthetical.
* **Time ranges** like `Start: (00:01.00) | End: (00:04.80)`, `[00:01 - 00:04]` or `00:00:01,000 --> 00:00:04,800`, on their own line or in front of the text.
* **Timestamps** in `MM:SS.ss`, `HH:MM:SS.mmm`, `HH:MM:SS,mmm`, `12.5s`, `1m 05s` and `HH:MM:SS:FF`. Frame timecodes are treated as ambiguous unless `frames_per_second` is set in `config/app_config.json`.

Episode identifiers come from metadata first, then the filename (`episode-012`, `ep12`, `E12`, leading number). Without either, the episode gets a filename-based ID and a warning. Two files claiming the same episode number get distinct IDs (`EP002`, `EP002-B`) and a warning.

**JSON and CSV** are read either as the engine's own normalised episode JSON or as event rows. Column names are matched through `field_mapping` in `config/app_config.json`. If the required text, start and end columns cannot be mapped, the file fails with a message listing the columns found. Add your column names to the mapping. No code changes are needed.

Files that cannot be parsed are listed with the reason. They are never silently skipped. Missing or invalid timestamps are never invented. Those events are kept, flagged, and cannot be used as cut points. Episodes without usable timestamps are excluded from promo detection with a warning.

Upload safety: path traversal, absolute paths, symlinks and encrypted members are rejected. Limits apply to upload size, per-file size, total uncompressed size, file count, nesting depth and compression ratio (zip-bomb guard). All limits are in `config/app_config.json`.

Every upload creates a new run with its own ID and directories. Nothing from a previous dataset is reused.

## Outputs

Each run writes to `outputs/<run_id>/`, with raw model outputs in `raw_outputs/<run_id>/`:

| File | Content |
|---|---|
| `candidates_all.json` | Every candidate: recommended, ranked, duplicate and rejected, with full detail |
| `ranked_promos.csv` | Eligible promos in rank order |
| `candidates_summary.csv` | One row per candidate, including rejected ones |
| `promo_segments.csv` | One row per segment with source file, timestamps, seconds and event IDs |
| `raw_outputs/<run_id>/raw_responses.jsonl` | Every model request and response, including failures, with call IDs |
| `parse_report.json` | File inventory, parser used, errors, warnings and per-episode summary |
| `normalized_episodes.json` | The parsed dataset in the common schema |
| `input_inventory.json` | Uploaded files with size, SHA-256 and status |
| `run_config.json` | Scoring parameters, model settings (no key), prompt version, uploads |
| `validation_report.json` | Per-candidate validation, rejection and review reasons, unverified evidence |
| `qa_report.json` and `qa_report.md` | Run summary, why promos were selected or rejected, failure cases |
| `run.log` | Execution log |
| `<run_id>_artifacts.zip` | Everything above in one file |

Every segment can be traced to its source. It records the episode, source file, original timestamp strings, seconds, start and end event IDs and the IDs of every event inside it. Candidates list the call IDs of the model responses that produced and scored them.

Reviewer decisions go to `feedback/reviewer_feedback.jsonl`, separate from run outputs, with a snapshot of the promo as reviewed.

## Scoring, eligibility and ranking

All parameters are in `config/scoring_config.json`. They can also be changed in the sidebar and downloaded as JSON. The file is validated on load: weights must sum to 100, minimum scores cannot exceed their weight, and the duration range must be consistent.

* **Points per criterion** = rating / 10 × weight. The defaults are hook 35, cliffhanger 35, context 20 and pacing 10.
* **Hard rejection** applies when total duration is outside 20 to 60 s, timestamps are invalid, a referenced event does not exist, segments overlap, or segments are out of order without a justification. Rejected candidates stay in the outputs with their reasons.
* **Human review** is flagged when hook is under 18, cliffhanger under 18, context under 10, total under 60, evidence is unverified, the hook or cliffhanger has no verified evidence, confidence is low, or a cross-episode join looks incoherent.
* **Duration** is the sum of segment lengths, never last end minus first start. It is a constraint, not a score, so a 30 s promo does not outrank an equally good 50 s promo.
* **Ranking order**: candidates passing every threshold come first, then by `ranking_score`. This equals the total minus 2 points per extra segment and 2 points for crossing episodes. A continuous single-episode promo therefore wins when scores are comparable.
* **Near-duplicates** share 50% or more of their footage with a better-ranked promo. They are marked `duplicate` and kept in the outputs.
* **Confidence** is the model's self-reported confidence, reduced for unverified quotes, mid-line cuts, dead air and validation warnings.

Other useful parameters: `candidates_per_episode`, `generation_repair_attempts`, `evaluation_retry_attempts`, `cross_episode_max_windows`, `cross_episode_max_episode_gap`, `cross_episode_require_shared_character` and `min_segment_seconds`.

## Hallucination prevention

| Risk | Deterministic control |
|---|---|
| Invented timestamps | Model outputs event IDs only. Times come from the source events. |
| Invented events or scenes | Unknown IDs reject the clip, or trigger a repair prompt listing the bad IDs. |
| Wrong duration | Python computes the sum of segment lengths and checks the range. Repair prompts list valid end events. |
| Fabricated quotes | Each quote must appear verbatim (case, quote style and whitespace normalised) in an event inside the clip. Failures trigger one re-evaluation, then the promo is flagged and confidence lowered. |
| Arbitrary cuts | Python notes cuts that slice a line of dialogue and silent stretches. The evaluator sees these notes and the QA report records them. |
| Context from other episodes | The evaluator never receives text from outside the clip. |
| Unrelated cross-episode joins | Episodes must be within the configured gap, and each cut must join segments that share a character found in the dataset's speaker labels. The evaluator also judges coherence. |
| Lost model output | Every call is logged with its request, raw response, status and latency. |

## Tests

```bash
pytest
```

The suite has 106 tests covering timestamp extraction and conversion, invalid and ambiguous timestamps, scene, dialogue and action parsing, every input format, ZIP safety, single- and multi-segment durations, scoring arithmetic, weight validation, eligibility, ranking preferences, duplicate detection, cross-episode validation, repeat processing with different datasets, the OpenAI-compatible HTTP client (against a local stub server), and Streamlit UI smoke tests.

Required edge cases and where they are tested:

| Edge case | Test |
|---|---|
| Screenplay without episode metadata | `test_parsers.py::test_screenplay_without_episode_metadata_uses_filename` |
| Missing timestamps | `test_parsers.py::test_missing_timestamps_are_flagged_and_not_invented` |
| Fewer than 5 episodes | `test_pipeline.py::test_heuristic_small_dataset_end_to_end` |
| 100+ episodes | `test_pipeline.py::test_dataset_with_over_100_episodes` (120 episodes) |
| Identical scene names | `test_parsers.py::test_identical_scene_names_get_unique_ids` |
| Overlapping camera and dialogue | `test_parsers.py::test_overlapping_camera_and_dialogue_not_double_counted` |
| Invalid or inconsistent timestamps | `test_parsers.py::test_invalid_and_inconsistent_timestamps_flagged` |
| No suitable promotional moments | `test_pipeline.py::test_no_suitable_promotional_moments` |
| Valid multi-segment promo | `test_pipeline.py::test_valid_multi_segment_promo` |
| Strong hook, no cliffhanger | `test_pipeline.py::test_strong_hook_with_no_cliffhanger` |

Tests use synthetic screenplays with placeholder characters. They do not depend on the provided dataset or on network access.

## Results on the provided dataset

Run `20261108-145302-fba199` processed the provided ZIP (50 TXT screenplays) with Google `gemini-3.1-flash-lite` through the OpenAI-compatible endpoint. Artifacts are in `outputs/20261108-145302-fba199/` and every model response is in `raw_outputs/20261108-145302-fba199/raw_responses.jsonl`.

| Measure | Result |
|---|---|
| Files parsed | 50 of 50, no parsing errors or warnings |
| Episodes and events | 50 episodes, 3,727 events, all with valid timestamps |
| Model calls | 212 (all succeeded; 25 generation repairs, 8 cross-episode repairs, 6 evidence retries) |
| Run time | 12.7 minutes |
| Candidates | 159 (133 single-segment, 11 multi-segment, 15 cross-episode) |
| Ranked (eligible) | 104, of which 86 pass every threshold |
| Rejected | 44, all first proposals with durations from 7 s to 243 s; 33 repaired versions became valid |
| Near-duplicates removed | 11 |
| Flagged for human review | 22 |
| Mean / median score | 81.5 / 83.5 (min 43.5, max 98.0) |

Phase 7 checks:

* **Invalid timestamps**: none. An independent recheck compared every segment's start, end and duration with the source events and found zero mismatches.
* **Unsupported files**: none. The ZIP's folder entry was skipped as expected.
* **Duplicates**: 11 candidates sharing 50% or more of their footage with a better promo were marked duplicate.
* **Hallucinated evidence**: 583 evidence quotes, all verified verbatim against the source after 6 evaluation retries. No candidate referenced a nonexistent event.

Top 10 recommendations:

| Rank | Score | Hook / Cliff / Context / Pacing | Duration | Segment | Title |
|---|---|---|---|---|---|
| 1 | 98.0 | 35 / 35 / 18 / 10 | 41.6 s | EP031 01:58.40-02:40.00 | The Price of Pride |
| 2 | 98.0 | 35 / 35 / 18 / 10 | 39.8 s | EP036 01:39.00-02:18.80 | The Betrayal Revealed |
| 3 | 95.5 | 35 / 32 / 20 / 9 | 23.7 s | EP002 00:00.00-00:23.70 | The Public Humiliation |
| 4 | 94.5 | 32 / 35 / 18 / 10 | 41.4 s | EP003 00:23.10-01:04.50 | Discarded Daughter |
| 5 | 94.5 | 32 / 35 / 18 / 10 | 24.0 s | EP015 00:37.00-01:01.00 | Defiance in the Mud |
| 6 | 94.5 | 32 / 35 / 18 / 10 | 30.5 s | EP023 00:57.50-01:28.00 | Deadly Diagnosis |
| 7 | 94.5 | 32 / 35 / 18 / 10 | 42.7 s | EP027 01:47.80-02:30.50 | The Poisoned Accusation |
| 8 | 94.5 | 32 / 35 / 18 / 10 | 43.5 s | EP028 01:35.00-02:18.50 | Sentenced to the Gallows |
| 9 | 94.5 | 32 / 35 / 18 / 10 | 20.9 s | EP043 00:00.60-00:21.50 | The Duke Returns |
| 10 | 93.5 | 32 / 35 / 18 / 9 | 22.1 s | EP005 00:06.80-00:28.90 | The Disowned Bride |

Full detail, including evidence, reasoning, weaknesses and the rejection log, is in `qa_report.md`, `ranked_promos.csv` and `candidates_all.json`.

What the run shows about the model:

* **Scores cluster high.** Flash-Lite rated most clips generously and reported confidence 0.9 to 1.0 on almost every one, so scores separate the top clips only weakly. A stronger model, or human review of the top 20, will order them better.
* **First proposals often miss the duration range.** Validation and repair handled this, but it costs extra calls.
* **Cross-episode promos were rarely better.** Of 15 proposed, 2 ranked, and the evaluator judged one of them incoherent.

Other backends tried on the same dataset:

* `qwen2.5:7b` on local Ollama produced valid JSON and verified quotes but took about two minutes per call on a 16 GB Mac, and rated a calm establishing shot 9/10 as a hook. It was stopped before finishing.
* `gemini-3.5-flash` hit the free tier's limit of 20 requests per day. The client now stops at the first daily-quota error instead of retrying every call.
* The offline `heuristic` baseline run (`outputs/20261108-121406-b344f9/`) ranks 142 of 150 candidates in about 1 second, with every promo flagged for review.

## Known limitations and failure cases

* **Script text, not video.** Scores estimate editorial quality from the screenplay. Acting, music, visuals and actual timing in the rendered episode are not seen. Scores are not predictions of views or conversions; no performance data was used.
* **Model quality matters.** Small local models follow the JSON contract but judge drama less reliably than large hosted models. The validator guarantees that clips are real and correctly timed. It cannot guarantee that a model's taste is good, which is why human review is built in.
* **Heuristic baseline.** The offline `heuristic` provider ranks by generic dramatic vocabulary, punctuation and timeline structure. It has no semantic understanding, caps its ratings at 8.5/10 and reports confidence 0.35, so all its promos are flagged for review. Use it for demos, smoke tests or when no model is available.
* **Cut points follow event boundaries.** A segment starts at an event's start and ends at an event's end. A camera shot that began slightly earlier is listed as partially included. Frame-accurate trims are left to the editor.
* **Character matching is lexical.** The cross-episode check matches names from speaker labels. Two different characters with the same role word (for example two different "guards") could look linked. The evaluator's coherence judgement and human review are the backstop.
* **Cross-episode search is bounded.** Only the best-scoring adjacent episode pairs are explored (`cross_episode_max_windows`). Wider gaps can be enabled in the config at higher cost.
* **Parsing is heuristic.** Unusual layouts may be read as plain description, or rejected as unrecognised. Scanned PDFs need OCR, which is not included. JSON and CSV with unfamiliar column names need a `field_mapping` entry.
* **Cost and time scale with episodes.** Roughly 1 generation call per episode, 1 evaluation call per valid candidate, plus repairs and cross-episode calls. For the 50-episode dataset with 3 candidates per episode, expect about 220 calls.

Failure cases the QA module detects and reports per candidate:

* strong hook but weak ending
* strong cliffhanger but insufficient context
* good dialogue but poor standalone narrative
* excessively long or short clip
* duplicate promotional moment
* missing or inconsistent timestamps
* cross-episode segments that do not form a coherent story
* unverified evidence and hallucinated source events

## Free deployment

### Streamlit Community Cloud

1. Push this folder to a GitHub repository. The `.gitignore` already excludes `.env`, outputs and feedback.
2. Go to https://share.streamlit.io, choose **New app**, pick the repository and set the main file to `app.py`. If the project is in a subfolder, give the path, for example `promo_discovery_engine/app.py`.
3. Under **Advanced settings**, choose Python 3.11 or newer and paste secrets in TOML format. Root-level secrets are exposed as environment variables, which the app reads:

   ```toml
   LLM_PROVIDER = "openai_compatible"
   LLM_API_BASE = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   LLM_API_KEY = "gsk_..."
   LLM_CONCURRENCY = "4"
   ```

4. Deploy. The free tier has limited memory and an ephemeral disk. Download the run ZIP from the Export tab, because `outputs/` and the feedback file are lost when the app restarts. Ollama is not available there, so use a hosted API or the heuristic provider.

### Hugging Face Spaces

Create a Space with the Docker or Streamlit SDK, upload the project, and add the `LLM_*` variables as Space secrets. Set the start command to `streamlit run app.py --server.port 7860 --server.address 0.0.0.0`.

## Project structure

```
promo_discovery_engine/
├── app.py                      Streamlit dashboard
├── cli.py                      Headless runner
├── config/
│   ├── settings.py             Scoring, app and LLM settings with validation
│   ├── scoring_config.json     Weights, thresholds, selection parameters
│   └── app_config.json         Upload limits, paths, JSON/CSV field mapping
├── core/
│   ├── ingestion.py            Uploads, safe ZIP handling, limits, duplicates
│   ├── parsers.py              Parser interface and registry; TXT/MD/PDF/DOCX/SRT/JSON/CSV
│   ├── normalizer.py           Common schema, IDs, timestamp checks, parse report
│   ├── timestamp_utils.py      Timestamp parsing/formatting, interval maths
│   ├── candidate_generator.py  Step 1 with validation-driven repair
│   ├── clip_builder.py         Exact clip content, hook and cliffhanger windows, notes
│   ├── promo_evaluator.py      Step 2 with evidence verification and retry
│   ├── scoring_engine.py       Weighted points, eligibility, review flags, confidence
│   ├── cross_episode_analyser.py  Step 3 and coherence checks
│   ├── validator.py            Segment resolution and hallucination checks
│   ├── deduplicator.py         Near-duplicate detection by shared footage
│   ├── ranking_engine.py       Final ordering and top-K
│   ├── quality_assurance.py    Failure cases, QA report, reviewer feedback store
│   └── pipeline.py             Orchestration, re-scoring, loading past runs
├── llm/
│   ├── provider.py             Provider abstraction: OpenAI-compatible, Ollama, mock
│   ├── heuristic_provider.py   Offline deterministic baseline
│   ├── prompts.py              Versioned prompt templates
│   └── structured_output.py    JSON extraction, schema validation, repair, raw log
├── models/schemas.py           Pydantic models
├── utils/                      Logging and exports
├── tests/                      Pytest suite
├── outputs/  raw_outputs/  feedback/
├── .env.example  requirements.txt  pytest.ini
```

To add a new input format, subclass `BaseParser` in `core/parsers.py`, return `RawEpisode` objects, and call `register_parser`. Normalisation, validation and everything downstream work unchanged.
