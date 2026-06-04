# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A patient–clinical-trial matching prototype. Given a free-text patient note, it retrieves candidate trials (hybrid BM25 + MedCPT semantic search over LanceDB) and assesses each trial's eligibility criteria with Claude, producing a ranked eligibility decision per trial. The retrieval layer deliberately mirrors the bundled **TrialGPT** reference implementation (`TrialGPT/trialgpt_retrieval/`) — same MedCPT encoders, BM25 field weighting, and RRF formula — but swaps FAISS for LanceDB (to gain metadata pre-filtering) and Azure OpenAI for Claude.

`README.md` is the authoritative spec for output schema, scoring math, and CLI flags. Two reference docs go deeper: `CODEBASE_WALKTHROUGH.md` (file-by-file — what each file/function is) and `PIPELINE_DEEP_DIVE.md` (why the 4-step design works). This file captures only what isn't obvious from reading individual files.

## Commands

```bash
# Env + deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"   # one-time tokenizer
cp .env.example .env                                   # then set ANTHROPIC_API_KEY

# Build retrieval index — REQUIRED before any matching (encodes 60K trials → LanceDB)
python pipeline/build_index.py                  # full corpus (slow on CPU: 2-6h)
python pipeline/build_index.py --max-trials 1000   # fast smoke-test subset
python pipeline/build_index.py --bm25-only      # skip the slow MedCPT encoding

# Run matching
python run_matching.py --patient-id sigir-20141            # full pipeline, one patient
python run_matching.py --all --top-k 10                    # all patients
python run_matching.py --patient-id sigir-20141 --skip-matching   # retrieval only (cheap, no LLM)
python run_matching.py --patient-id sigir-20141 --sex FEMALE --age 58  # LanceDB metadata pre-filter
python run_matching.py --patient-id sigir-20141 -v         # full per-criterion breakdown
```

There is no test suite, linter, or build step. This is **not a git repository**. Validation is done by running the CLI and inspecting `output/<patient_id>.json`.

## Architecture

Four sequential steps, orchestrated by `MatchingPipeline` in [pipeline/full_pipeline.py](pipeline/full_pipeline.py). Call `setup()` once (loads datasets + indices), then `match_patient()` per patient.

1. **Keyword extraction** — [retrieval/keyword_generator.py](retrieval/keyword_generator.py): Claude (FAST model) turns the patient note into `{summary, conditions[]}`, where `conditions` is a ranked list of medical terms used as search queries.
2. **Hybrid retrieval** — [retrieval/](retrieval/): `BM25Retriever` and `MedCPTRetriever` each return **one ranked NCT-ID list per condition** (a list-of-lists aligned with `conditions`). `hybrid_fusion.fuse()` combines them via RRF, down-weighting later conditions by `1/(condition_idx+1)`, yielding top-K candidate trials.
3. **Criterion matching** — [matching/criterion_matcher.py](matching/criterion_matcher.py): for each candidate, **one Claude call for inclusion + one for exclusion**. Returns per-criterion `{idx: [reasoning, [sentence_ids], label]}`. The patient note is fed in with numbered sentence IDs so the model can cite evidence.
4. **Aggregation** — [aggregation/score_aggregator.py](aggregation/score_aggregator.py): a rule-based `matching_score` from the labels, plus an LLM (SMART model) `relevance_R`/`eligibility_E` pass, combined into a final decision (`likely_eligible` / `not_eligible` / `needs_review`) and a normalized `[0,1]` score. Scoring formulas are documented in the module docstring and README.

Two Claude models, configured in [config.py](config.py): `CLAUDE_MODEL_FAST` (haiku) for high-volume steps 1 & 3; `CLAUDE_MODEL_SMART` (sonnet) for step 4. A full top-20 run is ~42 API calls.

### Key conventions and gotchas

- **Indices are cached and silently reused.** `BM25Retriever.build()` reuses `cache/bm25_corpus.json` if present; `MedCPTRetriever.build()/load()` reuses the LanceDB table if it exists — **ignoring the passed DataFrame**. Consequence: if you build the index with `--max-trials 1000` and later run without it, you still get the 1000-trial index. To rebuild on different data you must delete the relevant cache files (`cache/bm25_corpus.json`, `cache/medcpt_nctids.json`, `cache/lancedb/`).
- **Keyword results are cached by `patient_id`** in `cache/keywords_cache.json`. Re-running a patient won't re-call the LLM for step 1 unless you clear that cache.
- **Two sources of criteria.** Retrieval/metadata use `dataset/trials_clean.csv` (loaded via `get_trial_info`). Step 3 uses the finer-grained `dataset/eligibility_criteria_chunks.csv` (`criterion_type` ∈ {inclusion, exclusion}, ordered by `criterion_index`). If no chunks exist for a trial, the matcher falls back to splitting the raw `inclusion_criteria`/`exclusion_criteria` text from `trials_clean.csv`.
- **LanceDB pre-filtering** (`--sex`, `--age`) is the main design departure from TrialGPT's FAISS. Ages are normalized to integer years at build time (`min_age=-1`/`max_age=200` sentinels when unknown); the `where` clause matches the patient's sex OR `'ALL'`. Age/sex are **not** auto-parsed from the note — they must be passed on the CLI.
- **All Claude calls use `temperature=0`** and parse JSON from the response, stripping ``` ```json ``` fences. Parse failures degrade gracefully (error recorded, pipeline continues) rather than crashing.
- `MAX_CRITERIA_PER_TRIAL` (default 50) caps criteria per inclusion/exclusion group purely for cost control.
- `config.py` values are mostly overridable via environment variables (see the `os.getenv` calls).
