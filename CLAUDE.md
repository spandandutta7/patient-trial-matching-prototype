# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A patient–clinical-trial matching prototype. Given a free-text patient note, it classifies the patient's therapeutic area, retrieves semantically similar trials (MedCPT over LanceDB with indication pre-filtering), and assesses each trial's eligibility criteria with Claude, producing a ranked eligibility decision per trial. The semantic retrieval layer uses the same MedCPT encoders as the **TrialGPT** reference implementation but replaces FAISS with LanceDB (for metadata pre-filtering) and Azure OpenAI with Claude. BM25 and hybrid fusion have been removed — a LanceDB `source_condition_query` pre-filter handles coarse recall instead.

`README.md` is the authoritative spec for output schema, scoring math, and CLI flags. Two reference docs go deeper: `CODEBASE_WALKTHROUGH.md` (file-by-file — what each file/function is) and `PIPELINE_DEEP_DIVE.md` (why the 5-step design works). This file captures only what isn't obvious from reading individual files.

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

# Run matching
python run_matching.py --patient-indication breast-cancer       # full pipeline, one patient
python run_matching.py --all -v                                 # all patients, verbose
python run_matching.py --patient-indication breast-cancer --skip-matching  # retrieval only (no LLM)
python run_matching.py --patient-indication breast-cancer --sex FEMALE --age 70  # override auto-filter
python run_matching.py --patient-indication breast-cancer -v    # full per-criterion breakdown
```

There is no test suite, linter, or build step. Validation is done by running the CLI and inspecting `output/<patient_id>.json`.

## Architecture

Five sequential steps, orchestrated by `MatchingPipeline` in [pipeline/full_pipeline.py](pipeline/full_pipeline.py). Call `setup()` once (loads datasets + indices), then `match_patient()` per patient.

1. **Indication classification** — [retrieval/indication_classifier.py](retrieval/indication_classifier.py): Claude (FAST model) maps the patient note to one of 8 exact `source_condition_query` strings (e.g. `"breast cancer"`). Used as a LanceDB pre-filter in Step 3. Falls back to no filter on failure.
2. **Keyword extraction** — [retrieval/keyword_generator.py](retrieval/keyword_generator.py): Claude (FAST model) turns the patient note into `{summary, conditions[]}`, where `conditions` is a ranked list of up to 12 medical terms used as search queries.
3. **Semantic retrieval** — [retrieval/medcpt_retriever.py](retrieval/medcpt_retriever.py): `MedCPTRetriever` encodes each condition with MedCPT-Query-Encoder and runs one LanceDB ANN search per condition, pre-filtered by indication (and optionally sex/age). Per-condition result lists are merged inline with RRF, down-weighting later conditions by `1/(condition_idx+1)`, yielding top-K candidate NCT IDs. Trials are encoded at index-build time using `(title, brief_summary + inclusion_criteria)` — exclusion criteria are intentionally excluded to avoid negation contamination.
4. **Criterion matching** — [matching/criterion_matcher.py](matching/criterion_matcher.py): for each candidate, **one Claude call for inclusion + one for exclusion**. Returns per-criterion `{idx: [reasoning, [sentence_ids], label]}`. The patient note is fed in with numbered sentence IDs so the model can cite evidence.
5. **Aggregation** — [aggregation/score_aggregator.py](aggregation/score_aggregator.py): a rule-based `matching_score` from the labels, plus an LLM (SMART model) `relevance_R`/`eligibility_E` pass, combined into a final decision (`likely_eligible` / `not_eligible` / `needs_review`) and a normalized `[0,1]` score. Scoring formulas are documented in the module docstring and README.

Two Claude models, configured in [config.py](config.py): `CLAUDE_MODEL_FAST` (haiku) for high-volume steps 1, 2 & 4; `CLAUDE_MODEL_SMART` (sonnet) for step 5. A full top-10 run is ~32 API calls (1 indication + 1 keywords + 20 criterion + 10 aggregation).

### Key conventions and gotchas

- **LanceDB index is cached and silently reused.** `MedCPTRetriever.build()/load()` reuses the LanceDB table if it exists — **ignoring the passed DataFrame**. Consequence: if you build with `--max-trials 1000` and later run without it, you still get the 1000-trial index. To rebuild on different data delete `cache/lancedb/` and `cache/medcpt_nctids.json` first.
- **Keyword results are cached by `patient_id`** in `cache/keywords_cache.json`. Re-running a patient won't re-call the LLM for step 2 unless you clear that cache.
- **Two sources of criteria.** Retrieval/metadata use `dataset/trials_clean.csv` (loaded via `get_trial_info`). Step 4 uses the finer-grained `dataset/eligibility_criteria_chunks.csv` (`criterion_type` ∈ {inclusion, exclusion}, ordered by `criterion_index`). If no chunks exist for a trial, the matcher falls back to splitting the raw `inclusion_criteria`/`exclusion_criteria` text from `trials_clean.csv`.
- **Three-layer LanceDB pre-filtering.** The primary filter is the LLM-classified `indication` (`source_condition_query` = e.g. `"breast cancer"`), which reduces the search space by 73–98% depending on condition. Secondary filters are `sex` and `age`. Ages are normalized to integer years at build time (`min_age=-1`/`max_age=200` sentinels when unknown). Sex/age are **auto-parsed** from the `Sex / Age` line in each `patient_summary.md`; the `--sex`/`--age` CLI flags override the auto-parsed values. Indication classification failure degrades gracefully to no filter.
- **Exclusion criteria are excluded from trial embeddings.** Trials are indexed using `(title, brief_summary + inclusion_criteria)` only. Including exclusion criteria caused negation contamination — MedCPT embeddings are nearly blind to negation, so "No prior anthracyclines" would spuriously match a patient query for "anthracyclines."
- **All Claude calls use `temperature=0`** and parse JSON from the response, stripping ` ```json ` fences. Parse failures degrade gracefully (error recorded, pipeline continues) rather than crashing.
- `MAX_CRITERIA_PER_TRIAL` (default 50) caps criteria per inclusion/exclusion group purely for cost control.
- `config.py` values are mostly overridable via environment variables (see the `os.getenv` calls).
