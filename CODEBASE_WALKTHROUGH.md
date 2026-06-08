# Codebase Walkthrough

A file-by-file reference for the clinical trial patient-matching prototype. For *why* the design works, see [PIPELINE_DEEP_DIVE.md](PIPELINE_DEEP_DIVE.md).

---

## Entry points

### `run_matching.py`

The CLI. Parses args, sets up the pipeline, runs patients, writes JSON to `output/`, and prints the console summary.

Key details:
- `--patient-indication` accepts either a folder name (`breast-cancer`) or indication string (`breast cancer`) — resolved via `INDICATION_TO_FOLDER` in `indication_classifier.py`
- Sex and age are **auto-parsed** from the `Sex / Age` table row in `patient_summary.md`; `--sex` / `--age` override
- `_print_result()` renders the ranked trial list with `✓/✗/?` icons, `(conf X.XX)` or `(conf N/A)`, and verbose criterion breakdown for `likely_eligible` trials or when `-v` is passed

### `pipeline/build_index.py`

One-time script that loads `trials_clean.csv` and calls `MedCPTRetriever.build()`. Run before the first matching session. Accepts `--max-trials N` for smoke-test subsets. Silently reuses an existing LanceDB table — delete `cache/lancedb/` and `cache/medcpt_nctids.json` to force a rebuild.

---

## `config.py`

Central configuration. All values have sensible defaults; most are overridable via environment variable.

| Constant | Default | Role |
|----------|---------|------|
| `CLAUDE_MODEL_FAST` | `claude-haiku-4-5-20251001` | Steps 1, 2, 4 |
| `CLAUDE_MODEL_SMART` | `claude-sonnet-4-6` | Step 5 aggregation |
| `TOP_K_RETRIEVAL` | `10` | Candidates retrieved and assessed per patient |
| `MAX_CONDITIONS_PER_PATIENT` | `12` | Condition queries sent to MedCPT |
| `MAX_CRITERIA_PER_TRIAL` | `50` | Criteria cap per inclusion/exclusion group |
| `MEDCPT_TOP_N` | `2000` | ANN results per condition before RRF merge |
| `RRF_K` | `20` | RRF smoothing constant |
| `LIKELY_ELIGIBLE_THRESHOLD` | `1.5` | total_score ≥ this → `likely_eligible` |
| `NOT_ELIGIBLE_THRESHOLD` | `-0.5` | total_score ≤ this → `not_eligible` |
| `CONFIDENCE_REVIEW_THRESHOLD` | `0.6` | confidence < this → demote `likely_eligible` |

---

## `pipeline/full_pipeline.py`

### `MatchingPipeline`

Orchestrates all 5 steps. Call `setup()` once, then `match_patient()` per patient.

**`setup(max_trials=None)`**
Loads `trials_clean.csv` (via `load_trials`) and `eligibility_criteria_chunks.csv` (via `load_criteria_chunks`). Attempts `MedCPTRetriever.load()`; falls back to `build()` if the LanceDB index doesn't exist.

**`match_patient(patient_id, patient_text, top_k, skip_matching, sex_filter, age)`**

Runs the full pipeline:

1. `IndicationClassifier.classify()` → indication string (or `None` on failure)
2. `KeywordGenerator.generate()` → conditions list
3. `MedCPTRetriever.search()` → per-condition NCT ID lists; RRF fused inline
4. For each candidate: `CriterionMatcher.match()` → criterion predictions with confidence
5. `compute_matching_score()`, `compute_confidence()`, `aggregate_with_llm()`, `determine_eligibility()` → decision + score
6. Sort by `(DECISION_PRIORITY, -norm_score, -confidence, nct_id)`
7. `extract_criteria_breakdown()` → met/unmet/excluded/missing lists

If `skip_matching=True`, returns after Step 3 with retrieval scores only (`confidence: null`).

**`_DECISION_PRIORITY`** (module-level, imported from `score_aggregator`)
Maps decision labels to sort integers: `likely_eligible=0`, `needs_review=1`, `not_eligible=2`.

---

## `retrieval/indication_classifier.py`

### `IndicationClassifier`

Claude (Haiku) maps a patient note to one of 8 supported therapeutic area strings. The returned string is used verbatim as a `source_condition_query` LanceDB filter.

**`classify(patient_text) → str`**
Raises `ValueError` if Claude returns an unrecognized indication. The caller (`full_pipeline.py`) catches this and falls back to no filter.

**`SUPPORTED_INDICATIONS`** — the 8 valid indication strings (module-level list).

**`INDICATION_TO_FOLDER`** — maps indication strings to patient-dataset folder names; used by `get_patient_by_id()` in `data_loader.py` so `--patient-indication` accepts both naming conventions.

---

## `retrieval/keyword_generator.py`

### `KeywordGenerator`

Claude (Haiku) extracts ranked medical conditions from a patient note. Results are cached by `patient_id` in `cache/keywords_cache.json`; re-runs skip the API call unless `_CACHE_SCHEMA_VERSION` has changed.

**`generate(patient_id, patient_text) → dict`**
Returns `{"summary": str, "conditions": [str, ...]}`. Claude returns structured JSON with `term` + `status` per condition; the generator keeps only `active` and `historical` statuses and drops `negated`, `hypothetical`, and `family_history`. This prevents negated findings (e.g., "no prior chemo") from contaminating retrieval queries.

---

## `retrieval/medcpt_retriever.py`

### `MedCPTRetriever`

Manages the LanceDB vector index and runs ANN search.

**`build(trials_df, force=False)`**
Encodes trials with `ncbi/MedCPT-Article-Encoder` using `(title, brief_summary + inclusion_criteria)` pairs — exclusion criteria are intentionally excluded (see PIPELINE_DEEP_DIVE.md). Stores 768-dim float32 CLS embeddings in LanceDB with metadata: `nct_id`, `title`, `conditions`, `source_condition_query`, `sex`, `min_age`, `max_age`, `study_type`, `phase`. Ages are normalized to integer years (`-1` / `200` sentinels for unknown bounds). Silently reuses existing table unless `force=True`.

**`load()`**
Opens an existing LanceDB table. Raises `RuntimeError` if not found.

**`search(conditions, k, sex_filter, min_age, max_age, indication) → list[list[str]]`**
Encodes each condition with `ncbi/MedCPT-Query-Encoder`. Builds a `WHERE` clause from the provided filters and runs one ANN search per condition. Returns a list of NCT ID lists (one per condition), ordered by cosine similarity.

---

## `matching/criterion_matcher.py`

### `CriterionMatcher`

Calls Claude (Haiku) once per inclusion group and once per exclusion group for each candidate trial.

**`match(patient_text, trial_info, criteria_chunks_df, max_criteria) → dict`**
Returns:
```python
{
  "inclusion": {"0": [reasoning, [sent_ids], label, confidence], ...},
  "exclusion": {"0": [reasoning, [sent_ids], label, confidence], ...}
}
```
On Claude API failure, records `{"error": str(e)}` for that group and continues.

Criteria are loaded from `eligibility_criteria_chunks.csv` (ordered chunks); falls back to splitting the raw `inclusion_criteria` / `exclusion_criteria` text from `trials_clean.csv` if no chunks exist for a trial. Capped at `MAX_CRITERIA_PER_TRIAL` per group for cost control.

**`_format_patient_with_sentence_ids(patient_text)`**
Numbers each sentence so Claude can cite `[sent_id]` as evidence, enabling traceability from criterion decision back to specific patient note text.

**`_build_system_prompt(inc_exc)`**
Constructs the system prompt from `_SYSTEM_TEMPLATE`, injecting the appropriate criterion definition and label set. The prompt instructs Claude to report confidence ≥ 0.8 only for direct explicit evidence; 0.5 for inference or uncertainty.

---

## `aggregation/score_aggregator.py`

### Module-level constants

**`DECISION_PRIORITY`** — `{"likely_eligible": 0, "needs_review": 1, "not_eligible": 2}`. Exported; imported by `full_pipeline.py` for sort key construction.

**`_DEMOTION_SCORE_OFFSET = 0.001`** — gap subtracted from the `likely_eligible` boundary when capping a demoted trial's `norm_score`, ensuring it always ranks below genuine eligible trials.

### Shared iterator

**`_iter_criteria(matching_results) → Generator`**
Yields `(group, idx, info)` for every structurally valid criterion entry (must be a `list` with ≥ 3 elements). Used by all four traversal functions below, centralizing the validity guard.

### Scoring functions

**`compute_matching_score(matching_results) → float`**
Rule-based Stage 1 score ∈ [−2, 1]:
```
score = included / (included + not_included + no_info + ε)
score -= 1.0  if any not_included
score -= 1.0  if any excluded
```

**`compute_confidence(matching_results) → Optional[float]`**
Mean LLM-reported confidence across applicable criteria. Skips:
- `not applicable` labels (irrelevance certainty ≠ eligibility certainty)
- Criteria with no 4th element (old 3-element schema)
- Non-numeric or out-of-range confidence values

Returns `None` when no usable confidence values exist (zero criteria, all skipped). `None` signals `determine_eligibility` to skip the demotion gate entirely.

**`aggregate_with_llm(patient_text, matching_results, trial_info) → dict`**
Calls Claude (Sonnet) with the patient note, trial metadata, and formatted criterion predictions. Returns:
```python
{"relevance_score_R": float, "relevance_explanation": str,
 "eligibility_score_E": float, "eligibility_explanation": str}
```
On failure, returns zeroed scores with the error in `relevance_explanation`.

Uses the lazy `_get_client()` accessor so the Anthropic SDK is not initialized at import time — pure-logic functions (`compute_matching_score`, `compute_confidence`) remain importable and testable without an API key.

**`determine_eligibility(matching_score, agg_results, confidence) → tuple[str, float]`**
Computes `total = matching_score + (R + E) / 100`, applies decision thresholds, applies the confidence demotion gate, normalizes to [0, 1], and caps demoted scores. Returns `(decision, norm_score)`.

**`extract_criteria_breakdown(matching_results, criteria_chunks_df, nct_id) → dict`**
Builds `{met_inclusion_criteria, unmet_inclusion_criteria, triggered_exclusion_criteria, missing_information}` lists with criterion text (from chunks DataFrame) and reasoning. Falls back to `"Criterion {idx}"` if a chunk is missing.

---

## `utils/data_loader.py`

**`load_trials(nrows=None) → DataFrame`**
Loads `trials_clean.csv`, normalizes missing values, and returns the DataFrame. `nrows` limits rows for smoke tests.

**`load_criteria_chunks() → DataFrame`**
Loads `eligibility_criteria_chunks.csv`.

**`load_patients() → list[dict]`**
Iterates `patient-datasets/` subdirectories, reads each `patient_summary.md`, auto-parses `sex` and `age` from the `Sex / Age` table row using `_SEX_AGE_RE`.

**`get_patient_by_id(patient_id) → Optional[dict]`**
Accepts folder names or indication strings; resolves via `INDICATION_TO_FOLDER`.

**`get_trial_info(nct_id, trials_df) → Optional[dict]`**
Returns a flat dict of key trial fields (title, summary, criteria text, conditions/drugs as both pipe-separated strings and parsed lists, etc.) for a given NCT ID.

**`parse_age_to_years(age_str) → Optional[int]`**
Converts strings like `"20 Years"`, `"6 Months"`, `"2 Weeks"` to integer years. Used at index-build time to normalize `min_age` / `max_age` for LanceDB filtering.

---

## `utils/json_utils.py`

**`strip_json_fences(raw) → str`**
Removes ` ```json ` / ` ``` ` markdown fences from LLM response text. Used by all four Claude-calling modules (`indication_classifier`, `keyword_generator`, `criterion_matcher`, `score_aggregator`).
