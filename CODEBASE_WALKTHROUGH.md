# Codebase Walkthrough

A complete, file-by-file reference for the clinical-trial patient-matching prototype: **what every file contains**, what each function/class does, and how each piece feeds the overall pipeline and the final `output/<patient_id>.json`. This is the "where does X live / what is this file" map.

**Related docs:**
- [README.md](README.md) — architecture, output format, setup, and spec. Start there.
- [PIPELINE_DEEP_DIVE.md](PIPELINE_DEEP_DIVE.md) — *why the design works*: the reasoning and trade-offs behind each of the 4 steps. This walkthrough tells you *what* each file is; the deep dive tells you *why* it's built that way.

---

## The pipeline at a glance

```
patient note (dataset/patient-profiles.jsonl)
        │
        ▼
STEP 1  retrieval/keyword_generator.py   → {summary, conditions[]}      (Claude FAST)
        │
        ▼
STEP 2  retrieval/bm25_retriever.py      ┐
        retrieval/medcpt_retriever.py    ├─ per-condition ranked NCT lists
        retrieval/hybrid_fusion.py       ┘  → fused top-K candidate trials
        │
        ▼
STEP 3  matching/criterion_matcher.py    → per-criterion labels           (Claude FAST)
        │
        ▼
STEP 4  aggregation/score_aggregator.py  → matching_score + R/E + decision (Claude SMART)
        │
        ▼
        run_matching.py writes output/<patient_id>.json + prints summary
```

Everything is wired together by `MatchingPipeline` in `pipeline/full_pipeline.py`, driven by the `run_matching.py` CLI. `config.py` holds all tunables; `utils/data_loader.py` is the shared data-access layer.

---

## Root files

### `config.py` — central configuration
Single source of truth for paths, model names, and thresholds. Everything else imports `config` rather than hard-coding values.

- **Path constants**: `BASE_DIR`, `DATASET_DIR`, `CACHE_DIR`, `OUTPUT_DIR`, and the specific files (`TRIALS_CSV`, `CRITERIA_CHUNKS_CSV`, `PATIENT_PROFILES_JSONL`) and caches (`LANCEDB_DIR`, `BM25_CACHE_PATH`, `KEYWORDS_CACHE_PATH`).
- **MedCPT settings**: encoder model names (`MEDCPT_ARTICLE_ENCODER`, `MEDCPT_QUERY_ENCODER`), `EMBED_DIM=768`, token limits, and `MEDCPT_BATCH_SIZE` (env-overridable; lower it on OOM).
- **Claude models**: `CLAUDE_MODEL_FAST` (haiku — steps 1 & 3) and `CLAUDE_MODEL_SMART` (sonnet — step 4). Both env-overridable.
- **Retrieval knobs**: `TOP_K_RETRIEVAL=20` (final candidates), `BM25_TOP_N`/`MEDCPT_TOP_N=2000` (per-retriever depth before fusion), `RRF_K=20` (RRF smoothing), `BM25_WEIGHT`/`MEDCPT_WEIGHT=1` (set a weight to 0 to disable that retriever in fusion).
- **Matching/scoring**: `MAX_CRITERIA_PER_TRIAL=50` (cost cap per inc/exc group), `LIKELY_ELIGIBLE_THRESHOLD=1.5`, `NOT_ELIGIBLE_THRESHOLD=-0.5`.
- Loads `.env` via `python-dotenv` if available (optional import).

**Relation to output:** thresholds here decide each trial's `eligibility_decision`; `TOP_K_RETRIEVAL` decides how many trials appear in `results`.

### `run_matching.py` — CLI entry point
The user-facing driver. Parses args, runs the pipeline, writes JSON, prints a console summary.

- **`main()`**: builds the `argparse` parser. A mutually-exclusive required group forces either `--patient-id` or `--all`. Other flags: `--top-k`, `--skip-matching`, `--sex`, `--age`, `--max-trials`, `--output-dir`, `--verbose`. It then: creates the output dir → instantiates `MatchingPipeline` and calls `setup(max_trials=...)` once → resolves which patients to run (`load_patients()` or `get_patient_by_id()`) → loops calling `pipeline.match_patient(...)` → dumps each result to `output/<pid>.json` → calls `_print_result`.
- **`_print_result(result, verbose)`**: formats the console view — header (patient id, summary, top keywords, candidate count), then per-trial line with a ✓/✗/? icon by decision, score, NCT id, title, conditions. For `likely_eligible` trials (or any trial in `--verbose`) it also prints met/unmet inclusion, triggered exclusions, and missing info; `--verbose` adds the eligibility explanation.

**Relation to output:** this file *is* what produces `output/*.json` and the terminal report.

### `requirements.txt`
Runtime deps: `anthropic` (Claude API), `lancedb` + `pyarrow` (vector store), `rank-bm25` (BM25), `transformers` + `torch` (MedCPT encoders), `nltk` (tokenization), `pandas`/`numpy` (data), `tqdm` (progress bars), `python-dotenv` (.env).

### `.env` / `.env.example`
`.env.example` is the template; copy to `.env` and set `ANTHROPIC_API_KEY` (required). Optionally override `CLAUDE_MODEL_FAST`, `CLAUDE_MODEL_SMART`, `TOP_K_RETRIEVAL`, `MEDCPT_BATCH_SIZE`. `config.py` reads these via `os.getenv`.

### `instruction.txt`
The original project brief (objective, dataset description, the 4-step approach, and the instruction to mirror TrialGPT's retrieval). Reference/historical — not imported by any code.

### Documentation files
Human docs (none imported by code):
- **`README.md`** — architecture, output schema, scoring math, CLI, setup, limitations.
- **`CODEBASE_WALKTHROUGH.md`** (this file) — file-by-file reference of what each file/function is.
- **`PIPELINE_DEEP_DIVE.md`** — why the 4-step design works; reasoning and trade-offs.
- **`CLAUDE.md`** — guidance for future Claude Code sessions (commands + gotchas).

---

## `utils/` — shared data access

### `utils/data_loader.py`
The single layer every module uses to read the datasets. Pure functions, no LLM calls.

- **`load_trials(nrows=None)`**: reads `trials_clean.csv` into a DataFrame. Normalizes `conditions` (falls back to `source_condition_query`, then empty) and `fillna`s the text/metadata columns used downstream (`title`, `combined_text_for_retrieval`, `inclusion_criteria`, `exclusion_criteria`, `brief_summary`, `interventions`, `minimum_age`, `maximum_age`, `sex`→`"ALL"`, `phase`, `study_type`). `nrows` powers `--max-trials`.
- **`load_criteria_chunks(nct_ids=None)`**: reads `eligibility_criteria_chunks.csv`, optionally filtered to a set of NCT ids (used to load only the candidate trials' criteria in step 3). `fillna`s `criterion_text`.
- **`load_patients()`**: parses `patient-profiles.jsonl` line-by-line into a list of dicts (each has `_id`, `text`).
- **`get_patient_by_id(patient_id)`**: linear scan for one patient (used by `--patient-id`).
- **`get_trial_info(nct_id, trials_df)`**: extracts one trial's row into a normalized dict (`brief_title`, `official_title`, `brief_summary`, `conditions` + parsed `diseases_list`, `interventions` + parsed `drugs_list`, status, `study_type`, `phase`, `sex`, ages, raw inclusion/exclusion text, `clinicaltrials_url`). Pipe-separated `conditions`/`interventions` are split into lists. This dict is the `trial_info` consumed by the matcher and aggregator and copied into the output.
- **`parse_age_to_years(age_str)`**: converts ClinicalTrials.gov age strings (`"20 Years"`, `"6 Months"`, `"2 Weeks"`, `"30 Days"`) into integer years. Used at MedCPT index-build time to populate `min_age`/`max_age` columns for metadata filtering.

**Relation to output:** `get_trial_info` supplies the per-trial metadata fields (`title`, `conditions`, `study_type`, `phase`, `sex`, `min_age`, `max_age`, `clinicaltrials_url`) you see in each result.

---

## `retrieval/` — Steps 1 & 2 (find candidate trials)

### `retrieval/keyword_generator.py` — Step 1
Turns a free-text patient note into structured search queries via Claude.

- **`KeywordGenerator`**: holds an `anthropic.Anthropic()` client and a JSON cache keyed by `patient_id` (`cache/keywords_cache.json`).
  - **`generate(patient_id, patient_text)`**: returns `{"summary": str, "conditions": [str, ...]}`. Cache-first (no repeat LLM call for a seen patient). Calls the FAST model at `temperature=0` with `_SYSTEM_PROMPT` asking for a 1–2 sentence summary plus up to 32 ranked, searchable condition terms. Strips ``` ```json ``` fences, `json.loads` with a graceful fallback (uses note prefixes if parsing fails), guarantees both keys exist, caches, returns.
  - `_load_cache` / `_save_cache`: JSON cache I/O.

**Relation to output:** `summary` and `conditions` become the `summary` and `keywords` fields of the result. `conditions` is also the query list driving all of Step 2 — the ranking order matters because fusion down-weights later conditions.

### `retrieval/bm25_retriever.py` — Step 2 (lexical)
Keyword/probabilistic retriever using `BM25Okapi`, mirroring TrialGPT's field weighting.

- Module top: ensures NLTK `punkt`/`punkt_tab` tokenizers are present (auto-downloads).
- **`_build_tokenized_corpus(trials_df)`**: tokenizes each trial with TrialGPT weighting — **title ×3, each condition ×2, combined text ×1** (repetition = weight in BM25). Returns parallel lists of token-lists and NCT ids.
- **`BM25Retriever`**:
  - **`build(trials_df)`**: if `cache/bm25_corpus.json` exists, loads the tokenized corpus + NCT ids from it; otherwise tokenizes the corpus and writes the cache. Then constructs `BM25Okapi`. **Note:** an existing cache is reused regardless of the DataFrame passed.
  - **`search(conditions, n)`**: for each condition string, tokenizes and returns the top-`n` NCT ids by BM25 score. Output is a **list aligned with `conditions`** (list-of-lists).

### `retrieval/medcpt_retriever.py` — Step 2 (semantic)
Dense semantic retriever using NCBI MedCPT encoders, stored in LanceDB. The key departure from TrialGPT (which uses FAISS): LanceDB adds metadata pre-filtering.

- **`_get_device()`**: picks `cuda` → `mps` → `cpu`.
- **`_encode_articles(pairs, tokenizer, model, device, batch_size)`**: batch-encodes `(title, text)` pairs with the Article encoder; the embedding is the `[CLS]` token (`last_hidden_state[:,0,:]`), 768-dim float32. `tqdm` progress.
- **`MedCPTRetriever`**:
  - **`build(trials_df)`**: if the LanceDB `trials` table already exists, opens it and loads cached NCT ids (`cache/medcpt_nctids.json`) — **does not re-encode**. Otherwise: loads the Article encoder, encodes all `(title, combined_text_for_retrieval)` pairs, frees the model, and writes a LanceDB table with an explicit PyArrow schema: `nct_id, title, conditions, sex, min_age, max_age, study_type, phase, vector(768)`. `min_age`/`max_age` come from `parse_age_to_years` (sentinels `-1`/`200` when unknown) — these enable the age `where` clause. Persists `medcpt_nctids.json`.
  - **`load()`**: opens an existing table without encoding; raises `RuntimeError` (telling you to run `build_index.py`) if the table is missing.
  - **`_load_query_encoder()`**: lazily loads the Query encoder (separate from the Article encoder).
  - **`search(conditions, k, sex_filter, min_age, max_age)`**: encodes the condition strings with the Query encoder, builds an optional SQL `where` clause (`sex = 'X' OR sex = 'ALL'`; `max_age >= min_age`; `min_age <= max_age`), runs one ANN search per condition with `prefilter=True` (falls back if the LanceDB version doesn't accept the kwarg). Returns a **list aligned with `conditions`** of ranked NCT ids.

**Relation to output:** indirectly — retrieval decides *which* trials get assessed. The `--sex`/`--age` flags flow into the `where` clause here.

### `retrieval/hybrid_fusion.py` — Step 2 (fusion)
Combines the two retrievers' per-condition lists into a single ranking.

- **`fuse(bm25_results, medcpt_results, k, bm25_wt, medcpt_wt)`**: Reciprocal Rank Fusion, identical formula to TrialGPT:
  `score(trial) += (1 / (rank + k)) × (1 / (condition_idx + 1))`
  summed across both retrievers and all conditions. The `1/(condition_idx+1)` term means the patient's **primary** condition (index 0) contributes full weight and secondary conditions taper off. Setting `bm25_wt` or `medcpt_wt` to 0 drops that retriever. Returns `[(nct_id, score), ...]` sorted descending; the pipeline takes the top-`top_k`.

**Relation to output:** the order here becomes the candidate set; in `--skip-matching` mode this score is reported directly as `retrieval_score`.

---

## `matching/` — Step 3 (per-criterion assessment)

### `matching/criterion_matcher.py`
Assesses each trial's individual inclusion/exclusion criteria against the patient note with Claude.

- Module top: ensures NLTK tokenizers; defines the allowed label sets (`_INCLUSION_LABELS`, `_EXCLUSION_LABELS`).
- **`_format_patient_with_sentence_ids(text)`**: splits the note into sentences and prefixes each with an index (`0. ...`, `1. ...`) so the model can cite evidence by sentence id.
- **`_build_system_prompt(inc_exc)`**: fills `_SYSTEM_TEMPLATE` with the inclusion-vs-exclusion definition and the matching label set. The prompt instructs: judge applicability, look for direct evidence then infer, and (importantly) **if a criterion would clearly appear in a complete note but is absent, assume it's false for this patient**. Output must be `dict{str(index): [reasoning, [sentence_ids], label]}`.
- **`_format_trial_with_criteria(trial_info, inc_exc, criteria)`**: renders the trial (title, target diseases, interventions, summary) plus the numbered criteria list for the prompt.
- **`_parse_criteria_from_chunks(df, nct_id, inc_exc)`**: pulls that trial's criteria from the chunks DataFrame, filtered by `criterion_type` and ordered by `criterion_index`.
- **`CriterionMatcher.match(patient_text, trial_info, criteria_chunks_df, max_criteria)`**: the core. For each of `inclusion` and `exclusion`: gets the criteria (from chunks; **falls back** to splitting raw `inclusion_criteria`/`exclusion_criteria` text if no chunks), caps at `MAX_CRITERIA_PER_TRIAL`, builds the prompt, calls the FAST model at `temperature=0`, strips fences, `json.loads`. On any failure it records `{"error": ...}` so the pipeline continues. A `time.sleep(0.3)` between the two calls eases rate limits. Returns `{"inclusion": {...}, "exclusion": {...}}`.

**Labels:** inclusion → `included` / `not included` / `not applicable` / `not enough information`; exclusion → `excluded` / `not excluded` / `not applicable` / `not enough information`.

**Relation to output:** these labels drive the rule-based score (step 4a) and are sorted into the output's `met_inclusion_criteria`, `unmet_inclusion_criteria`, `triggered_exclusion_criteria`, and `missing_information` lists.

---

## `aggregation/` — Step 4 (scoring & decision)

### `aggregation/score_aggregator.py`
Turns criterion labels into a trial-level decision via a two-stage score.

- **`compute_matching_score(matching_results)`** — *Stage 1, rule-based*. Counts inclusion labels and excluded count, then:
  `score = included / (included + not_included + no_info + ε)`, minus `1.0` if any inclusion is `not included`, minus another `1.0` if any exclusion is `excluded`. Range `[-2, 1]`. Malformed entries (e.g. the `{"error":...}` fallback) are skipped.
- **`aggregate_with_llm(patient_text, matching_results, trial_info)`** — *Stage 2, LLM*. Calls the SMART model with `_AGG_SYSTEM`, which asks for a relevance score **R (0–100)** and an eligibility score **E (−R…R)**, plus explanations. Helpers `_build_agg_user_prompt` and `_format_predictions` assemble the note, trial summary, and a readable dump of the per-criterion predictions. Returns `{relevance_explanation, relevance_score_R, eligibility_explanation, eligibility_score_E}`; on error returns zeros with the error text.
- **`determine_eligibility(matching_score, agg_results)`**: `agg_score = (R + E) / 100` (range `[-2,2]`); `total = matching_score + agg_score` (range `[-4,3]`). Decision: `likely_eligible` if `total ≥ 1.5`, `not_eligible` if `total ≤ -0.5`, else `needs_review`. Normalizes to `[0,1]` as `(total + 4)/7` → the reported `eligibility_score`. Returns `(decision, normalized_score)`.
- **`extract_criteria_breakdown(matching_results, criteria_chunks_df, nct_id)`**: joins labels back to their criterion *text* (via the chunks DataFrame) and buckets them into the four output lists, each entry `{index, criterion, reasoning}`.

**Relation to output:** produces `matching_score`, `relevance_score_R`, `eligibility_score_E`, `eligibility_decision`, `eligibility_score`, the explanations, and the four criteria-breakdown lists.

---

## `pipeline/` — orchestration

### `pipeline/full_pipeline.py`
`MatchingPipeline` ties all four steps together.

- **`__init__`**: instantiates the four stateful components (`KeywordGenerator`, `BM25Retriever`, `MedCPTRetriever`, `CriterionMatcher`); `_ready=False`.
- **`setup(max_trials=None)`**: loads trials + criteria DataFrames, builds/loads the BM25 index, then tries `medcpt.load()` and falls back to `medcpt.build()` if the LanceDB table is missing. Sets `_ready`. Call once.
- **`match_patient(patient_id, patient_text, top_k, skip_matching, sex_filter, age)`**: the per-patient driver.
  - *Step 1*: `keyword_gen.generate(...)` → `conditions`, `summary`.
  - *Step 2*: `bm25.search(conditions, BM25_TOP_N)` and `medcpt.search(conditions, MEDCPT_TOP_N, sex/age filters)` → `fuse(...)` → top-`top_k` `candidate_nctids`.
  - If `skip_matching`: builds lightweight results (`rank, nct_id, title, conditions, clinicaltrials_url, retrieval_score`) and returns early.
  - *Steps 3+4*: loads only the candidates' criteria (`load_criteria_chunks(nct_ids=candidate_nctids)`). For each candidate: `get_trial_info` → `matcher.match` → `compute_matching_score` → `aggregate_with_llm` (with a `0.2s` sleep) → `determine_eligibility` → `extract_criteria_breakdown`. Sorts trials by normalized score descending, assigns `rank`, and assembles the full per-trial result dicts.
  - Returns `{patient_id, summary, keywords, total_candidates, results}`.

**Relation to output:** this method *is* the in-memory shape of `output/<patient_id>.json`.

### `pipeline/build_index.py`
One-time index builder, run before any matching.

- **`main()`**: flags `--max-trials` (subset), `--bm25-only`, `--medcpt-only`. Ensures cache/output dirs, `load_trials(nrows=...)`, then builds the requested index(es). Prints cache locations. The encoding step (MedCPT) is the slow part — warns about CPU time.

**Relation to output:** produces the `cache/` artifacts that retrieval reads; without it, `MedCPTRetriever.load()` fails (the pipeline then auto-builds, but that's slow and unintended for a normal run).

### `*/__init__.py`
All empty — they only mark `retrieval`, `matching`, `aggregation`, `pipeline`, `utils` as importable packages.

---

## `dataset/` — inputs

- **`trials_clean.csv`** (60,337 rows): cleaned trial-level data. Columns include `source_condition_query, nct_id, title, official_title, brief_summary, conditions, interventions, overall_status, study_type, phase, sex, minimum_age, maximum_age, healthy_volunteers, eligibility_criteria, inclusion_criteria, exclusion_criteria, combined_text_for_retrieval, clinicaltrials_url`. `combined_text_for_retrieval` is what MedCPT/BM25 embed/index. Used by `load_trials`/`get_trial_info`.
- **`eligibility_criteria_chunks.csv`** (825,507 rows): criteria split into individual chunks. Columns: `criterion_id, nct_id, title, source_condition_query, conditions, criterion_type` (inclusion/exclusion), `criterion_index, criterion_text, criterion_length, clinicaltrials_url`. Used by the matcher (Step 3) and the breakdown extractor (Step 4).
- **`patient-profiles.jsonl`** (58 lines): one patient per line, `{"_id": "...", "text": "..."}` (SIGIR-format clinical vignettes). The matching input.
- **`raw/clinical_trials_raw_patient2trial_conditions.csv`** (~158 MB): the original download before cleaning. Not used at runtime — provenance only.
- **`statistics/*.csv`**: pre-computed dataset summaries (overview, per-condition counts, chunk counts, sex/phase/status distributions, missing-value report). The corpus spans **8 conditions** (breast cancer, type 2 diabetes, COVID-19, anxiety, COPD, rheumatoid arthritis, glaucoma, sickle cell anemia). Reference only — not imported.

---

## `cache/` — generated artifacts (do not hand-edit)

- **`bm25_corpus.json`**: tokenized corpus + NCT ids for BM25. Reused if present.
- **`lancedb/`**: the LanceDB vector store (`trials` table with embeddings + metadata).
- **`medcpt_nctids.json`**: ordered NCT ids parallel to the embeddings.
- **`keywords_cache.json`**: Step 1 results keyed by `patient_id`.

**Gotcha:** these are reused on existence and ignore changes in the source data/`--max-trials`. Delete them to force a rebuild.

---

## `output/` — results

One `output/<patient_id>.json` per matched patient. Top-level keys: `patient_id`, `summary`, `keywords`, `total_candidates`, `results[]`. Each `results` entry (20 per patient at default `top_k`) carries: identity/metadata (`rank, nct_id, title, conditions, study_type, phase, sex, min_age, max_age, clinicaltrials_url`), decision/scores (`eligibility_decision, eligibility_score, matching_score, relevance_score_R, eligibility_score_E`), the four criteria lists (`met_inclusion_criteria, unmet_inclusion_criteria, triggered_exclusion_criteria, missing_information`), and the two `*_explanation` strings. Each criteria-list item is `{index, criterion, reasoning}`.

(In `--skip-matching` mode, entries instead carry `rank, nct_id, title, conditions, clinicaltrials_url, retrieval_score`.)

---

## `TrialGPT/` — reference implementation (not part of runtime)
A locally cloned copy of the TrialGPT repo, kept for reference. The retrieval design here intentionally mirrors `TrialGPT/trialgpt_retrieval/` (`keyword_generation.py`, `hybrid_fusion_retrieval.py`) — same MedCPT encoders, same BM25 field weighting, same RRF formula. `trialgpt_matching/` and `trialgpt_ranking/` informed Steps 3–4. **Key differences in this prototype:** LanceDB instead of FAISS (enabling sex/age metadata pre-filtering), Claude instead of Azure OpenAI, and a richer JSON output. None of `TrialGPT/` is imported by this project's code.

---

## End-to-end trace (one patient)

1. `run_matching.py` loads the patient dict, calls `pipeline.setup()` (datasets + indices) then `match_patient()`.
2. **Step 1** `keyword_generator` → `summary` + ranked `conditions` (cached). → output `summary`, `keywords`.
3. **Step 2** `bm25_retriever` + `medcpt_retriever` produce per-condition ranked NCT lists; `hybrid_fusion.fuse` merges → top-K `candidate_nctids`. → output `total_candidates`.
4. **Step 3** for each candidate, `criterion_matcher.match` labels every inclusion/exclusion criterion.
5. **Step 4** `compute_matching_score` + `aggregate_with_llm` + `determine_eligibility` → decision + normalized score; `extract_criteria_breakdown` → the four lists. Trials sorted by score, ranked.
6. `run_matching.py` writes `output/<pid>.json` and prints the summary.
