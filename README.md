# Clinical Trial Patient-Matching Prototype

A prototype that matches patient profiles against clinical trial eligibility criteria. Given a free-text patient note, it classifies the patient's therapeutic area, retrieves semantically similar trials via MedCPT + LanceDB (with indication pre-filtering), assesses each trial's criteria with Claude at the criterion level, and produces a ranked eligibility decision per trial.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| **README.md** (this file) | Architecture, output schema, setup, CLI flags, scoring math |
| [CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md) | File-by-file reference — what each file and function does |
| [PIPELINE_DEEP_DIVE.md](PIPELINE_DEEP_DIVE.md) | Why the 5-step design works — rationale and trade-offs |

---

## Pipeline Overview

| Step | What happens | Key component |
|------|-------------|---------------|
| 1 | **Indication classification** — Claude maps the patient note to one of 8 therapeutic area strings, used as a LanceDB pre-filter | `retrieval/indication_classifier.py` |
| 2 | **Keyword extraction** — Claude extracts ranked medical conditions from the note as structured search queries | `retrieval/keyword_generator.py` |
| 3 | **Semantic retrieval** — MedCPT encodes conditions as queries; LanceDB ANN search (pre-filtered by indication + sex/age) returns candidates per condition; RRF fuses them | `retrieval/medcpt_retriever.py` |
| 4 | **Criterion matching** — Claude assesses each inclusion and exclusion criterion individually, with per-criterion confidence | `matching/criterion_matcher.py` |
| 5 | **Aggregation & ranking** — Rule-based score + LLM relevance/eligibility scores + confidence demotion gate → final ranked output | `aggregation/score_aggregator.py` |

---

## Project Structure

```
.
├── config.py                        # Central configuration (paths, models, thresholds)
├── run_matching.py                  # CLI entry point
├── requirements.txt
├── .env.example
│
├── pipeline/
│   ├── build_index.py               # One-time index builder (encode trials → LanceDB)
│   └── full_pipeline.py             # MatchingPipeline orchestrating all 5 steps
│
├── retrieval/
│   ├── indication_classifier.py     # Claude-based therapeutic area classification
│   ├── keyword_generator.py         # Claude-based keyword/condition extraction
│   └── medcpt_retriever.py          # MedCPT + LanceDB ANN retrieval
│
├── matching/
│   └── criterion_matcher.py         # Per-criterion LLM eligibility assessment
│
├── aggregation/
│   └── score_aggregator.py          # Scoring, confidence, eligibility decision
│
├── utils/
│   ├── data_loader.py               # Dataset loading helpers
│   └── json_utils.py                # Shared JSON fence-stripping utility
│
├── dataset/
│   ├── trials_clean.csv             # 60,337 clinical trials
│   └── eligibility_criteria_chunks.csv  # 825,507 individual criterion chunks
│
├── patient-datasets/                # One folder per patient condition
│   ├── breast-cancer/               #   patient_summary.md  ← pipeline input
│   ├── type2-diabetes/
│   └── ...                          # (8 conditions total)
│
├── cache/                           # Auto-created: LanceDB files, keywords cache
└── output/                          # Auto-created: per-patient JSON results
```

---

## Quick Start

> **Cloning note:** the two large datasets (`dataset/trials_clean.csv`,
> `dataset/eligibility_criteria_chunks.csv`) are stored with **[Git LFS](https://git-lfs.com)**.
> Install it *before* cloning:
> ```bash
> git lfs install
> git clone <repo-url>
> ```
> Already cloned without LFS? Run `git lfs install && git lfs pull`.

### 1. Environment and dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"   # one-time tokenizer
```

> **PyTorch note:** the above installs the CPU build. For GPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
> ```

### 2. Configure API key

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
```

### 3. Build the retrieval index (one-time)

Encodes all 60K trials with MedCPT-Article-Encoder and stores them in LanceDB.

| Hardware | Estimated time |
|----------|----------------|
| GPU (T4 / A10) | ~10–20 min |
| Apple M-series (MPS) | ~30–60 min |
| CPU only | ~2–6 hours |


```bash
python pipeline/build_index.py                    # full corpus
python pipeline/build_index.py --max-trials 1000  # fast smoke test
```

> If you previously built with `--max-trials N` and want the full corpus,
> delete `cache/lancedb/` and `cache/medcpt_nctids.json` first — the builder
> silently reuses an existing table.

### 4. Run patient matching

```bash
# Full pipeline — single patient
python run_matching.py --patient-indication breast-cancer

# All 8 patients with verbose criterion breakdown
python run_matching.py --all -v

# Retrieval only (skip LLM criterion assessment — fast, no cost)
python run_matching.py --patient-indication breast-cancer --skip-matching

# Override auto-parsed sex/age (parsed from patient_summary.md Sex/Age row)
python run_matching.py --patient-indication breast-cancer --sex FEMALE --age 70

# Quick test using a 1000-trial index
python run_matching.py --patient-indication breast-cancer --max-trials 1000
```

---

## Output

Results saved as JSON to `output/<patient_id>.json` and summarised on the console.

### Console output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PATIENT  BREAST-CANCER
  58-year-old woman with HER2-positive metastatic breast cancer ...
  Keywords : HER2-positive breast cancer, trastuzumab, metastatic ...
  Trials   : 10 evaluated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [ 1]  ✓ LIKELY ELIGIBLE   0.821  (conf 0.87)  NCT04512345
        HER2-Positive Breast Cancer Phase III Trial

  [ 2]  ? NEEDS REVIEW      0.714  (conf 0.52)  NCT03891234
        ...

  [ 3]  ✗ NOT ELIGIBLE      0.214  (conf 0.91)  NCT02345678
        ...
```

`(conf N/A)` appears when no applicable criteria provided confidence values (e.g. retrieval-only mode or all criteria labeled `not applicable`).

### JSON output schema

```json
{
  "patient_id": "breast-cancer",
  "indication": "breast cancer",
  "summary": "58-year-old woman with HER2-positive metastatic breast cancer ...",
  "keywords": ["HER2-positive breast cancer", "trastuzumab", ...],
  "total_candidates": 10,
  "results": [
    {
      "rank": 1,
      "nct_id": "NCT04512345",
      "title": "...",
      "conditions": "...",
      "study_type": "INTERVENTIONAL",
      "phase": "PHASE3",
      "sex": "ALL",
      "min_age": "18 Years",
      "max_age": "80 Years",
      "clinicaltrials_url": "https://clinicaltrials.gov/study/NCT04512345",
      "eligibility_decision": "likely_eligible",
      "eligibility_score": 0.821,
      "confidence": 0.87,
      "matching_score": 0.75,
      "eligibility_score_E": 70,
      "met_inclusion_criteria": [
        {"index": "0", "criterion": "HER2-positive breast cancer", "reasoning": "..."}
      ],
      "unmet_inclusion_criteria": [],
      "triggered_exclusion_criteria": [],
      "missing_information": [
        {"index": "3", "criterion": "ECOG PS ≤ 2", "reasoning": "Performance status not documented"}
      ],
      "eligibility_explanation": "Patient meets most criteria; performance status unknown.",
      "_debug": {
        "relevance_score_R": 85,
        "relevance_explanation": "Patient conditions align well with trial target population.",
        "matching_errors": []
      }
    }
  ]
}
```

**Field notes:**
- `eligibility_score` — normalized total score ∈ [0, 1]; primary sort key within each decision tier
- `confidence` — mean LLM-reported confidence across applicable criteria ∈ [0, 1]; `null` when unavailable
- `eligibility_score_E` — raw LLM eligibility score (−R to R); kept for analysis
- `_debug.relevance_score_R` — LLM relevance rating (0–100); informational, not used in ranking
- `_debug.matching_errors` — list of criterion groups (`"inclusion"` / `"exclusion"`) where Claude returned an error instead of predictions

### Eligibility decisions

| Decision | Meaning |
|----------|---------|
| `likely_eligible` | Score ≥ threshold and confidence ≥ 0.6 (or no confidence data) |
| `needs_review` | Borderline score, or `likely_eligible` demoted by low confidence |
| `not_eligible` | Score ≤ lower threshold |

---

## Scoring

### Step 4 — Criterion matching labels

Each criterion gets a 4-element response from Claude: `[reasoning, [sentence_ids], label, confidence]`.

| Label | Meaning |
|-------|---------|
| `included` / `excluded` | Patient clearly meets / triggers this criterion |
| `not included` / `not excluded` | Patient clearly does not meet / trigger this criterion |
| `not applicable` | Criterion is irrelevant to this patient |
| `not enough information` | Patient note lacks the data to decide |

Confidence ∈ [0, 1]: Claude is instructed to report ≥ 0.8 only when there is direct explicit evidence in the patient note; 0.5 for inferred or uncertain assessments.

### Step 5 — Score computation

```
# Stage 1: rule-based matching score  ∈ [-2, 1]
matching_score = included / (included + not_included + no_info + ε)
               - 1.0  if any not_included
               - 1.0  if any excluded

# Stage 2: LLM aggregation  ∈ [-2, 2]
#   R ∈ [0, 100]:   relevance of trial to patient condition
#   E ∈ [-R, R]:    patient eligibility given criteria predictions
agg_score = (R + E) / 100

# Combined
total_score        = matching_score + agg_score   ∈ [-4, 3]
eligibility_score  = (total_score + 4) / 7        ∈ [0, 1]
```

**Decision thresholds** (configurable in `config.py`):
- `likely_eligible` : `total_score ≥ 1.5`
- `not_eligible`    : `total_score ≤ -0.5`
- `needs_review`    : otherwise

**Confidence demotion** (applied after thresholding):
- If decision is `likely_eligible` and `confidence < 0.6` → demoted to `needs_review`
- Demoted trial's `eligibility_score` is capped just below the `likely_eligible` boundary (≈ 0.786) so demoted trials always rank after genuine `likely_eligible` ones
- Gate is skipped entirely when `confidence` is `None` (no usable measurements)

### Sort order

```
(decision_priority, -eligibility_score, -confidence, nct_id)
```

`decision_priority`: `likely_eligible=0`, `needs_review=1`, `not_eligible=2` — all eligible trials always precede all review trials regardless of score.

---

## Technical Architecture

```
Patient Note
    │
    ▼
[Step 1] IndicationClassifier (Claude Haiku)
    │  → "breast cancer"  (one of 8 therapeutic area strings)
    │    Used as LanceDB source_condition_query pre-filter
    │
    ▼
[Step 2] KeywordGenerator (Claude Haiku)
    │  → {"summary": "...", "conditions": ["HER2-positive breast cancer", ...]}
    │    Active + historical conditions only; negated/hypothetical/family dropped
    │    Cached by patient_id; re-runs skip API call
    │
    ▼
[Step 3] MedCPTRetriever (LanceDB)
    │  MedCPT-Query-Encoder encodes each condition → 768-dim vector
    │  LanceDB ANN search per condition, pre-filtered by:
    │    - indication (source_condition_query = "breast cancer")
    │    - sex / age  (optional, from patient_summary.md or --sex / --age)
    │  RRF fusion across per-condition results:
    │    score += (1 / (rank + 20)) × (1 / (condition_idx + 1))
    │  → top-K candidate NCT IDs
    │
    ▼
[Step 4] CriterionMatcher (Claude Haiku)
    │  For each candidate × {inclusion, exclusion}:
    │    One Claude call per group
    │    Patient note is numbered sentence-by-sentence for evidence citation
    │    → {criterion_idx: [reasoning, [sent_ids], label, confidence]}
    │
    ▼
[Step 5] ScoreAggregator
    ├─ compute_matching_score()  — rule-based  ∈ [-2, 1]
    ├─ compute_confidence()      — mean criterion confidence ∈ [0,1] or None
    ├─ aggregate_with_llm()      (Claude Sonnet) → R, E scores
    └─ determine_eligibility()   → decision + normalised score + demotion gate
         → ranked results
```

**Model assignment:**
- Claude **Haiku** — Steps 1, 2, 4 (high-volume structured extraction; ~22 calls per top-10 run)
- Claude **Sonnet** — Step 5 aggregation only (holistic judgment; 1 call per trial)
- Total API calls for top-10: ~32 (first run); ~31 on re-run (Step 2 cached)

---

## Configuration

All values in `config.py`; most are overridable via environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Set in `.env` |
| `CLAUDE_MODEL_FAST` | `claude-haiku-4-5-20251001` | Steps 1, 2, 4 |
| `CLAUDE_MODEL_SMART` | `claude-sonnet-4-6` | Step 5 aggregation |
| `TOP_K_RETRIEVAL` | `10` | Trials retrieved and assessed per patient |
| `MAX_CONDITIONS_PER_PATIENT` | `12` | Max condition queries sent to MedCPT |
| `MAX_CRITERIA_PER_TRIAL` | `50` | Max criteria per inc/exc group (cost cap) |
| `CONFIDENCE_REVIEW_THRESHOLD` | `0.6` | Confidence below which `likely_eligible` → `needs_review` |
| `LIKELY_ELIGIBLE_THRESHOLD` | `1.5` | total_score threshold for `likely_eligible` |
| `NOT_ELIGIBLE_THRESHOLD` | `-0.5` | total_score threshold for `not_eligible` |
| `MEDCPT_BATCH_SIZE` | `32` | Encoding batch size (reduce if GPU OOM) |

---

## Limitations

- **Index build time:** CPU encoding of 60K trials takes 2–6 hours. Use a GPU or `--max-trials` for testing.
- **Stale index:** the builder silently reuses an existing `cache/lancedb/` table. Delete it before rebuilding on different data.
- **Keyword cache:** Step 2 results are cached by `patient_id` in `cache/keywords_cache.json`. Clear it to force re-extraction.
- **LLM cost:** top-10 full pipeline = ~32 Claude API calls. Use `--skip-matching` for retrieval-only results.
- **Criterion cap:** `MAX_CRITERIA_PER_TRIAL=50` skips longer criteria lists to control cost.
- **Sex/age parsing:** auto-parsed from the `Sex / Age` row in each `patient_summary.md`. Use `--sex` / `--age` to override.
- **Confidence gate is one-directional:** low confidence can only demote `likely_eligible` → `needs_review`, never promote upward.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `trials_clean.csv` | 60,337 | Clinical trials with eligibility criteria |
| `eligibility_criteria_chunks.csv` | 825,507 | Individual criteria (372K inclusion, 453K exclusion) |
| `patient-datasets/<condition>/` | 8 | Synthetic longitudinal patient records |

Conditions: breast cancer, type 2 diabetes, COVID-19, anxiety, COPD, rheumatoid arthritis, glaucoma, sickle cell anemia.

---

## Relationship to TrialGPT

The semantic retrieval layer (Step 3) uses the same MedCPT encoders as TrialGPT (`ncbi/MedCPT-Article-Encoder` / `ncbi/MedCPT-Query-Encoder`) and the same RRF fusion formula. Key differences:

| Aspect | TrialGPT | This prototype |
|--------|----------|----------------|
| Retrieval | BM25 + MedCPT + RRF | MedCPT + LanceDB only (BM25 removed) |
| Pre-filtering | None | LanceDB indication + sex + age filters |
| LLM | Azure OpenAI | Claude (Haiku for extraction, Sonnet for aggregation) |
| Criterion output | Label only | Label + confidence + reasoning + sentence citations |
| Output | Score only | Rich JSON with criterion breakdown, confidence, decisions |

The TrialGPT paper is referenced for architecture context only — none of its code is present or imported at runtime.

---

## License

Research prototype. MedCPT models are from NCBI and subject to HuggingFace model terms.
