# Clinical Trial Patient-Matching Prototype

A prototype system that matches patient profiles against clinical trial eligibility criteria using a hybrid retrieval approach (BM25 + MedCPT semantic search) backed by a LanceDB vector store, with LLM-powered criterion-level assessment.

- **BM25** — a standard probabilistic algorithm for keyword-based (lexical) search.
- **MedCPT** — NCBI's biomedical text embedding model ([query encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder)) for semantic search.

## Overview

Clinical trial recruitment is slow because coordinators must manually compare free-text eligibility criteria against patient profiles at scale. This prototype automates that process through a four-step pipeline:

| Step | What happens | Key component |
|------|-------------|---------------|
| 1 | **Keyword extraction** — LLM reads the patient note and extracts ranked medical conditions as search queries in structured JSON format | `retrieval/keyword_generator.py` |
| 2 | **Hybrid retrieval** — BM25 (keyword / lexical) + MedCPT (semantic) ranked lists fused with RRF → top-K candidate trials | `retrieval/`, `pipeline/build_index.py` |
| 3 | **Criterion-level assessment** — LLM assesses each inclusion/exclusion criterion individually: `included` / `not included` / `not applicable` / `not enough information` | `matching/criterion_matcher.py` |
| 4 | **Aggregation & ranking** — Rule-based matching score + LLM relevance/eligibility scores → final ranked output | `aggregation/score_aggregator.py` |


---

## Documentation

This repo ships three complementary docs. Start here, then dig in as needed:

| Doc | Purpose | Read it when you want to… |
|-----|---------|---------------------------|
| **README.md** (this file) | Architecture, output format, setup, and spec | Install, run, and understand the pipeline's shape and outputs |
| [CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md) | A file-by-file reference — *what each file and function is* | Find where something lives or what a specific module does |
| [PIPELINE_DEEP_DIVE.md](PIPELINE_DEEP_DIVE.md) | A step-by-step explanation of *why the design works* | Understand the reasoning and trade-offs behind each of the 4 steps |

---

## Project Structure

```
.
├── config.py                     # Central configuration (paths, models, thresholds)
├── run_matching.py               # Main CLI entry point
├── requirements.txt
├── .env.example
│
├── pipeline/
│   ├── build_index.py            # One-time index builder (run before matching)
│   └── full_pipeline.py          # MatchingPipeline class orchestrating all steps
│
├── retrieval/
│   ├── keyword_generator.py      # Claude-based keyword extraction
│   ├── bm25_retriever.py         # Weighted BM25 index
│   ├── medcpt_retriever.py       # MedCPT + LanceDB vector store
│   └── hybrid_fusion.py          # Reciprocal Rank Fusion
│
├── matching/
│   └── criterion_matcher.py      # Per-criterion LLM eligibility assessment
│
├── aggregation/
│   └── score_aggregator.py       # Score aggregation + eligibility decision
│
├── utils/
│   └── data_loader.py            # Dataset loading helpers
│
├── dataset/
│   ├── trials_clean.csv          # 60,337 clinical trials
│   ├── eligibility_criteria_chunks.csv  # 825,507 individual criterion chunks
│   └── patient-profiles.jsonl    # 58 patient case descriptions
│
├── cache/                        # Auto-created: BM25 JSON + LanceDB files
└── output/                       # Auto-created: per-patient JSON results
```

---

## Quick Start

> **Cloning note:** the two large datasets (`dataset/trials_clean.csv`,
> `dataset/eligibility_criteria_chunks.csv`) are stored with **[Git LFS](https://git-lfs.com)**.
> Install it *before* cloning so they download as real files, not pointers:
> ```bash
> git lfs install
> git clone https://github.com/spandandutta7/Saama---patient-trial-matching-prototype.git
> ```
> (Already cloned without LFS? Run `git lfs install && git lfs pull`.)
> The raw pre-cleaning data (`dataset/raw/`) is not tracked — it is provenance only and not needed to run.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch note:** The command above installs the CPU build. For GPU support:
> ```bash
> # CUDA 12.1
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

Download the NLTK tokeniser (once):
```python
python -c "import nltk; nltk.download('punkt_tab')"
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

### 3. Build the retrieval index (one-time setup)

This encodes all 60K trials with MedCPT and stores them in LanceDB. **Encoding time depends on hardware:**

| Hardware | Estimated time |
|----------|----------------|
| GPU (T4/A10) | ~10–20 min |
| Apple M-series (MPS) | ~30–60 min |
| CPU only | ~2–6 hours |

```bash
python pipeline/build_index.py
```

For a quick smoke test on a small subset:
```bash
python pipeline/build_index.py --max-trials 1000
```

BM25 only (no MedCPT encoding, very fast):
```bash
python pipeline/build_index.py --bm25-only
```

### 4. Run patient matching

```bash
# Match a single patient (full pipeline)
python run_matching.py --patient-id sigir-20141

# Match all 58 patients, top-10 trials each
python run_matching.py --all --top-k 10

# Retrieval only (no LLM criterion assessment, much faster/cheaper)
python run_matching.py --patient-id sigir-20141 --skip-matching

# With patient metadata for LanceDB pre-filtering
python run_matching.py --patient-id sigir-20141 --sex FEMALE --age 58

# Verbose output with full criterion breakdown
python run_matching.py --patient-id sigir-20141 --verbose

# Quick test using only 1000-trial index
python run_matching.py --patient-id sigir-20141 --max-trials 1000
```

---

## Output

Results are saved as JSON to `output/<patient_id>.json` and summarised in the console.

### Console output example
```
======================================================================
Patient: sigir-20141
Summary: 58-year-old woman with chest pain and cardiovascular risk factors
Keywords: chest pain, hypertension, obesity, angina pectoris, ...
Candidates evaluated: 20
======================================================================

  [ 1] ✓ LIKELY_ELIGIBLE     score=0.821   NCT04512345
       Title: Aspirin in Stable Angina with Hypertension
       Conditions: Angina Pectoris | Hypertension
       Met inclusion (3): Age 40-75 years; Chest pain; Hypertension diagnosis
       Missing info (1): Echocardiogram results

  [ 2] ? NEEDS_REVIEW         score=0.571   NCT03891234
       ...

  [ 3] ✗ NOT_ELIGIBLE         score=0.214   NCT02345678
       ...
```

### JSON output schema
```json
{
  "patient_id": "sigir-20141",
  "summary": "58-year-old woman with chest pain and cardiovascular risk factors",
  "keywords": ["chest pain", "hypertension", "obesity", ...],
  "total_candidates": 20,
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
      "matching_score": 0.75,
      "relevance_score_R": 85,
      "eligibility_score_E": 70,
      "met_inclusion_criteria": [
        {"index": "0", "criterion": "Age 18-80 years", "reasoning": "Patient is 58 years old"}
      ],
      "unmet_inclusion_criteria": [],
      "triggered_exclusion_criteria": [],
      "missing_information": [
        {"index": "2", "criterion": "LVEF > 40%", "reasoning": "No echocardiogram data in note"}
      ],
      "relevance_explanation": "Patient's chest pain and hypertension align well ...",
      "eligibility_explanation": "Patient meets most criteria but echocardiogram data missing ..."
    }
  ]
}
```

### Eligibility decisions
| Decision | Meaning |
|----------|---------|
| `likely_eligible` | Meets most criteria, no hard exclusions triggered |
| `not_eligible` | Failed one or more inclusion criteria, or triggered exclusion |
| `needs_review` | Insufficient information to determine, or borderline scores |

---

## Scoring Details

### Step 3 — Criterion matching labels
| Label | Meaning |
|-------|---------|
| `included` / `excluded` | Patient clearly meets / triggers this criterion |
| `not included` / `not excluded` | Patient clearly does not meet / trigger this criterion |
| `not applicable` | Criterion is irrelevant to this patient |
| `not enough information` | Patient note lacks the data to decide |

### Step 4 — Score components
```
matching_score = included / (included + not_included + no_info + ε)
               - 1.0  if any not_included
               - 1.0  if any excluded
               ∈ [-2, 1]

agg_score = (relevance_R + eligibility_E) / 100   ∈ [-2, 2]

total_score = matching_score + agg_score           ∈ [-4, 3]

eligibility_score (reported) = (total + 4) / 7    ∈ [0, 1]
```

Thresholds (configurable in `config.py`):
- `likely_eligible` : `total_score ≥ 1.5`
- `not_eligible`    : `total_score ≤ -0.5`
- `needs_review`    : otherwise

---

## Technical Architecture

```
Patient Note
    │
    ▼
[Step 1] KeywordGenerator (Claude claude-haiku-4-5)
    │  → {"summary": "...", "conditions": ["hypertension", "chest pain", ...]}
    │
    ▼
[Step 2] Hybrid Retrieval
    ├─ BM25Retriever
    │    BM25 index on trials_clean.csv
    │    Title ×3 | Conditions ×2 | Text ×1
    │    → per-condition ranked NCT ID lists
    │
    ├─ MedCPTRetriever (LanceDB)
    │    MedCPT-Article-Encoder encodes trials (768-dim CLS embedding)
    │    MedCPT-Query-Encoder encodes conditions
    │    LanceDB ANN search with optional metadata pre-filtering
    │    → per-condition ranked NCT ID lists
    │
    └─ hybrid_fusion.fuse() — Reciprocal Rank Fusion
         score += (1/(rank+k)) × (1/(cond_idx+1))
         → top-K candidate NCT IDs
    │
    ▼
[Step 3] CriterionMatcher (Claude claude-haiku-4-5)
    │  For each candidate trial × {inclusion, exclusion}:
    │    One Claude call per group → {idx: [reasoning, [sent_ids], label]}
    │
    ▼
[Step 4] ScoreAggregator
    ├─ compute_matching_score() — rule-based
    ├─ aggregate_with_llm() (Claude claude-sonnet-4-6)
    │    → relevance_R, eligibility_E
    └─ determine_eligibility() → decision + normalised score
```

> **Going deeper:** for a function-by-function walkthrough of all four pipeline steps —
> the design choices behind retrieval, criterion matching, and aggregation — see
> [PIPELINE_DEEP_DIVE.md](PIPELINE_DEEP_DIVE.md).

---

## Configuration

Key settings in `config.py` (override via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Set in `.env` |
| `CLAUDE_MODEL_FAST` | `claude-haiku-4-5-20251001` | Used for keyword gen + criterion matching |
| `CLAUDE_MODEL_SMART` | `claude-sonnet-4-6` | Used for trial-level aggregation |
| `TOP_K_RETRIEVAL` | `20` | Trials retrieved and assessed per patient |
| `MAX_CRITERIA_PER_TRIAL` | `50` | Max criteria sent to LLM (cost control) |
| `MEDCPT_BATCH_SIZE` | `32` | Encoding batch size (reduce if OOM) |

---

## Limitations

- **Index build time:** Encoding 60K trials with MedCPT is slow on CPU. Use a GPU or the `--max-trials` flag for testing.
- **LLM cost:** Full pipeline for one patient (top-20 trials) makes ~42 Claude API calls. Use `--skip-matching` for retrieval-only results.
- **Criterion cap:** `MAX_CRITERIA_PER_TRIAL=50` skips longer criteria lists to control cost.
- **No age/sex extraction:** `--sex` and `--age` flags must be provided manually; the pipeline does not auto-parse them from the patient note.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `trials_clean.csv` | 60,337 | Clinical trials with eligibility criteria |
| `eligibility_criteria_chunks.csv` | 825,507 | Individual criteria (372K inclusion, 453K exclusion) |
| `patient-profiles.jsonl` | 58 | Patient case descriptions (SIGIR 2016 format) |

Trials span 8 conditions: breast cancer, type 2 diabetes, COVID-19, anxiety, COPD, rheumatoid arthritis, glaucoma, sickle cell anemia.

---

## Relationship to TrialGPT

The retrieval layer (Steps 1–2) closely follows **TrialGPT** (`TrialGPT/trialgpt_retrieval/`,
cloned locally for reference): the same MedCPT encoders
(`ncbi/MedCPT-Article-Encoder` / `ncbi/MedCPT-Query-Encoder`), the same BM25 field weighting
(title ×3, conditions ×2, text ×1), and the same Reciprocal Rank Fusion formula. Key
differences in this prototype:

| Aspect | TrialGPT | This prototype |
|--------|----------|----------------|
| Vector store | FAISS (ANN only) | LanceDB (ANN **+** metadata filtering by sex/age) |
| LLM | Azure OpenAI | Claude (Haiku for matching, Sonnet for aggregation) |
| Retrieval algorithm | BM25 + MedCPT + RRF | Identical |
| Output | Score only | Rich JSON with criterion breakdown, decisions, explanations |

None of `TrialGPT/` is imported at runtime — it is reference material only.

---

## License

This project is a research prototype. The TrialGPT reference code in `TrialGPT/` is subject to its own license. MedCPT models are from NCBI and subject to HuggingFace model terms.
