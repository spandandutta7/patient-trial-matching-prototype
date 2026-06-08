# Pipeline Deep Dive

Why the 5-step design works — rationale and trade-offs for each stage. For what each file and function *is*, see [CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md).

---

## The core problem

Clinical trial matching requires two things that are difficult to do simultaneously: broad recall (don't miss trials the patient might qualify for) and precise assessment (accurately determine eligibility against structured criteria). No single technique does both well, so the pipeline separates them into distinct stages and uses the right tool for each.

---

## Step 1 — Indication Classification

**What:** Claude (Haiku) reads the patient note and returns one of 8 fixed `source_condition_query` strings (e.g. `"breast cancer"`).

**Why:** The trial corpus has ~60K trials. Running a full ANN search across all 60K for every query is slow and wastes compute on completely irrelevant trials. Classifying first reduces the search space by 73–98% depending on condition — a breast cancer query, for example, goes from 60K to ~1K–2K trials before a single vector multiplication is computed.

**Why LLM classification instead of a keyword match:** Patient notes use varied terminology. "HER2+ metastatic adenocarcinoma of the breast" should resolve to `"breast cancer"`, but a naive keyword rule would need to enumerate every variant. Claude handles this robustly with a single zero-shot call.

**Failure mode:** If Claude returns an unrecognized string, the classifier raises `ValueError`; `full_pipeline.py` catches it, logs a warning, and falls back to no filter. This degrades to a 60K search (slower, but no crash or missed results).

---

## Step 2 — Keyword Extraction

**What:** Claude (Haiku) returns a ranked list of up to 12 medical terms extracted from the patient note, categorized by status (active, historical, negated, hypothetical, family_history). Only `active` and `historical` terms are kept as retrieval queries.

**Why not just encode the full patient note as one query?** MedCPT is a bi-encoder: query and document are encoded independently and compared by cosine similarity. A full patient note produces a single averaged vector that blends many concepts together. Extracting specific conditions (e.g. `"HER2-positive breast cancer"`, `"trastuzumab"`, `"liver metastases"`) as separate queries produces sharper vectors that retrieve more relevant trials per condition. Running one ANN search per condition and fusing the results (Step 3) consistently outperforms the single-query approach.

**Why drop negated/hypothetical/family_history?** A negated condition like "no prior anthracyclines" should not be used as a retrieval query — it would surface trials that require prior anthracyclines, the opposite of what we want. The status field lets Claude make this judgment reliably.

**Caching:** Results are cached by `patient_id` in `cache/keywords_cache.json`. Keywords don't change across runs for the same patient, so there's no reason to re-call the LLM. Cache entries are invalidated by `_CACHE_SCHEMA_VERSION` when the prompt schema changes.

---

## Step 3 — Semantic Retrieval

**What:** Each extracted condition is encoded with `ncbi/MedCPT-Query-Encoder`. One LanceDB ANN search runs per condition, pre-filtered by the indication from Step 1 (and optionally sex/age). Per-condition result lists are merged with Reciprocal Rank Fusion (RRF), with earlier (higher-ranked) conditions weighted more heavily.

**Why MedCPT specifically?** MedCPT is trained on PubMed article-query pairs from the clinical domain, making it far better at biomedical semantic similarity than general-purpose embedders. It uses a bi-encoder architecture (separate encoders for articles and queries) optimized for retrieval, not reranking.

**Why LanceDB instead of FAISS?** FAISS is ANN-only — metadata filtering must be done post-retrieval, which wastes recall: you over-fetch and then discard. LanceDB runs the `source_condition_query` pre-filter *inside* the ANN index, so the 2000 results per condition come only from the relevant therapeutic area. Sex and age filters work the same way.

**Why exclude exclusion criteria from trial embeddings?** Trials are encoded as `(title, brief_summary + inclusion_criteria)` — exclusion criteria are deliberately left out. MedCPT embeddings are nearly blind to negation: "No prior anthracyclines" and "prior anthracyclines" produce nearly identical vectors. Including exclusion text would cause spurious matches: a patient querying for "anthracyclines" would surface trials that exclude anthracycline-treated patients. Exclusion criteria are assessed precisely in Step 4 instead.

**RRF fusion and condition weighting:**
```
score(nct_id) += (1 / (rank + 20)) × (1 / (condition_idx + 1))
```
The `1 / (rank + 20)` term is standard RRF — it rewards consistently high ranks across conditions. The `1 / (condition_idx + 1)` term is a domain-specific addition: Claude returns conditions ranked by importance, so the first condition (primary diagnosis) should dominate the fusion. Later conditions (comorbidities, prior treatments) refine but don't override.

**Index caching:** The LanceDB table is written once and silently reused on subsequent runs. This is intentional — encoding 60K trials takes 2–6 hours on CPU. To rebuild on different data, delete `cache/lancedb/` and `cache/medcpt_nctids.json` first.

---

## Step 4 — Criterion Matching

**What:** For each candidate trial, two Claude (Haiku) calls — one for inclusion criteria, one for exclusion. Each criterion gets a 4-element response: `[reasoning, [sentence_ids], label, confidence]`.

**Why two calls instead of one?** Inclusion and exclusion criteria have opposite logic and different valid label sets. A single prompt mixing both would force Claude to track two incompatible decision frames simultaneously. Splitting the calls keeps each prompt focused and reduces label confusion.

**Why numbered sentence IDs?** The patient note is rendered with per-sentence IDs (0, 1, 2, ...) so Claude can cite specific sentences as evidence (`"sentence_ids": [3, 7]`). This serves two purposes: (1) it forces grounding in the actual note rather than hallucinated inferences, and (2) it produces an evidence trail useful for debugging false positives and false negatives.

**Why a confidence score per criterion?** The eligibility label alone doesn't capture how certain Claude is. A criterion labeled `"included"` could be based on an explicit statement in the note ("diagnosed with HER2+ breast cancer"), an inference ("tumor markers consistent with HER2 positivity"), or a guess ("no evidence to the contrary"). The confidence score (≥ 0.8 for explicit evidence, ~0.5 for inference) captures this distinction and feeds the demotion gate in Step 5.

**Why Haiku for this step?** Step 4 makes 2 API calls per candidate trial — 20 calls for the default top-10. Using Sonnet here would multiply API cost by ~5×. The task (structured extraction + classification) is well within Haiku's capabilities; Sonnet is reserved for the holistic judgment in Step 5 where model quality has more impact.

**Criteria fallback:** `eligibility_criteria_chunks.csv` contains pre-split, ordered criteria for most trials. For trials without chunks, the matcher splits the raw `inclusion_criteria` / `exclusion_criteria` text from `trials_clean.csv` on newlines. The chunked version is preferred because it preserves the original criterion boundaries more cleanly.

**Cost cap:** `MAX_CRITERIA_PER_TRIAL = 50` limits the criteria sent per group. Trials with more criteria than this have their lower-priority criteria silently dropped. This is a cost control decision, not a quality decision — longer criteria lists tend to have more edge-case and administrative criteria anyway.

---

## Step 5 — Aggregation and Scoring

Step 5 has four sub-steps that run sequentially for each candidate.

### 5a — Rule-based matching score

```
score = included / (included + not_included + no_info + ε)
score -= 1.0  if any not_included
score -= 1.0  if any excluded
```

Range: [−2, 1]. The ratio component rewards meeting more inclusion criteria; the two penalties are hard deductions for hard disqualifiers. This is a deterministic, fast baseline that catches clear cases (many met criteria, no exclusions) and clear non-cases (exclusions triggered).

**Why a ratio instead of a count?** Trials vary wildly in the number of criteria. A raw count would systematically favor trials with few criteria. The ratio normalizes across trials.

**Why are `not_included` and `excluded` treated as hard deductions rather than ratio terms?** Because they are qualitatively different from `not enough information`. A patient who doesn't meet a required criterion is probably ineligible — a 1.0 deduction reflects that severity. `not enough information` is softer: the patient might still be eligible but the note is incomplete.

### 5b — Confidence computation

`compute_confidence()` returns the mean LLM-reported confidence across all applicable criteria (excluding `not applicable` labels, which represent irrelevance certainty rather than eligibility certainty). Returns `None` when no usable measurements exist.

**Why exclude `not applicable`?** If Claude labels a criterion "not applicable" with confidence 0.9, that 0.9 reflects how certain Claude is that the criterion doesn't apply — not how confident it is in the patient's eligibility. Including it would dilute the mean and give a misleadingly high confidence for a patient where no criteria could actually be assessed.

**Why return `None` rather than 0.0 when there are no measurements?** `None` means "no data." `0.0` means "measured, low confidence." Conflating them would cause the demotion gate to fire on trials where Claude simply had nothing to assess (e.g. a trial with no applicable criteria, or old-schema 3-element responses). The gate is skipped when `confidence is None` — absence of measurement is not evidence of low confidence.

### 5c — LLM aggregation

Claude (Sonnet) receives the patient note, trial metadata, and the full set of criterion-level predictions and returns two scores:

- **R ∈ [0, 100]:** Overall relevance — how well the trial's target condition matches the patient's condition
- **E ∈ [−R, R]:** Eligibility — how eligible this patient is, given the criteria predictions

`agg_score = (R + E) / 100`, range [−2, 2].

**Why a second LLM pass after the rule-based score?** The rule-based score treats all criteria equally. In reality, one hard exclusion criterion matters far more than five soft `not_included` ones, and a highly relevant trial with one missing-information criterion should rank above an irrelevant trial with perfect criterion coverage. Sonnet's holistic judgment captures these distinctions. The combination of deterministic + LLM scores is more robust than either alone.

**Why Sonnet (not Haiku) here?** Step 5 runs once per trial (10 calls for top-10), not once per criterion. The task requires synthesizing multiple pieces of evidence into a nuanced judgment — a task where the larger model's reasoning quality has real impact on ranking accuracy. The volume is low enough to absorb the cost.

### Decision and demotion

`total = matching_score + agg_score`, range [−4, 3].

| total | decision |
|-------|----------|
| ≥ 1.5 | `likely_eligible` |
| ≤ −0.5 | `not_eligible` |
| otherwise | `needs_review` |

After thresholding, the **confidence demotion gate** runs: if the decision is `likely_eligible` and `confidence < 0.6`, the decision is demoted to `needs_review`. This implements the "recall-first" principle — it's better to surface a borderline trial for human review than to confidently declare eligibility based on uncertain evidence.

Demotion is one-directional: low confidence can never promote a trial upward.

When a trial is demoted, its `norm_score` is capped just below the `likely_eligible` boundary (≈ 0.785) so it never sorts above a genuine `likely_eligible` trial, even if its raw score was higher.

### Sort order

```
(decision_priority, -norm_score, -confidence, nct_id)
```

Decision label is the primary sort key, ensuring all `likely_eligible` trials appear before all `needs_review` trials regardless of score. Within each tier, higher score ranks first, then higher confidence, then NCT ID for deterministic tiebreaking.

---

## Design decisions that cut across steps

**All Claude calls use `temperature=0`:** Matching and scoring should be deterministic. Non-zero temperature introduces variance that makes results harder to debug and compare across runs.

**All Claude calls parse JSON:** Structured output avoids natural-language parsing heuristics. Failures (malformed JSON, API errors) are caught per-call and degrade gracefully — the pipeline records the error and continues rather than crashing.

**Two data sources for criteria:** `trials_clean.csv` is used for trial metadata and retrieval. `eligibility_criteria_chunks.csv` is used for criterion matching. The chunks file preserves the original criterion boundaries and ordering, which are important for correct per-criterion assessment. If chunks are missing for a trial, the matcher falls back to splitting the raw text.

**Three LanceDB filters, not post-filtering:** Indication, sex, and age filters run inside the LanceDB ANN search. This is more efficient than over-fetching and filtering afterward — the 2000 candidates per condition are already within the relevant subset. The cost is that filter logic must be expressible as a SQL-style `WHERE` clause, which the three chosen filters satisfy cleanly.
