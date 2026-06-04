# Pipeline Deep Dive (Steps 1–4)

An explanation of **why the design works**: how the pipeline turns a free-text patient note
into a ranked, explainable eligibility decision, and the technical trade-offs behind each
part. This is the "why," not the "what" or the "how to run."

**Related docs:**
- [README.md](README.md) — architecture, output format, setup, and spec. Start there.
- [CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md) — a file-by-file reference for *what each file is*. Use it to locate things; use this doc to understand the reasoning.

**The four steps:**

| Step | File(s) | Job | Model |
|------|---------|-----|-------|
| 1 — Keyword extraction | `retrieval/keyword_generator.py` | note → ranked search queries | FAST |
| 2 — Hybrid retrieval | `retrieval/bm25_retriever.py`, `medcpt_retriever.py`, `hybrid_fusion.py` | queries → top-K candidate trials | — |
| 3 — Criterion matching | `matching/criterion_matcher.py` | per-criterion eligibility labels | FAST |
| 4 — Aggregation | `aggregation/score_aggregator.py` | labels → decision + score | SMART |

A useful framing for the whole pipeline: **Steps 1–2 optimize for recall** (don't miss a
good trial), while **Steps 3–4 optimize for precision** (correctly judge each candidate).
Retrieval casts a wide net cheaply; matching/aggregation read carefully and expensively.

---

# Steps 1–2: Retrieval

Four files turn a free-text patient note into a ranked shortlist of candidate trials.

```
keyword_generator.generate(note)
        │  → {summary, conditions:[c0, c1, c2, ...]}   (ranked by importance)
        ├───────────────────────────────┐
        ▼                                ▼
bm25.search(conditions)            medcpt.search(conditions, sex, age)
   list-of-lists (lexical)            list-of-lists (semantic, filtered)
        └───────────────┬────────────────┘
                        ▼
               fuse(bm25_results, medcpt_results)   → top-K candidate NCT IDs
```

The unifying idea: Step 1 produces one **ranked** list of condition queries; both
retrievers run that same list and each return *one ranked NCT-ID list per condition*
(a list-of-lists). Because both outputs share that exact shape, fusion can treat them
identically.

## 1. `keyword_generator.py` — note → search queries (Step 1)

A raw clinical vignette is a terrible query: stopwords and narrative phrasing pollute both
lexical and semantic matching. Feeding a whole paragraph to BM25 buries the medical signal
in narrative words ("presents to the ER", "began two days earlier"); feeding it to MedCPT
averages everything into a blurred embedding. So Step 1 uses Claude (FAST model,
`temperature=0`) to distill the note into clean, searchable medical terms.

**The system prompt** ([keyword_generator.py:13-25](retrieval/keyword_generator.py#L13-L25))
asks for a JSON object with exactly two keys:

- **`summary`** — 1–2 sentences, human-readable. Flows straight to the output file.
- **`conditions`** — up to 32 medical terms, **ranked by importance, most important first**.

Three choices baked into the prompt:

- **Two outputs in one call** — `summary` (for humans) and `conditions` (for machines),
  one API call.
- **"ranked by importance"** is the contract the whole retrieval stage depends on. Fusion
  down-weights later conditions, so the order *sets the priorities* for retrieval. The
  model is explicitly told to put the chief complaint first.
- **"specific, searchable medical terms"** nudges toward terms that actually match trial
  text/embeddings ("angina pectoris" over "chest discomfort feeling").

**Caching** ([keyword_generator.py:34-45](retrieval/keyword_generator.py#L34-L45)) —
results are cached by `patient_id` in `cache/keywords_cache.json`. Step 1 is deterministic
(`temperature=0`) and the note doesn't change between runs, so re-running a patient costs
**zero** Step-1 API calls. (Gotcha: editing a note won't invalidate the cache — clear it
manually.)

**Robustness** ([keyword_generator.py:68-79](retrieval/keyword_generator.py#L68-L79)) —
strips ``` ```json ``` fences (models add them even when told not to) and falls back to
note prefixes if the model returns invalid JSON, so the pipeline never crashes here. This
defensive parse pattern recurs in every LLM-calling file.

## 2. `bm25_retriever.py` — lexical retrieval (Step 2, half one)

A classic BM25 keyword index over all ~60K trials. Fast, CPU-only, no neural model, and
excellent at **exact term matches** (the query word literally appears in the trial text).

**Field weighting (the key trick)** ([bm25_retriever.py:32-57](retrieval/bm25_retriever.py#L32-L57))
— BM25 has no native concept of "this field matters more," so the code fakes it by
**repeating tokens**:

```python
tokens  = word_tokenize(title.lower()) * 3        # title × 3
tokens += word_tokenize(cond.lower())  * 2        # each condition × 2  (per condition)
tokens += word_tokenize(text.lower())             # combined text × 1
```

If "diabetes" appears once in the title, it's inserted 3 times into the document's token
list, so BM25's term-frequency component scores a title hit ~3× a body-text hit. The
weights reflect signal density: title (most concentrated) > curated condition list >
broad combined text. Same weighting as TrialGPT.

**`build()`** ([bm25_retriever.py:66-86](retrieval/bm25_retriever.py#L66-L86)) — gets the
tokenized corpus (from `cache/bm25_corpus.json` if present, else tokenizes and caches it),
then constructs `BM25Okapi` fresh in memory. The *tokens* are cached, not the BM25 object.
(Gotcha: an existing cache is reused regardless of the DataFrame passed — delete the cache
to rebuild on different data.)

**`search()`** ([bm25_retriever.py:88-102](retrieval/bm25_retriever.py#L88-L102)) — for
each condition, tokenizes it the same way and returns the top `BM25_TOP_N` (2000) NCT IDs
in rank order, one list per condition. `get_top_n(tokens, self._nctids, n)` scores by
token but returns the *parallel* NCT-id list, so you get trial IDs back directly. Output
is a **list aligned with conditions** — exactly the shape fusion expects.

## 3. `medcpt_retriever.py` — semantic retrieval (Step 2, half two)

Encodes trials and queries into 768-dim vectors with NCBI's biomedical **MedCPT** model and
finds matches by vector similarity. Catches **conceptual** matches BM25 misses (e.g. "MI" ≈
"myocardial infarction" ≈ "heart attack"). This is the heaviest file in the project.

**Two encoders (bi-encoder design)** ([config.py:26-27](config.py#L26-L27)) — MedCPT is two
*separate* co-trained networks:

- **Article encoder** (`ncbi/MedCPT-Article-Encoder`) embeds trials at **build** time
  (max 512 tokens).
- **Query encoder** (`ncbi/MedCPT-Query-Encoder`) embeds conditions at **search** time
  (max 256 tokens).

They were co-trained on PubMed query→article click data so a query vector lands near the
trials that answer it — *even with zero shared words*. That's why semantic retrieval works,
and why the code loads two different models at two different times.

**Embedding = the `[CLS]` token** ([medcpt_retriever.py:62](retrieval/medcpt_retriever.py#L62),
`last_hidden_state[:, 0, :]`) — BERT-style models prepend a special `[CLS]` token whose
final hidden state is trained to summarize the whole input. That single 768-dim vector *is*
the trial's (or query's) embedding. Encoding runs under `torch.no_grad()` in batches
(`MEDCPT_BATCH_SIZE=32`), with `truncation=True` at the max length — which is why the
pre-curated `combined_text_for_retrieval` field exists rather than dumping whole records.

**Storage = LanceDB (not FAISS)** — the main departure from TrialGPT. FAISS stores only
vectors; you get back row indices and *cannot* filter. LanceDB stores the vector **and
metadata columns** (`sex`, `min_age`, `max_age`, `study_type`, `phase`, …) together, which
enables **pre-filtering** by demographics that pure similarity can't enforce. The build step
([medcpt_retriever.py:135-168](retrieval/medcpt_retriever.py#L135-L168)) writes those columns
with an explicit PyArrow schema, including a fixed-size `list_(float32, 768)` vector column
so LanceDB can build an ANN index. Ages are normalized to integer years with **sentinels**
(`-1` for unknown lower bound, `200` for unknown upper bound) so trials without a stated
bound always pass the age filter.

**`build()` is idempotent** ([medcpt_retriever.py:96-104](retrieval/medcpt_retriever.py#L96-L104))
— if the LanceDB `trials` table exists, it loads instead of re-encoding (so `build_index.py`
is safe to re-run). `load()` ([lines 171-182](retrieval/medcpt_retriever.py#L171-L182)) is
the fast path normal runs use; it raises a helpful error if the index is missing. The query
encoder is loaded **lazily** ([lines 80-89](retrieval/medcpt_retriever.py#L80-L89)) so the
two encoders are never in memory at once.

**`search()`** ([medcpt_retriever.py:184-243](retrieval/medcpt_retriever.py#L184-L243)) has
three phases: (1) encode all condition strings with the query encoder → `[CLS]` vectors;
(2) build an optional `WHERE` clause from `--sex` / `--age` — an interval-overlap test
`max_age >= patient_age AND min_age <= patient_age`, plus `sex = X OR sex = 'ALL'`;
(3) run an approximate-nearest-neighbor vector search per condition
([medcpt_retriever.py:234](retrieval/medcpt_retriever.py#L234), `.search(query_vector)`)
returning the top `MEDCPT_TOP_N` (2000) NCT IDs. `prefilter=True` applies the demographic
`WHERE` *during* the search so you still get a full `k` results that satisfy the constraints.
Output is again a **list aligned with conditions**, identical in shape to BM25 — which is
what makes the two fusible.

## 4. `hybrid_fusion.py` — combining the two rankings

BM25 and MedCPT produce scores on **incompatible scales** — BM25 is unbounded TF-IDF
(`3.2`, `14.7`, `28.0`…), MedCPT is a vector distance (`0.41`, `0.83`…). Adding them
directly would let BM25 dominate purely because its numbers are bigger. **Reciprocal Rank
Fusion (RRF)** sidesteps this by discarding raw scores and using only **rank position** —
the common currency both methods can speak.

```python
nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_index + 1))
```

- **`1 / (rank + k)`** converts a rank into a contribution that decreases down the list,
  with `k = RRF_K = 20` as a dampener so the exact #1-vs-#3 ordering doesn't dominate
  (rank 0 → 0.050, rank 2 → 0.045, rank 9 → 0.034, rank 1999 → 0.0005). The dampener
  matters because neither retriever is precise enough at the very top to trust exact slots.
- **`1 / (condition_index + 1)`** weights the patient's **primary** condition fully (×1.0),
  the second ×0.5, the third ×0.33… — so matches on the chief complaint count most. This is
  why Step 1's ranking matters.

**How the two methods actually combine** (the part worth internalizing): there is **no**
separate BM25 total and MedCPT total that get averaged. There is **one** dictionary,
`nctid2score`, and both loops add into the **same key** via `+=`:

```python
nctid2score: dict[str, float] = {}                              # one number per trial
# BM25 loop:   nctid2score[id] = nctid2score.get(id, 0.0) + (1/(rank+k))*cond_w
# MedCPT loop: nctid2score[id] = nctid2score.get(id, 0.0) + (1/(rank+k))*cond_w
```

A trial's final score is just the **running sum of every place it appears** — across both
retrievers and all conditions. The `.get(id, 0.0) + ...` *is* the entire combination
mechanism: when the MedCPT loop runs, it reads whatever BM25 already deposited and stacks on
top. Consequences:

- **Agreement is rewarded.** A trial both methods rank highly collects deposits from both
  and floats up — consensus between independent lexical and semantic evidence is a strong
  signal. (A trial that's BM25's #1 *and* MedCPT's #3 outranks one that's only MedCPT's #1.)
- **Multi-condition relevance is rewarded.** A trial relevant to several of the patient's
  conditions accumulates across condition iterations.
- **The balance is implicit and 1:1.** Both methods are flattened onto the same
  `[0.0005, 0.050]` reciprocal-rank scale *before* being added, so neither can dominate by
  raw magnitude. BM25's #1 and MedCPT's #1 each contribute exactly 0.050. *That* conversion
  (score → rank) is where the "balancing" silently happens — there's no explicit weighting
  step.

> **Note on weights:** `BM25_WEIGHT` / `MEDCPT_WEIGHT` are currently used only as on/off
> gates (`> 0`), **not** as multipliers — so `2` behaves the same as `1`; only `0` (disable
> that retriever) changes the result. True weighted fusion would require multiplying each
> contribution by its weight.

The fused list is sorted descending ([hybrid_fusion.py:56](retrieval/hybrid_fusion.py#L56))
and the top `TOP_K_RETRIEVAL` (20) NCT IDs become the candidates handed to Step 3 (or, with
`--skip-matching`, are returned directly as `retrieval_score`). Note the `n=2000` depth: a
wide net before narrowing to 20 lets consensus and multi-condition signals surface trials no
single list ranked near the top.

---

# Step 3: Criterion Matching — `matching/criterion_matcher.py`

## The core design choice: per-criterion, not whole-trial

Retrieval handed us ~20 candidates, each with free-text eligibility criteria. The naive
approach — "here's a patient and a whole trial, eligible? yes/no" — fails because: a trial
can have 30+ criteria and a single holistic verdict skims and misses specifics; there's no
explainability (which criterion disqualified them?); and you can't compute a calibrated
score from one yes/no. So this file does **per-criterion assessment** — judging each
inclusion/exclusion criterion separately. Everything else follows from that choice.

## The label sets ([criterion_matcher.py:40-41](matching/criterion_matcher.py#L40-L41))

Inclusion and exclusion get **different label vocabularies**, which is medically correct:

- Inclusion: `included` (patient meets it) / `not included` (patient fails it).
- Exclusion: `excluded` (patient triggers it → bad) / `not excluded` (patient is clear → good).

Both share two neutral labels:

- **`not applicable`** — criterion irrelevant to this patient (e.g. a pregnancy criterion
  for a male patient). Correctly kept *out* of the score denominator later.
- **`not enough information`** — the note doesn't say. The honest "I don't know" that
  becomes the output's `missing_information` — clinically valuable because it tells a
  coordinator *what to go ask the patient*.

This four-way scheme (vs binary yes/no) is what lets the system distinguish "patient fails
this" from "we can't tell" — a distinction that drives both the score and `needs_review`.

## Sentence-ID grounding ([criterion_matcher.py:74-77](matching/criterion_matcher.py#L74-L77))

The patient note is split into **numbered sentences** before being shown to the model,
which is then asked to cite the supporting sentence IDs for each judgment. This is an
anti-hallucination / auditability technique: forcing the model to point at evidence makes
it less likely to invent facts and gives a human a traceable path ("criterion 3 was met
because of sentence 5"). Borrowed from TrialGPT.

## The prompt and the closed-world assumption ([criterion_matcher.py:43-85](matching/criterion_matcher.py#L43-L85))

`_build_system_prompt(inc_exc)` fills a parameterized template with the right definition
(`_INCLUSION_DEF` / `_EXCLUSION_DEF`) and label set, and asks for **three elements per
criterion**: reasoning, supporting sentence IDs, and the label.

The single most important instruction ([lines 53-55](matching/criterion_matcher.py#L53-L55)):

> *"If the criterion would clearly appear in a complete patient note but does not, assume
> it is not true for this patient."*

This is a deliberate **closed-world assumption**, balanced against "use *not enough
information* sparingly" ([line 58](matching/criterion_matcher.py#L58)). Without it, almost
everything would be "not enough information" (real notes never state every criterion
explicitly), and every trial would land in `needs_review` — useless. With it, the model
makes reasonable clinical inferences (a chest-pain note that never mentions pregnancy
implies the patient isn't in a pregnancy sub-cohort). The tension between these two
instructions is the calibration knob for how aggressive the matcher is.

## Reading criteria + context ([criterion_matcher.py:88-107](matching/criterion_matcher.py#L88-L107))

- **`_format_trial_with_criteria`** renders the trial's *purpose* (title, target diseases,
  interventions, summary) alongside the numbered criteria — context the model needs to
  judge applicability.
- **`_parse_criteria_from_chunks`** pulls this trial's criteria from the chunks DataFrame,
  filtered by `criterion_type` and **sorted by `criterion_index`**. The sort matters: the
  prompt index must align with `criterion_index` so Step 4 can map labels back to the right
  criterion text.

## `CriterionMatcher.match()` — the core ([criterion_matcher.py:114-186](matching/criterion_matcher.py#L114-L186))

Runs once per candidate trial. Key behaviors:

- **Two calls per trial** ([line 139](matching/criterion_matcher.py#L139)) — inclusion and
  exclusion are handled in separate LLM calls, because each needs a different definition,
  label set, and framing. Cost: 2 calls × ~20 trials = ~40 calls — the bulk of the
  pipeline's API spend, hence the **FAST/cheap** model here.
- **Fallback** ([lines 143-150](matching/criterion_matcher.py#L143-L150)) — if the chunks
  CSV has no rows for a trial, it degrades to splitting the raw criteria text from
  `trials_clean.csv` (filtering blanks, <5-char fragments, and "Inclusion Criteria:"
  headers). The trial still gets assessed even with missing chunk data.
- **Cost cap** ([line 157](matching/criterion_matcher.py#L157)) — `criteria[:max_criteria]`
  (`MAX_CRITERIA_PER_TRIAL=50`). A real tradeoff: a disqualifying exclusion at position 60
  would be silently missed.
- **Deterministic** — FAST model, `temperature=0`, `max_tokens=2048`, same fence-strip +
  `json.loads` pattern.
- **Fail soft** ([lines 179-181](matching/criterion_matcher.py#L179-L181)) — any exception
  or parse failure stores `{"error": str(e)}` instead of crashing; downstream guards skip
  malformed entries, so one bad trial never sinks the patient run.

**Output schema:**

```python
{
  "inclusion": {"0": ["reasoning", [sent_ids], "included"], "1": [...], ...},
  "exclusion": {"0": ["reasoning", [sent_ids], "not excluded"], ...}
}
```

## Relation to the final goal

Step 3 produces the *raw evidence* — per-criterion labels + reasoning + grounding. It
doesn't decide eligibility; it produces the structured facts Step 4 scores, and that
populate the output's four criteria lists. **All of the final output's explainability is
created here.**

---

# Step 4: Aggregation — `aggregation/score_aggregator.py`

## The two-stage design and why

Step 4 turns per-criterion labels into one decision (`likely_eligible` / `not_eligible` /
`needs_review`) and one rankable score, using **two independent scoring stages that are
summed** — a deliberate hedge:

1. **Rule-based** (`compute_matching_score`) — deterministic, transparent, derived purely
   from counting labels. Trustworthy and cheap, but rigid.
2. **LLM-based** (`aggregate_with_llm`) — a holistic relevance + eligibility judgment from
   the SMART model. Captures nuance the counting rules can't, but is a black box.

Summing them means neither dominates: the rules anchor the score in hard criterion facts;
the LLM adds clinical judgment. Mirrors TrialGPT's matching-score + aggregation-score.

## Stage 1 — `compute_matching_score()` ([score_aggregator.py:80-112](aggregation/score_aggregator.py#L80-L112))

```python
score = included / (included + not_included + no_info_inc + ε)
if not_included > 0: score -= 1.0
if excluded > 0:     score -= 1.0
```

- **Base term:** the fraction of *decided* inclusion criteria the patient meets. Only
  inclusion labels are counted; `not applicable` is excluded from the denominator (irrelevant
  criteria shouldn't dilute the score). `ε = 1e-9` prevents divide-by-zero.
- **Penalty 1:** *any* unmet inclusion → −1.0 (flat, not proportional — failing even one
  inclusion is decisive).
- **Penalty 2:** *any* triggered exclusion → −1.0.

Range **[−2, 1]**. Penalties stack, so a trial that both fails an inclusion and triggers an
exclusion sinks hardest. Malformed Step-3 entries are skipped by `isinstance/len==3` guards.

## Stage 2 — `aggregate_with_llm()` ([score_aggregator.py:115-139](aggregation/score_aggregator.py#L115-L139))

Calls the **SMART model** (sonnet) — the one place worth the more capable/expensive model,
since it's one call per trial and most affects ranking. The prompt
([lines 31-46](aggregation/score_aggregator.py#L31-L46)) asks for two numbers:

- **Relevance R (0–100):** topical relatedness, independent of eligibility.
- **Eligibility E (−R … R):** eligibility, **bounded by** relevance.

The **E-bounded-by-R design** is the clever bit: it couples eligibility to relevance so an
irrelevant trial (low R) can't accidentally produce a strongly-eligible signal — eligibility
is only as confident as relevance allows. `_build_agg_user_prompt` + `_format_predictions`
([lines 49-77](aggregation/score_aggregator.py#L49-L77)) feed the model *both* the raw note
*and* the Step-3 per-criterion analysis — so it reviews and synthesizes Step 3's work rather
than redoing it.

## Combining + deciding — `determine_eligibility()` ([score_aggregator.py:142-167](aggregation/score_aggregator.py#L142-L167))

```python
agg_score = (R + E) / 100.0          # range [-2, 2]
total = matching_score + agg_score   # range [-4, 3]  ← the actual rule+LLM fusion
```

Decision thresholds (configurable in `config.py`):

```
total >= 1.5   → likely_eligible
total <= -0.5  → not_eligible
otherwise      → needs_review
```

The asymmetry is intentional and clinically conservative: the bar to *claim eligible* (1.5)
is high, and the middle band routes to `needs_review` (a human should look) rather than
auto-rejecting. The system triages and surfaces uncertainty — it does not make autonomous
accept/reject calls.

**Normalization** ([line 166](aggregation/score_aggregator.py#L166)): `(total + 4) / 7`
clamped to [0, 1], mapping [−4, 3] to a clean `eligibility_score` for display and ranking.
(The decision uses the raw `total`; the reported score is normalized.)

## Mapping back to output — `extract_criteria_breakdown()` ([score_aggregator.py:170-219](aggregation/score_aggregator.py#L170-L219))

The scores answer "how eligible?"; this answers "*why?*". It joins Step-3 labels back to
their **criterion text** (via the chunks DataFrame, keyed by `criterion_index`) and buckets
them:

```
included              → met_inclusion_criteria
not included          → unmet_inclusion_criteria
excluded              → triggered_exclusion_criteria
not enough information → missing_information
```

Each entry is `{index, criterion, reasoning}` — the human-readable explainability that
makes the output actionable. `not applicable` and `not excluded` are dropped (non-events).

## Relation to the final goal

Step 4 is where everything converges into the deliverable: the `eligibility_decision`, the
rankable `eligibility_score` (which sets each trial's `rank` after sorting in
[full_pipeline.py:197](pipeline/full_pipeline.py#L197)), the component scores
(`matching_score`, `relevance_score_R`, `eligibility_score_E`) for transparency, and the
four criteria-breakdown lists — exactly what the project brief asked for.

---

# Cross-cutting design themes

These principles span all four steps and explain the system's overall shape:

- **Recall first, then precision.** Cheap wide-net retrieval (Steps 1–2) feeds careful,
  expensive judgment (Steps 3–4). Wrong-but-fast retrieval is acceptable because the LLM
  re-checks; the costly reasoning is reserved for the ~20 survivors.
- **Decompose, then recompose.** Break the hard "is this patient eligible?" question into
  many small, well-scoped LLM judgments (Step 3), then reassemble with deterministic rules +
  one holistic LLM pass (Step 4). Small focused prompts are more accurate and debuggable.
- **Two models by role.** FAST/cheap for high-volume work (keyword gen, per-criterion
  matching); SMART for the single high-leverage aggregation judgment. Cost is concentrated
  where it matters.
- **Rank as a common currency.** Fusion combines incompatible score scales by converting
  both to reciprocal rank — an elegant way to balance methods 1:1 without hand-tuned weights.
- **Explainability is a first-class output.** Reasoning, sentence grounding, and the four
  criteria buckets exist so a human can audit *why*, not just *what*.
- **Fail soft everywhere.** Malformed LLM output becomes `{"error":...}` and is skipped by
  guards; invalid keyword JSON falls back to note prefixes. One bad trial or criterion never
  sinks a run.
- **Honest uncertainty.** The `not enough information` label and the `needs_review` band let
  the system say "I don't know — a human should check," the safe behavior for clinical triage.
- **Cache the expensive artifact.** Keywords, BM25 tokens, and MedCPT embeddings are each
  cached and reused on existence (with the shared caveat that an existing cache ignores new
  input data — delete to rebuild).
