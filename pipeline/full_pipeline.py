"""
Full patient-trial matching pipeline.

Orchestrates all steps for a single patient:

  Step 1 — Indication classification  (Claude → one of 8 therapeutic areas)
  Step 2 — Keyword extraction          (Claude → conditions list)
  Step 3 — Semantic retrieval          (MedCPT + LanceDB pre-filter → top-K trial IDs)
  Step 4 — Criterion matching          (Claude per criterion → met / not_met / unknown / missing)
  Step 5 — Aggregation                 (Claude → R/E scores → ranked eligibility decision)

Usage (programmatic):
    pipeline = MatchingPipeline()
    pipeline.setup()          # loads index; call once
    result = pipeline.match_patient("breast-cancer", patient_text, top_k=10)
"""

from typing import Optional
import pandas as pd

import config
from utils.data_loader import load_trials, load_criteria_chunks, get_trial_info
from retrieval.indication_classifier import IndicationClassifier
from retrieval.keyword_generator import KeywordGenerator
from retrieval.medcpt_retriever import MedCPTRetriever
from matching.criterion_matcher import CriterionMatcher
from aggregation.score_aggregator import (
    DECISION_PRIORITY,
    compute_matching_score,
    compute_confidence,
    aggregate_with_llm,
    determine_eligibility,
    extract_criteria_breakdown,
)


class MatchingPipeline:
    def __init__(self):
        self._trials_df: Optional[pd.DataFrame] = None
        self._criteria_df: Optional[pd.DataFrame] = None
        self._classifier = IndicationClassifier()
        self._keyword_gen = KeywordGenerator()
        self._medcpt = MedCPTRetriever()
        self._matcher = CriterionMatcher()
        self._ready = False

    # ------------------------------------------------------------------
    # Setup (call once before matching)
    # ------------------------------------------------------------------

    def setup(self, max_trials: Optional[int] = None) -> None:
        """Load datasets and retrieval index.

        Raises RuntimeError if the MedCPT index has not been built yet.
        """
        print("Setting up pipeline ...")
        self._trials_df = load_trials(nrows=max_trials)
        self._criteria_df = load_criteria_chunks()
        print(f"  Datasets  : {len(self._trials_df):,} trials  |  {len(self._criteria_df):,} criteria chunks")

        try:
            self._medcpt.load()
        except RuntimeError:
            print("  [WARN] LanceDB index not found — building now (this may take a while) ...")
            self._medcpt.build(self._trials_df)

        self._ready = True
        print("Ready.\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def match_patient(
        self,
        patient_id: str,
        patient_text: str,
        top_k: int = None,
        skip_matching: bool = False,
        sex_filter: str = None,
        age: Optional[int] = None,
    ) -> dict:
        """Run the full pipeline for one patient.

        Args:
            patient_id:    Unique patient identifier (used for keyword caching).
            patient_text:  Raw clinical note text.
            top_k:         Number of top trials to retrieve and evaluate.
            skip_matching: If True, skip Steps 4-5 (return retrieval results only).
            sex_filter:    e.g. "FEMALE" — passed to LanceDB for pre-filtering.
            age:           Patient age in years — used for LanceDB metadata filtering.

        Returns:
            A dict with keys: patient_id, indication, summary, keywords, results.
        """
        if not self._ready:
            raise RuntimeError("Call setup() before match_patient().")

        top_k = top_k or config.TOP_K_RETRIEVAL

        # ---------------------------------------------------------------
        # Step 1: Indication classification
        # ---------------------------------------------------------------
        print(f"[{patient_id}]  Step 1  Classifying indication ...")
        try:
            indication = self._classifier.classify(patient_text)
        except Exception as e:
            print(f"             [WARN] Indication classification failed ({e}); searching all conditions.")
            indication = None
        print(f"             Indication: {indication or '(none — searching all)'}")

        # ---------------------------------------------------------------
        # Step 2: Keyword extraction
        # ---------------------------------------------------------------
        print(f"[{patient_id}]  Step 2  Extracting keywords ...")
        kw = self._keyword_gen.generate(patient_id, patient_text)
        conditions = kw.get("conditions", [patient_text[:100]])[:config.MAX_CONDITIONS_PER_PATIENT]
        summary = kw.get("summary", "")

        # ---------------------------------------------------------------
        # Step 3: Semantic retrieval with indication pre-filter
        # ---------------------------------------------------------------
        print(f"[{patient_id}]  Step 3  Retrieving top-{top_k} trials ({len(conditions)} queries) ...")

        medcpt_results = self._medcpt.search(
            conditions,
            k=config.MEDCPT_TOP_N,
            sex_filter=sex_filter,
            min_age=age,
            max_age=age,
            indication=indication,
        )

        # RRF over per-condition MedCPT results; primary condition gets full weight
        nctid2score: dict[str, float] = {}
        for condition_idx, nct_ids in enumerate(medcpt_results):
            condition_weight = 1.0 / (condition_idx + 1)
            for rank, nct_id in enumerate(nct_ids):
                nctid2score[nct_id] = nctid2score.get(nct_id, 0.0) + (
                    (1.0 / (rank + config.RRF_K)) * condition_weight
                )
        fused = sorted(nctid2score.items(), key=lambda x: -x[1])
        candidate_nctids = [nct_id for nct_id, _ in fused[:top_k]]
        print(f"             {len(candidate_nctids)} candidates found.")

        if skip_matching:
            results = []
            for rank, nct_id in enumerate(candidate_nctids, 1):
                info = get_trial_info(nct_id, self._trials_df) or {}
                results.append({
                    "rank": rank,
                    "nct_id": nct_id,
                    "title": info.get("brief_title", ""),
                    "conditions": info.get("conditions", ""),
                    "clinicaltrials_url": info.get("clinicaltrials_url", ""),
                    "retrieval_score": fused[rank - 1][1],
                    "confidence": None,
                })
            return {
                "patient_id": patient_id,
                "indication": indication,
                "summary": summary,
                "keywords": conditions,
                "results": results,
            }

        # ---------------------------------------------------------------
        # Steps 4 + 5: Criterion matching + aggregation
        # ---------------------------------------------------------------
        print(f"[{patient_id}]  Steps 4-5  Assessing criteria + scoring ...")
        criteria_df_subset = self._criteria_df[
            self._criteria_df["nct_id"].isin(set(candidate_nctids))
        ]

        scored_trials = []
        for i, nct_id in enumerate(candidate_nctids, 1):
            info = get_trial_info(nct_id, self._trials_df)
            if info is None:
                continue

            print(f"             {i}/{len(candidate_nctids)}  {nct_id}")

            # Step 4: Criterion-level matching
            matching = self._matcher.match(patient_text, info, criteria_df_subset)
            matching_errors = [g for g in ("inclusion", "exclusion") if "error" in matching.get(g, {})]
            if matching_errors:
                print(f"             [WARN] Matching errors in {matching_errors} for {nct_id}")

            # Step 5a: Rule-based matching score
            matching_score = compute_matching_score(matching)

            # Step 5b: Confidence from per-criterion assessments
            confidence = compute_confidence(matching)

            # Step 5c: LLM aggregation score
            agg = aggregate_with_llm(patient_text, matching, info)

            # Step 5d: Final decision
            decision, norm_score = determine_eligibility(matching_score, agg, confidence)

            # Build criteria breakdown
            breakdown = extract_criteria_breakdown(matching, criteria_df_subset, nct_id)

            scored_trials.append({
                "nct_id": nct_id,
                "matching_score": round(matching_score, 3),
                "relevance_score_R": agg.get("relevance_score_R", 0),
                "eligibility_score_E": agg.get("eligibility_score_E", 0),
                "norm_score": norm_score,
                "confidence": confidence,
                "decision": decision,
                "matching_errors": matching_errors,
                "info": info,
                "agg": agg,
                "breakdown": breakdown,
            })

        # Sort: decision label first (eligible > review > not eligible), then score,
        # then confidence (None treated as 0), then nct_id for deterministic tiebreaking
        scored_trials.sort(key=lambda x: (
            DECISION_PRIORITY.get(x["decision"], 1),
            -x["norm_score"],
            -(x["confidence"] if x["confidence"] is not None else 0.0),
            x["nct_id"],
        ))

        results = []
        for rank, t in enumerate(scored_trials, 1):
            info = t["info"]
            results.append({
                "rank": rank,
                "nct_id": t["nct_id"],
                "title": info.get("brief_title", ""),
                "conditions": info.get("conditions", ""),
                "study_type": info.get("study_type", ""),
                "phase": info.get("phase", ""),
                "sex": info.get("sex", ""),
                "min_age": info.get("minimum_age", ""),
                "max_age": info.get("maximum_age", ""),
                "clinicaltrials_url": info.get("clinicaltrials_url", ""),
                "eligibility_decision": t["decision"],
                "eligibility_score": t["norm_score"],
                "confidence": t["confidence"],
                "matching_score": t["matching_score"],
                "eligibility_score_E": t["eligibility_score_E"],
                "met_inclusion_criteria": t["breakdown"]["met_inclusion_criteria"],
                "unmet_inclusion_criteria": t["breakdown"]["unmet_inclusion_criteria"],
                "triggered_exclusion_criteria": t["breakdown"]["triggered_exclusion_criteria"],
                "missing_information": t["breakdown"]["missing_information"],
                "eligibility_explanation": t["agg"].get("eligibility_explanation", ""),
                "_debug": {
                    "relevance_score_R": t["relevance_score_R"],
                    "relevance_explanation": t["agg"].get("relevance_explanation", ""),
                    "matching_errors": t["matching_errors"],
                },
            })

        return {
            "patient_id": patient_id,
            "indication": indication,
            "summary": summary,
            "keywords": conditions,
            "total_candidates": len(candidate_nctids),
            "results": results,
        }
