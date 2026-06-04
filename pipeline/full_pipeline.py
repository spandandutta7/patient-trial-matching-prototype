"""
Full patient-trial matching pipeline.

Orchestrates all four steps for a single patient:

  Step 1 — Keyword extraction  (Claude → conditions list)
  Step 2 — Hybrid retrieval    (BM25 + MedCPT via RRF → top-K trial IDs)
  Step 3 — Criterion matching  (Claude per criterion → met / not_met / unknown / missing)
  Step 4 — Aggregation         (Claude → R/E scores → ranked eligibility decision)

Usage (programmatic):
    pipeline = MatchingPipeline()
    pipeline.setup()          # loads indices; call once
    result = pipeline.match_patient("sigir-20141", patient_text, top_k=20)
"""

import time
from typing import Optional
import pandas as pd

import config
from utils.data_loader import load_trials, load_criteria_chunks, get_trial_info
from retrieval.keyword_generator import KeywordGenerator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.medcpt_retriever import MedCPTRetriever
from retrieval.hybrid_fusion import fuse
from matching.criterion_matcher import CriterionMatcher
from aggregation.score_aggregator import (
    compute_matching_score,
    aggregate_with_llm,
    determine_eligibility,
    extract_criteria_breakdown,
)


class MatchingPipeline:
    def __init__(self):
        self._trials_df: Optional[pd.DataFrame] = None
        self._criteria_df: Optional[pd.DataFrame] = None
        self._keyword_gen = KeywordGenerator()
        self._bm25 = BM25Retriever()
        self._medcpt = MedCPTRetriever()
        self._matcher = CriterionMatcher()
        self._ready = False

    # ------------------------------------------------------------------
    # Setup (call once before matching)
    # ------------------------------------------------------------------

    def setup(self, max_trials: Optional[int] = None) -> None:
        """Load datasets and retrieval indices.

        Raises RuntimeError if indices have not been built yet.
        """
        print("Loading trials dataset …")
        self._trials_df = load_trials(nrows=max_trials)
        print(f"Loaded {len(self._trials_df):,} trials.")

        print("Loading criteria chunks dataset …")
        self._criteria_df = load_criteria_chunks()
        print(f"Loaded {len(self._criteria_df):,} criterion chunks.")

        print("Loading BM25 index …")
        self._bm25.build(self._trials_df)

        print("Loading MedCPT/LanceDB index …")
        try:
            self._medcpt.load()
        except RuntimeError:
            print("[WARN] LanceDB index not found — building now (this may take a while).")
            self._medcpt.build(self._trials_df)

        self._ready = True
        print("Pipeline ready.\n")

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
            skip_matching: If True, skip Steps 3-4 (return retrieval results only).
            sex_filter:    e.g. "FEMALE" — passed to LanceDB for pre-filtering.
            age:           Patient age in years — used for LanceDB metadata filtering.

        Returns:
            A dict with keys: patient_id, summary, keywords, results (list of ranked trials).
        """
        if not self._ready:
            raise RuntimeError("Call setup() before match_patient().")

        top_k = top_k or config.TOP_K_RETRIEVAL

        # ---------------------------------------------------------------
        # Step 1: Keyword extraction
        # ---------------------------------------------------------------
        print(f"[{patient_id}] Step 1: Extracting keywords …")
        kw = self._keyword_gen.generate(patient_id, patient_text)
        conditions = kw.get("conditions", [patient_text[:100]])
        summary = kw.get("summary", "")
        print(f"  Summary: {summary}")
        print(f"  Conditions ({len(conditions)}): {conditions[:5]}")

        # ---------------------------------------------------------------
        # Step 2: Hybrid retrieval
        # ---------------------------------------------------------------
        print(f"[{patient_id}] Step 2: Hybrid retrieval (top-{top_k}) …")

        bm25_results = self._bm25.search(conditions, n=config.BM25_TOP_N)
        medcpt_results = self._medcpt.search(
            conditions,
            k=config.MEDCPT_TOP_N,
            sex_filter=sex_filter,
            min_age=age,
            max_age=age,
        )

        fused = fuse(bm25_results, medcpt_results)
        candidate_nctids = [nct_id for nct_id, _ in fused[:top_k]]
        print(f"  Retrieved {len(candidate_nctids)} candidates.")

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
                })
            return {
                "patient_id": patient_id,
                "summary": summary,
                "keywords": conditions,
                "results": results,
            }

        # ---------------------------------------------------------------
        # Steps 3 + 4: Criterion matching + aggregation
        # ---------------------------------------------------------------
        print(f"[{patient_id}] Steps 3+4: Matching and aggregation …")
        criteria_df_subset = load_criteria_chunks(nct_ids=candidate_nctids)

        scored_trials = []
        for i, nct_id in enumerate(candidate_nctids, 1):
            info = get_trial_info(nct_id, self._trials_df)
            if info is None:
                continue

            print(f"  [{i}/{len(candidate_nctids)}] {nct_id}: {info['brief_title'][:60]}")

            # Step 3: Criterion-level matching
            matching = self._matcher.match(patient_text, info, criteria_df_subset)

            # Step 4a: Rule-based matching score
            matching_score = compute_matching_score(matching)

            # Step 4b: LLM aggregation score
            agg = aggregate_with_llm(patient_text, matching, info)
            time.sleep(0.2)

            # Step 4c: Final decision
            decision, norm_score = determine_eligibility(matching_score, agg)

            # Build criteria breakdown
            breakdown = extract_criteria_breakdown(matching, criteria_df_subset, nct_id)

            scored_trials.append({
                "nct_id": nct_id,
                "matching_score": round(matching_score, 3),
                "relevance_score_R": agg.get("relevance_score_R", 0),
                "eligibility_score_E": agg.get("eligibility_score_E", 0),
                "norm_score": norm_score,
                "decision": decision,
                "info": info,
                "agg": agg,
                "breakdown": breakdown,
            })

        # Sort by normalised eligibility score (descending)
        scored_trials.sort(key=lambda x: -x["norm_score"])

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
                "matching_score": t["matching_score"],
                "relevance_score_R": t["relevance_score_R"],
                "eligibility_score_E": t["eligibility_score_E"],
                "met_inclusion_criteria": t["breakdown"]["met_inclusion_criteria"],
                "unmet_inclusion_criteria": t["breakdown"]["unmet_inclusion_criteria"],
                "triggered_exclusion_criteria": t["breakdown"]["triggered_exclusion_criteria"],
                "missing_information": t["breakdown"]["missing_information"],
                "relevance_explanation": t["agg"].get("relevance_explanation", ""),
                "eligibility_explanation": t["agg"].get("eligibility_explanation", ""),
            })

        return {
            "patient_id": patient_id,
            "summary": summary,
            "keywords": conditions,
            "total_candidates": len(candidate_nctids),
            "results": results,
        }
