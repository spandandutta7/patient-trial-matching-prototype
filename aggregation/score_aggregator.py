"""
Score aggregation and final eligibility determination.

Two-stage scoring mirrors TrialGPT's approach:

Stage 1 — Rule-based matching score (from criterion predictions):
    score = included / (included + not_included + no_info + ε)
    score -= 1   if any not_included
    score -= 1   if any excluded
    Range: [-2, 1]

Stage 2 — LLM aggregation (Claude rates relevance R and eligibility E):
    agg_score = (R + E) / 100   Range: [-2, 2]

Final score = matching_score + agg_score   Range: [-4, 3]

Eligibility decision thresholds (configurable in config.py):
    likely_eligible   : score >= LIKELY_ELIGIBLE_THRESHOLD   (default 1.5)
    not_eligible      : score <= NOT_ELIGIBLE_THRESHOLD       (default -0.5)
    needs_review      : otherwise
"""

import json
import time
from typing import Optional
import anthropic
import config

_EPS = 1e-9

_AGG_SYSTEM = """\
You are a clinical trial eligibility expert. You will be given a patient note, \
a clinical trial, and per-criterion eligibility predictions. Output two scores:

1. Relevance score R (0–100): Overall relevance between patient and trial.
   R=0 means totally irrelevant; R=100 means perfectly matched conditions.

2. Eligibility score E (–R to R): Patient's eligibility.
   E=R means fully eligible (all inclusion met, no exclusion triggered).
   E=–R means fully ineligible.
   E=0 means neutral / insufficient information.

Output ONLY a JSON dict:
{"relevance_explanation": str, "relevance_score_R": float, \
"eligibility_explanation": str, "eligibility_score_E": float}
"""


def _build_agg_user_prompt(patient_text: str, trial_info: dict, matching_results: dict) -> str:
    trial_str = (
        f"Title: {trial_info.get('brief_title', '')}\n"
        f"Target conditions: {', '.join(trial_info.get('diseases_list', []))}\n"
        f"Summary: {trial_info.get('brief_summary', '')}"
    )
    pred_str = _format_predictions(matching_results, trial_info)
    return (
        f"Patient note:\n{patient_text}\n\n"
        f"Clinical trial:\n{trial_str}\n\n"
        f"Criterion-level eligibility predictions:\n{pred_str}\n\n"
        "Plain JSON output:"
    )


def _format_predictions(matching_results: dict, trial_info: dict) -> str:
    """Convert criterion predictions to a readable string for the aggregation prompt."""
    lines = []
    for inc_exc in ("inclusion", "exclusion"):
        preds = matching_results.get(inc_exc, {})
        for idx, info in preds.items():
            if not isinstance(info, list) or len(info) != 3:
                continue
            reasoning, sent_ids, label = info
            lines.append(f"{inc_exc} criterion {idx}: {label}")
            lines.append(f"  Reasoning: {reasoning}")
            if sent_ids:
                lines.append(f"  Evidence sentences: {sent_ids}")
    return "\n".join(lines) if lines else "No criterion predictions available."


def compute_matching_score(matching_results: dict) -> float:
    """Rule-based score from criterion predictions (Stage 1)."""
    included = 0
    not_included = 0
    no_info_inc = 0
    excluded = 0

    inc_preds = matching_results.get("inclusion", {})
    for info in inc_preds.values():
        if not isinstance(info, list) or len(info) != 3:
            continue
        label = info[2]
        if label == "included":
            included += 1
        elif label == "not included":
            not_included += 1
        elif label == "not enough information":
            no_info_inc += 1

    exc_preds = matching_results.get("exclusion", {})
    for info in exc_preds.values():
        if not isinstance(info, list) or len(info) != 3:
            continue
        label = info[2]
        if label == "excluded":
            excluded += 1

    score = included / (included + not_included + no_info_inc + _EPS)
    if not_included > 0:
        score -= 1.0
    if excluded > 0:
        score -= 1.0
    return score


def aggregate_with_llm(
    patient_text: str, matching_results: dict, trial_info: dict
) -> dict:
    """Call Claude for trial-level relevance + eligibility scores (Stage 2)."""
    client = anthropic.Anthropic()
    user_prompt = _build_agg_user_prompt(patient_text, trial_info, matching_results)
    try:
        message = client.messages.create(
            model=config.CLAUDE_MODEL_SMART,
            max_tokens=1024,
            system=_AGG_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "relevance_explanation": f"Error: {e}",
            "relevance_score_R": 0.0,
            "eligibility_explanation": "",
            "eligibility_score_E": 0.0,
        }


def determine_eligibility(
    matching_score: float, agg_results: dict
) -> tuple[str, float]:
    """Map scores to a decision label and a normalised eligibility score [0,1].

    Returns: (decision, normalised_score)
    """
    try:
        R = float(agg_results.get("relevance_score_R", 0))
        E = float(agg_results.get("eligibility_score_E", 0))
        agg_score = (R + E) / 100.0
    except (TypeError, ValueError):
        agg_score = 0.0

    total = matching_score + agg_score

    if total >= config.LIKELY_ELIGIBLE_THRESHOLD:
        decision = "likely_eligible"
    elif total <= config.NOT_ELIGIBLE_THRESHOLD:
        decision = "not_eligible"
    else:
        decision = "needs_review"

    # Normalise to [0, 1]: total in [-4, 3] → (total + 4) / 7
    normalised = max(0.0, min(1.0, (total + 4.0) / 7.0))
    return decision, round(normalised, 3)


def extract_criteria_breakdown(matching_results: dict, criteria_chunks_df, nct_id: str) -> dict:
    """Return lists of met/unmet/excluded/missing criteria with their texts."""
    met_inclusion = []
    unmet_inclusion = []
    excluded = []
    missing_info = []

    # Build index: criterion_index → criterion_text for this trial
    if criteria_chunks_df is not None:
        inc_chunks = criteria_chunks_df[
            (criteria_chunks_df["nct_id"] == nct_id)
            & (criteria_chunks_df["criterion_type"] == "inclusion")
        ].sort_values("criterion_index")
        exc_chunks = criteria_chunks_df[
            (criteria_chunks_df["nct_id"] == nct_id)
            & (criteria_chunks_df["criterion_type"] == "exclusion")
        ].sort_values("criterion_index")
        inc_texts = dict(zip(inc_chunks["criterion_index"].astype(str), inc_chunks["criterion_text"]))
        exc_texts = dict(zip(exc_chunks["criterion_index"].astype(str), exc_chunks["criterion_text"]))
    else:
        inc_texts, exc_texts = {}, {}

    for idx, info in matching_results.get("inclusion", {}).items():
        if not isinstance(info, list) or len(info) != 3:
            continue
        label = info[2]
        text = inc_texts.get(str(idx), f"Criterion {idx}")
        entry = {"index": idx, "criterion": text, "reasoning": info[0]}
        if label == "included":
            met_inclusion.append(entry)
        elif label == "not included":
            unmet_inclusion.append(entry)
        elif label == "not enough information":
            missing_info.append(entry)

    for idx, info in matching_results.get("exclusion", {}).items():
        if not isinstance(info, list) or len(info) != 3:
            continue
        label = info[2]
        text = exc_texts.get(str(idx), f"Criterion {idx}")
        entry = {"index": idx, "criterion": text, "reasoning": info[0]}
        if label == "excluded":
            excluded.append(entry)

    return {
        "met_inclusion_criteria": met_inclusion,
        "unmet_inclusion_criteria": unmet_inclusion,
        "triggered_exclusion_criteria": excluded,
        "missing_information": missing_info,
    }
