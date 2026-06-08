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

Confidence demotion:
    likely_eligible is demoted to needs_review when confidence < CONFIDENCE_REVIEW_THRESHOLD.
    confidence is None when no applicable criteria provide a measured value (old schema,
    zero criteria, or all criteria labeled 'not applicable') — the gate is skipped in that case.
    Demoted trials have their norm_score capped below the likely_eligible boundary so they
    always rank after genuine likely_eligible trials.
"""

import json
from typing import Generator, Optional
import anthropic
import config
from utils.json_utils import strip_json_fences

# Lazy-initialised — avoids SDK startup cost when only pure-logic functions are used
_client: Optional[anthropic.Anthropic] = None

# Decision labels ordered from most to least eligible — used for sorting trial results.
DECISION_PRIORITY: dict[str, int] = {"likely_eligible": 0, "needs_review": 1, "not_eligible": 2}

_EPS = 1e-9

# Gap kept between a demoted trial's norm_score and the likely_eligible boundary,
# ensuring demoted trials always sort after genuine likely_eligible ones.
_DEMOTION_SCORE_OFFSET = 0.001

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


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _iter_criteria(matching_results: dict) -> Generator:
    """Yield (group, idx, info) for each structurally valid criterion entry."""
    for group in ("inclusion", "exclusion"):
        for idx, info in matching_results.get(group, {}).items():
            if isinstance(info, list) and len(info) >= 3:
                yield group, idx, info


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
    for group, idx, info in _iter_criteria(matching_results):
        reasoning, sent_ids, label = info[:3]
        lines.append(f"{group} criterion {idx}: {label}")
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

    for group, idx, info in _iter_criteria(matching_results):
        label = info[2]
        if group == "inclusion":
            if label == "included":
                included += 1
            elif label == "not included":
                not_included += 1
            elif label == "not enough information":
                no_info_inc += 1
        else:
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
    user_prompt = _build_agg_user_prompt(patient_text, trial_info, matching_results)
    try:
        message = _get_client().messages.create(
            model=config.CLAUDE_MODEL_SMART,
            max_tokens=1024,
            system=_AGG_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        raw = strip_json_fences(message.content[0].text.strip())
        return json.loads(raw)
    except Exception as e:
        return {
            "relevance_explanation": f"Error: {e}",
            "relevance_score_R": 0.0,
            "eligibility_explanation": "",
            "eligibility_score_E": 0.0,
        }


def compute_confidence(matching_results: dict) -> Optional[float]:
    """Mean confidence across applicable criteria that have a measured confidence value.

    Skips:
    - Criteria labeled 'not applicable' (irrelevance certainty ≠ eligibility certainty)
    - Criteria without a 4th element (pre-redesign 3-element schema)
    - Non-numeric or out-of-range confidence values

    Returns None when no applicable criteria provide a usable confidence measurement.
    None tells determine_eligibility to skip the confidence demotion gate rather than
    treating absent data as low confidence.
    """
    confidences = []
    for group, idx, info in _iter_criteria(matching_results):
        if info[2] == "not applicable" or len(info) < 4:
            continue
        try:
            conf = float(info[3])
        except (TypeError, ValueError):
            continue
        if 0.0 <= conf <= 1.0:
            confidences.append(conf)
    return round(sum(confidences) / len(confidences), 3) if confidences else None


def determine_eligibility(
    matching_score: float, agg_results: dict, confidence: Optional[float]
) -> tuple[str, float]:
    """Map scores to a decision label and a normalised eligibility score [0,1].

    Returns: (decision, normalised_score)

    When confidence is None (no measured values), the demotion gate is skipped.
    When demotion fires, norm_score is capped below the likely_eligible boundary
    so demoted trials always rank after genuine likely_eligible ones.
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

    demoted = False
    if decision == "likely_eligible" and confidence is not None and confidence < config.CONFIDENCE_REVIEW_THRESHOLD:
        decision = "needs_review"
        demoted = True

    # Normalise to [0, 1]: total in [-4, 3] → (total + 4) / 7
    normalised = max(0.0, min(1.0, (total + 4.0) / 7.0))
    if demoted:
        # Cap below the likely_eligible boundary so demoted trials rank after eligible ones
        eligible_boundary = round((config.LIKELY_ELIGIBLE_THRESHOLD + 4.0) / 7.0, 3)
        normalised = min(normalised, eligible_boundary - _DEMOTION_SCORE_OFFSET)

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

    for group, idx, info in _iter_criteria(matching_results):
        label = info[2]
        texts = inc_texts if group == "inclusion" else exc_texts
        text = texts.get(str(idx), f"Criterion {idx}")
        entry = {"index": idx, "criterion": text, "reasoning": info[0]}
        if group == "inclusion":
            if label == "included":
                met_inclusion.append(entry)
            elif label == "not included":
                unmet_inclusion.append(entry)
            elif label == "not enough information":
                missing_info.append(entry)
        else:
            if label == "excluded":
                excluded.append(entry)

    return {
        "met_inclusion_criteria": met_inclusion,
        "unmet_inclusion_criteria": unmet_inclusion,
        "triggered_exclusion_criteria": excluded,
        "missing_information": missing_info,
    }
