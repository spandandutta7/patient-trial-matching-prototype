"""
Criterion-level eligibility assessment using Claude.

For each candidate trial, the matcher:
1. Loads the trial's individual inclusion and exclusion criteria from
   eligibility_criteria_chunks.csv.
2. Calls Claude once for inclusion criteria and once for exclusion criteria.
3. Returns a structured assessment per criterion: met / not_met / unknown /
   missing_info (mapped to TrialGPT's label set internally).

Output schema per trial:
  {
    "inclusion": {
      "0": ["reasoning", [sentence_ids], "included|not included|not applicable|not enough information", 0.9],
      ...
    },
    "exclusion": {
      "0": ["reasoning", [sentence_ids], "excluded|not excluded|not applicable|not enough information", 0.85],
      ...
    }
  }
"""

import json
import time
from nltk.tokenize import sent_tokenize
import anthropic
import config
import nltk
from utils.json_utils import strip_json_fences

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

_INCLUSION_LABELS = '{"not applicable", "not enough information", "included", "not included"}'
_EXCLUSION_LABELS = '{"not applicable", "not enough information", "excluded", "not excluded"}'

_SYSTEM_TEMPLATE = """\
You are a clinical trial eligibility assessor. Your task is to compare a \
patient note against the {inc_exc} criteria of a clinical trial to determine \
the patient's eligibility at the criterion level.

{criterion_definition}

Check each {inc_exc} criterion one by one and output four elements per criterion:
  Element 1 — Reasoning: Judge whether the criterion is not applicable, then look \
for direct evidence in the patient note. If no direct evidence, infer from context. \
If the criterion would clearly appear in a complete patient note but does not, \
assume it is not true for this patient.
  Element 2 — Relevant sentence IDs from the patient note (empty list if none).
  Element 3 — Eligibility label chosen from {labels}. \
Use "not applicable" only for criteria irrelevant to this patient. \
Use "not enough information" sparingly.
  Element 4 — Confidence in [0,1]: score ≥0.8 only when direct explicit evidence \
in the patient note supports your label; use 0.5 for inferred or uncertain assessments.

Output ONLY a JSON dict:
dict{{str(criterion_index): list[str(reasoning), list[int(sentence_id)], str(label), float(confidence)]}}
"""

_INCLUSION_DEF = (
    "Inclusion criteria are factors that allow someone to participate. "
    "They are based on age, gender, disease type/stage, prior treatments, and other conditions."
)
_EXCLUSION_DEF = (
    "Exclusion criteria are factors that disqualify someone from participating. "
    "They are based on age, gender, disease type/stage, prior treatments, and other conditions."
)


def _format_patient_with_sentence_ids(patient_text: str) -> str:
    """Add numeric sentence IDs so the LLM can reference specific sentences."""
    sentences = sent_tokenize(patient_text)
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def _build_system_prompt(inc_exc: str) -> str:
    defn = _INCLUSION_DEF if inc_exc == "inclusion" else _EXCLUSION_DEF
    labels = _INCLUSION_LABELS if inc_exc == "inclusion" else _EXCLUSION_LABELS
    return _SYSTEM_TEMPLATE.format(
        inc_exc=inc_exc, criterion_definition=defn, labels=labels
    )


def _format_trial_with_criteria(trial_info: dict, inc_exc: str, criteria: list[str]) -> str:
    lines = [
        f"Title: {trial_info.get('brief_title', '')}",
        f"Target diseases: {', '.join(trial_info.get('diseases_list', []))}",
        f"Interventions: {', '.join(trial_info.get('drugs_list', []))}",
        f"Summary: {trial_info.get('brief_summary', '')}",
        f"\n{inc_exc.capitalize()} criteria:",
    ]
    for i, criterion in enumerate(criteria):
        lines.append(f"{i}. {criterion}")
    return "\n".join(lines)


def _parse_criteria_from_chunks(criteria_chunks_df, nct_id: str, inc_exc: str) -> list[str]:
    """Return ordered list of criterion texts for a trial from the chunks DataFrame."""
    mask = (criteria_chunks_df["nct_id"] == nct_id) & (
        criteria_chunks_df["criterion_type"] == inc_exc
    )
    sub = criteria_chunks_df[mask].sort_values("criterion_index")
    return sub["criterion_text"].tolist()


class CriterionMatcher:
    def __init__(self):
        self._client = anthropic.Anthropic()

    def match(
        self,
        patient_text: str,
        trial_info: dict,
        criteria_chunks_df,
        max_criteria: int = None,
    ) -> dict:
        """Assess all inclusion and exclusion criteria for one patient-trial pair.

        Args:
            patient_text: Raw patient clinical note.
            trial_info: Dict with trial metadata (from data_loader.get_trial_info).
            criteria_chunks_df: Filtered DataFrame from eligibility_criteria_chunks.csv.
            max_criteria: Cap per inc/exc group to control LLM cost.

        Returns:
            {"inclusion": {idx: [reasoning, [sent_ids], label, confidence], ...},
             "exclusion": {idx: [reasoning, [sent_ids], label, confidence], ...}}
        """
        max_criteria = max_criteria or config.MAX_CRITERIA_PER_TRIAL
        nct_id = trial_info["nct_id"]
        patient_with_ids = _format_patient_with_sentence_ids(patient_text)

        results = {}

        for inc_exc in ("inclusion", "exclusion"):
            criteria = _parse_criteria_from_chunks(criteria_chunks_df, nct_id, inc_exc)

            # Fall back to raw criteria text if chunks are missing
            if not criteria:
                raw = trial_info.get(f"{inc_exc}_criteria", "")
                criteria = [
                    line.strip()
                    for line in raw.split("\n")
                    if line.strip() and len(line.strip()) > 5
                    and "criteria" not in line.lower()[:20]
                ]

            if not criteria:
                results[inc_exc] = {}
                continue

            # Limit criteria count to control cost - important silent tradeoff
            criteria = criteria[:max_criteria]

            system_prompt = _build_system_prompt(inc_exc)
            trial_str = _format_trial_with_criteria(trial_info, inc_exc, criteria)
            user_prompt = (
                f"Patient note (sentence IDs shown):\n{patient_with_ids}\n\n"
                f"Clinical trial:\n{trial_str}\n\n"
                "Plain JSON output:"
            )

            try:
                message = self._client.messages.create(
                    model=config.CLAUDE_MODEL_FAST,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0,
                )
                raw = strip_json_fences(message.content[0].text.strip())
                results[inc_exc] = json.loads(raw)
            except Exception as e:
                # On failure, record raw text so pipeline can continue
                results[inc_exc] = {"error": str(e)}

            
        return results
