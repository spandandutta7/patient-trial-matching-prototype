"""
LLM-based indication classifier.

Maps a patient's clinical summary to one of the supported therapeutic area
buckets. The returned string is used as a LanceDB pre-filter so MedCPT only
searches within the relevant condition subset (~1K–16K trials instead of 60K).
"""

import json
import anthropic
import config
from utils.json_utils import strip_json_fences

SUPPORTED_INDICATIONS = [
    "breast cancer",
    "type 2 diabetes",
    "covid-19",
    "anxiety",
    "chronic obstructive pulmonary disease",
    "rheumatoid arthritis",
    "glaucoma",
    "sickle cell anemia",
]

# Maps each supported indication string to the patient-datasets folder name.
# Used so --patient-indication accepts either the folder name or the indication string.
INDICATION_TO_FOLDER = {
    "breast cancer": "breast-cancer",
    "type 2 diabetes": "type2-diabetes",
    "covid-19": "covid19",
    "anxiety": "anxiety",
    "chronic obstructive pulmonary disease": "copd",
    "rheumatoid arthritis": "rheumatoid-arthritis",
    "glaucoma": "glaucoma",
    "sickle cell anemia": "sickle-cell-anemia",
}

_SYSTEM_PROMPT = (
    "You are a clinical classification assistant. Given a patient's clinical summary, "
    "identify which of the following therapeutic areas best matches the patient's primary diagnosis.\n\n"
    "Therapeutic areas:\n"
    + "\n".join(f"- {ind}" for ind in SUPPORTED_INDICATIONS)
    + "\n\nRespond with a JSON object with exactly one key:\n"
    '- "indication": one of the exact strings from the list above (lowercase, exact match)\n\n'
    "Output only valid JSON, no markdown, no extra text."
)


class IndicationClassifier:
    def __init__(self):
        self._client = anthropic.Anthropic()

    def classify(self, patient_text: str) -> str:
        """Return the best-matching indication string for this patient.

        Raises ValueError if the LLM returns an unrecognized indication.
        """
        message = self._client.messages.create(
            model=config.CLAUDE_MODEL_FAST,
            max_tokens=64,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": patient_text}],
            temperature=0,
        )
        raw = strip_json_fences(message.content[0].text.strip())
        result = json.loads(raw)
        indication = result.get("indication", "").lower().strip()
        if indication not in SUPPORTED_INDICATIONS:
            raise ValueError(f"LLM returned unrecognized indication: {indication!r}")
        return indication
