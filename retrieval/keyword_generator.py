"""
LLM-based keyword extraction for retrieval queries.

Mirrors TrialGPT's keyword_generation.py but uses Claude instead of Azure OpenAI.
Given a patient note, Claude extracts ranked medical conditions to use as
search queries for both BM25 and MedCPT retrievers.
"""

import json
import anthropic
import config

_SYSTEM_PROMPT = """\
You are a medical expert assisting with clinical trial matching. \
Given a patient's clinical note, extract the key medical conditions \
and characteristics useful for finding suitable clinical trials.

Output a JSON object with exactly two keys:
- "summary": A concise 1-2 sentence summary of the patient's main medical problems.
- "conditions": A list of up to 32 medical condition/disease terms ranked by importance \
(most important first). These will be used as search queries — prefer specific, \
searchable medical terms over generic descriptions.

Output only valid JSON, no markdown, no extra text.\
"""


class KeywordGenerator:
    def __init__(self, cache_path: str = None):
        self._client = anthropic.Anthropic()
        self._cache_path = str(cache_path or config.KEYWORDS_CACHE_PATH)
        self._cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        try:
            with open(self._cache_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        import os
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def generate(self, patient_id: str, patient_text: str) -> dict:
        """Return {"summary": str, "conditions": [str, ...]} for a patient.

        Results are cached by patient_id to avoid redundant API calls.
        """
        if patient_id in self._cache:
            return self._cache[patient_id]

        message = self._client.messages.create(
            model=config.CLAUDE_MODEL_FAST,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Patient note:\n{patient_text}\n\nJSON output:",
                }
            ],
            temperature=0,
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"summary": patient_text[:200], "conditions": [patient_text[:100]]}

        # Ensure both keys exist
        result.setdefault("summary", "")
        result.setdefault("conditions", [])

        self._cache[patient_id] = result
        self._save_cache()
        return result
