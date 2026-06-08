"""
LLM-based keyword extraction for retrieval queries.

Mirrors TrialGPT's keyword_generation.py but uses Claude instead of Azure OpenAI.
Given a patient note, Claude extracts ranked medical conditions to use as
search queries for MedCPT semantic retrieval.
"""

import json
import logging
import anthropic
import config
from utils.json_utils import strip_json_fences

logger = logging.getLogger(__name__)

# Bump this whenever the prompt schema changes so stale cache entries are invalidated.
_CACHE_SCHEMA_VERSION = "v2"

_SYSTEM_PROMPT = f"""\
You are a medical expert assisting with clinical trial matching.
Given a patient's clinical summary note, extract the key medical conditions,
codes, and characteristics useful for finding suitable clinical trials.

Output a JSON object with exactly two keys:
- "summary": A concise 1-2 sentence summary of the patient's main medical problems.
- "conditions": A list of up to {config.MAX_CONDITIONS_PER_PATIENT} objects ranked by importance \
(most important first), each with:
    - "term": the specific, searchable medical condition or characteristic
    - "status": one of "active", "historical", "negated", "hypothetical", "family_history"
        - "active": currently confirmed for this patient
        - "historical": past or resolved (e.g. prior treatments, prior events)
        - "negated": explicitly denied ("no prior chemo", "denies X")
        - "hypothetical": ruled-out, suspected, uncertain, or conditional \
("rule out X", "if X worsens", "possible X", "concerning for X")
        - "family_history": condition belonging to a family member, not the patient

CRITICAL: biomarker/receptor status terms like "HER2-negative", "triple-negative", \
"ER-positive", and qualified diagnoses like "diabetes without complications" are \
ALWAYS "active" — they define the patient's disease subtype, not a negated finding.

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
        Cache entries from older schema versions are automatically re-fetched.
        """
        cached = self._cache.get(patient_id)
        if cached and cached.get("_schema") == _CACHE_SCHEMA_VERSION:
            return cached

        message = self._client.messages.create(
            model=config.CLAUDE_MODEL_FAST,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Patient note:\n{patient_text}\n\nJSON output:"}],
            temperature=0,
        )
        raw = strip_json_fences(message.content[0].text.strip())

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"summary": patient_text[:200], "conditions": [patient_text[:100]]}

        result.setdefault("summary", "")
        result.setdefault("conditions", [])

        # Normalize: keep only active/historical conditions; drop negated/hypothetical/family_history.
        # Always return a flat list of term strings so downstream code is unchanged.
        _KEEP_STATUSES = {"active", "historical"}
        _ALL_STATUSES = _KEEP_STATUSES | {"negated", "hypothetical", "family_history"}
        normalized = []
        for c in result["conditions"]:
            if isinstance(c, dict):
                status = c.get("status", "active")
                if status not in _ALL_STATUSES:
                    logger.warning(
                        "keyword_generator: unknown status %r for term %r — dropping",
                        status, c.get("term", ""),
                    )
                elif status in _KEEP_STATUSES:
                    normalized.append(c.get("term", ""))
            else:
                normalized.append(str(c))
        result["conditions"] = [t for t in normalized if t]

        if not result["conditions"]:
            logger.warning(
                "keyword_generator: all conditions filtered out for patient %r — "
                "retrieval will return no candidates",
                patient_id,
            )

        self._cache[patient_id] = {**result, "_schema": _CACHE_SCHEMA_VERSION}
        self._save_cache()
        return result
