"""Shared JSON parsing utilities."""


def strip_json_fences(raw: str) -> str:
    """Remove ```json / ``` markdown fences from an LLM response string."""
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return raw
