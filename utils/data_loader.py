"""Utilities for loading the clinical trial datasets."""

import re
import pandas as pd
from pathlib import Path
from typing import Optional
import config


def load_trials(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the cleaned trials dataset."""
    df = pd.read_csv(config.TRIALS_CSV, nrows=nrows, low_memory=False)
    # Normalize conditions: use source_condition_query as primary, fall back to conditions col
    df["conditions"] = df["conditions"].fillna(df["source_condition_query"])
    df["conditions"] = df["conditions"].fillna("")
    df["title"] = df["title"].fillna("")
    df["inclusion_criteria"] = df["inclusion_criteria"].fillna("")
    df["exclusion_criteria"] = df["exclusion_criteria"].fillna("")
    df["brief_summary"] = df["brief_summary"].fillna("")
    df["interventions"] = df["interventions"].fillna("")
    df["minimum_age"] = df["minimum_age"].fillna("")
    df["maximum_age"] = df["maximum_age"].fillna("")
    df["sex"] = df["sex"].fillna("ALL")
    df["phase"] = df["phase"].fillna("")
    df["study_type"] = df["study_type"].fillna("")
    return df


def load_criteria_chunks() -> pd.DataFrame:
    """Load all eligibility criteria chunks."""
    df = pd.read_csv(config.CRITERIA_CHUNKS_CSV, low_memory=False)
    df["criterion_text"] = df["criterion_text"].fillna("")
    return df


# Matches the summary table row:  | **Sex / Age** | female, 70 at enrollment |
_SEX_AGE_RE = re.compile(
    r"Sex\s*/\s*Age\s*\*\*\s*\|\s*(male|female)\s*,\s*(\d+)", re.IGNORECASE
)


def _parse_patient_summary(summary_path: Path, patient_id: str) -> dict:
    """Build a patient dict from a patient_summary.md file.

    Returns {_id, text, sex, age} where:
      - text: the full summary markdown (the matching pipeline's input note)
      - sex:  "FEMALE" / "MALE" / None  (uppercased to match trial metadata)
      - age:  int years at enrollment / None
    These drive the LanceDB sex/age pre-filter; missing values fall back to None
    (no filtering) so a malformed summary degrades gracefully.
    """
    text = summary_path.read_text()
    sex, age = None, None
    m = _SEX_AGE_RE.search(text)
    if m:
        sex = m.group(1).upper()
        age = int(m.group(2))
    return {"_id": patient_id, "text": text, "sex": sex, "age": age}


def load_patients() -> list[dict]:
    """Load all patients from patient-datasets/<id>/patient_summary.md.

    Each immediate subdirectory of PATIENT_DATASETS_DIR is one patient; the
    folder name is the patient _id (e.g. "breast-cancer").
    """
    patients = []
    base = config.PATIENT_DATASETS_DIR
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = folder / config.PATIENT_SUMMARY_FILENAME
        if summary_path.exists():
            patients.append(_parse_patient_summary(summary_path, folder.name))
    return patients


def get_patient_by_id(patient_id: str) -> Optional[dict]:
    """Retrieve a single patient by folder name or by indication string.

    Accepts either the raw folder name (e.g. "breast-cancer") or the
    corresponding SUPPORTED_INDICATIONS string (e.g. "breast cancer"), so
    --patient-indication works with both naming conventions.
    """
    from retrieval.indication_classifier import INDICATION_TO_FOLDER

    folder_name = INDICATION_TO_FOLDER.get(patient_id.lower(), patient_id)
    summary_path = (
        config.PATIENT_DATASETS_DIR / folder_name / config.PATIENT_SUMMARY_FILENAME
    )
    if summary_path.exists():
        return _parse_patient_summary(summary_path, folder_name)
    return None


def get_trial_info(nct_id: str, trials_df: pd.DataFrame) -> Optional[dict]:
    """Return a dict of key trial fields for a given NCT ID."""
    rows = trials_df[trials_df["nct_id"] == nct_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    # Parse conditions list from pipe-separated string
    conditions_str = str(row.get("conditions", ""))
    conditions_list = [c.strip() for c in conditions_str.split("|") if c.strip()]
    # Parse interventions list
    interventions_str = str(row.get("interventions", ""))
    interventions_list = [i.strip() for i in interventions_str.split("|") if i.strip()]
    return {
        "nct_id": nct_id,
        "brief_title": str(row.get("title", "")),
        "official_title": str(row.get("official_title", "")),
        "brief_summary": str(row.get("brief_summary", "")),
        "conditions": conditions_str,
        "diseases_list": conditions_list,
        "interventions": interventions_str,
        "drugs_list": interventions_list,
        "overall_status": str(row.get("overall_status", "")),
        "study_type": str(row.get("study_type", "")),
        "phase": str(row.get("phase", "")),
        "sex": str(row.get("sex", "ALL")),
        "minimum_age": str(row.get("minimum_age", "")),
        "maximum_age": str(row.get("maximum_age", "")),
        "inclusion_criteria": str(row.get("inclusion_criteria", "")),
        "exclusion_criteria": str(row.get("exclusion_criteria", "")),
        "clinicaltrials_url": str(row.get("clinicaltrials_url", "")),
    }


def parse_age_to_years(age_str: str) -> Optional[int]:
    """Parse age strings like '20 Years', '6 Months' to integer years."""
    if not age_str or age_str.strip() == "":
        return None
    age_str = age_str.strip().lower()
    parts = age_str.split()
    try:
        value = float(parts[0])
        if "month" in age_str:
            return int(value / 12)
        elif "week" in age_str:
            return int(value / 52)
        elif "day" in age_str:
            return int(value / 365)
        else:  # years
            return int(value)
    except (ValueError, IndexError):
        return None
