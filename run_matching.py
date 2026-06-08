"""
Clinical Trial Patient-Matching CLI

Usage examples:

  # Match a single patient (full pipeline: retrieval + matching + aggregation)
  python run_matching.py --patient-indication breast-cancer

  # Match all patients, verbose criterion breakdown
  python run_matching.py --all -v

  # Retrieval only (fast, no LLM criterion assessment)
  python run_matching.py --patient-indication breast-cancer --skip-matching

  # Override auto-parsed sex/age for LanceDB pre-filtering
  python run_matching.py --patient-indication breast-cancer --sex FEMALE --age 70

  # Quick smoke test (builds index on first 500 trials)
  python run_matching.py --patient-indication breast-cancer --max-trials 500
"""

import argparse
import json
import os
import sys
import textwrap

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Project root on path
sys.path.insert(0, os.path.dirname(__file__))

import config
from utils.data_loader import load_patients, get_patient_by_id
from pipeline.full_pipeline import MatchingPipeline


_W = 72  # output width

_ICONS  = {"likely_eligible": "✓", "not_eligible": "✗", "needs_review": "?"}
_LABELS = {"likely_eligible": "LIKELY ELIGIBLE", "not_eligible": "NOT ELIGIBLE  ", "needs_review": "NEEDS REVIEW  "}


def _wrap(text: str, indent: int) -> str:
    p = " " * indent
    return textwrap.fill(text, width=_W, initial_indent=p, subsequent_indent=p)


def _wrap_bullet(text: str, bullet_indent: int = 10) -> str:
    return textwrap.fill(
        text, width=_W,
        initial_indent=" " * bullet_indent + "• ",
        subsequent_indent=" " * (bullet_indent + 2),
    )


def _print_result(result: dict, verbose: bool = False) -> None:
    # ── Patient header ──────────────────────────────────────────────────
    print(f"\n{'━' * _W}")
    print(f"  PATIENT  {result['patient_id'].upper()}")
    if result.get("summary"):
        print(_wrap(result["summary"], indent=2))
    kwds = result.get("keywords", [])[:6]
    if kwds:
        kwd_line = ", ".join(kwds)
        print(f"  Keywords : {kwd_line[:_W - 12]}" + ("…" if len(kwd_line) > _W - 12 else ""))
    print(f"  Trials   : {result.get('total_candidates', len(result['results']))} evaluated")
    print(f"{'━' * _W}\n")

    # ── Per-trial results ───────────────────────────────────────────────
    for trial in result["results"]:
        decision = trial.get("eligibility_decision", "")
        icon  = _ICONS.get(decision, " ")
        label = _LABELS.get(decision, "")

        if decision:
            conf = trial.get("confidence")
            conf_str = f"(conf {conf:.2f})" if conf is not None else "(conf N/A)"
            print(f"  [{trial['rank']:2d}]  {icon} {label}  {trial['eligibility_score']:.3f}  {conf_str}  {trial['nct_id']}")
        else:
            print(f"  [{trial['rank']:2d}]  retrieval score {trial.get('retrieval_score', 0):.4f}  {trial['nct_id']}")

        print(_wrap(trial["title"], indent=8))

        if decision and (verbose or decision == "likely_eligible"):
            met   = trial.get("met_inclusion_criteria", [])
            unmet = trial.get("unmet_inclusion_criteria", [])
            exc   = trial.get("triggered_exclusion_criteria", [])
            miss  = trial.get("missing_information", [])

            print()
            for heading, items, limit in [
                ("Met inclusion",         met,   4),
                ("Not met",               unmet, 3),
                ("Triggered exclusions",  exc,   3),
                ("Missing info",          miss,  3),
            ]:
                if items:
                    print(f"        {heading} ({len(items)}):")
                    for c in items[:limit]:
                        print(_wrap_bullet(c["criterion"].strip()))

            if trial.get("eligibility_explanation"):
                print()
                print(_wrap(trial["eligibility_explanation"], indent=8))

        print(f"\n  {'─' * (_W - 4)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Match patient profiles against clinical trials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--patient-indication", type=str, help="Patient indication to match (e.g. breast-cancer).")
    target.add_argument("--all", action="store_true", help="Match all patients.")

    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K_RETRIEVAL,
        help=f"Number of trials to retrieve per patient (default: {config.TOP_K_RETRIEVAL}).",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip criterion-level matching; return retrieval results only.",
    )
    parser.add_argument(
        "--sex", type=str, default=None,
        help="Override patient sex (MALE/FEMALE); default auto-parsed from summary.",
    )
    parser.add_argument(
        "--age", type=int, default=None,
        help="Override patient age in years; default auto-parsed from summary.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit trial corpus to first N rows (for quick testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory to save JSON results (default: ./output).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed per-criterion breakdown."
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialise and load indices
    pipeline = MatchingPipeline()
    pipeline.setup(max_trials=args.max_trials)

    # Determine which patients to process
    if args.all:
        patients = load_patients()
    else:
        patient = get_patient_by_id(args.patient_indication)
        if patient is None:
            print(f"ERROR: '{args.patient_indication}' not found in {config.PATIENT_DATASETS_DIR}")
            sys.exit(1)
        patients = [patient]

    print(f"\nMatching {len(patients)} patient(s), top-{args.top_k} trials each.\n")

    for patient in patients:
        pid = patient["_id"]
        text = patient["text"]

        # Auto-fill sex/age from the patient summary; CLI flags override.
        sex_filter = args.sex or patient.get("sex")
        age = args.age if args.age is not None else patient.get("age")
        if sex_filter or age is not None:
            print(f"[{pid}]  Filter  sex={sex_filter}  age={age}")

        result = pipeline.match_patient(
            patient_id=pid,
            patient_text=text,
            top_k=args.top_k,
            skip_matching=args.skip_matching,
            sex_filter=sex_filter,
            age=age,
        )

        # Save JSON output
        out_path = os.path.join(output_dir, f"{pid}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}\n")

        # Print summary to console
        _print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
