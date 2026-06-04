"""
Clinical Trial Patient-Matching CLI

Usage examples:

  # Match a specific patient (full pipeline: retrieval + matching + aggregation)
  python run_matching.py --patient-id sigir-20141

  # Match all patients, top-10 trials each
  python run_matching.py --all --top-k 10

  # Retrieval only (fast, no LLM criterion assessment)
  python run_matching.py --patient-id sigir-20141 --skip-matching

  # Add patient metadata for smarter LanceDB filtering
  python run_matching.py --patient-id sigir-20141 --sex FEMALE --age 58

  # Quick smoke test (builds index on first 500 trials)
  python run_matching.py --patient-id sigir-20141 --max-trials 500
"""

import argparse
import json
import os
import sys

# Project root on path
sys.path.insert(0, os.path.dirname(__file__))

import config
from utils.data_loader import load_patients, get_patient_by_id
from pipeline.full_pipeline import MatchingPipeline


def _print_result(result: dict, verbose: bool = False) -> None:
    pid = result["patient_id"]
    print(f"\n{'='*70}")
    print(f"Patient: {pid}")
    print(f"Summary: {result.get('summary', '')}")
    print(f"Keywords: {', '.join(result.get('keywords', [])[:8])}")
    print(f"Candidates evaluated: {result.get('total_candidates', len(result['results']))}")
    print(f"{'='*70}")

    for trial in result["results"]:
        decision = trial["eligibility_decision"]
        icon = {"likely_eligible": "✓", "not_eligible": "✗", "needs_review": "?"}.get(
            decision, "?"
        )
        print(
            f"\n  [{trial['rank']:2d}] {icon} {decision.upper():<16} "
            f"score={trial['eligibility_score']:.3f}   {trial['nct_id']}"
        )
        print(f"       Title: {trial['title'][:80]}")
        print(f"       Conditions: {trial['conditions'][:60]}")

        if verbose or decision == "likely_eligible":
            met = trial.get("met_inclusion_criteria", [])
            unmet = trial.get("unmet_inclusion_criteria", [])
            exc = trial.get("triggered_exclusion_criteria", [])
            miss = trial.get("missing_information", [])

            if met:
                print(f"       Met inclusion ({len(met)}): "
                      + "; ".join(c["criterion"][:40] for c in met[:3]))
            if unmet:
                print(f"       Unmet inclusion ({len(unmet)}): "
                      + "; ".join(c["criterion"][:40] for c in unmet[:2]))
            if exc:
                print(f"       Triggered exclusions ({len(exc)}): "
                      + "; ".join(c["criterion"][:40] for c in exc[:2]))
            if miss:
                print(f"       Missing info ({len(miss)}): "
                      + "; ".join(c["criterion"][:40] for c in miss[:2]))

            if verbose and trial.get("eligibility_explanation"):
                print(f"       Explanation: {trial['eligibility_explanation'][:200]}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Match patient profiles against clinical trials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--patient-id", type=str, help="Single patient ID to match.")
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
    parser.add_argument("--sex", type=str, default=None, help="Patient sex (MALE/FEMALE).")
    parser.add_argument("--age", type=int, default=None, help="Patient age in years.")
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
        patient = get_patient_by_id(args.patient_id)
        if patient is None:
            print(f"ERROR: Patient '{args.patient_id}' not found in {config.PATIENT_PROFILES_JSONL}")
            sys.exit(1)
        patients = [patient]

    print(f"\nMatching {len(patients)} patient(s), top-{args.top_k} trials each.\n")

    for patient in patients:
        pid = patient["_id"]
        text = patient["text"]

        result = pipeline.match_patient(
            patient_id=pid,
            patient_text=text,
            top_k=args.top_k,
            skip_matching=args.skip_matching,
            sex_filter=args.sex,
            age=args.age,
        )

        # Save JSON output
        out_path = os.path.join(output_dir, f"{pid}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Saved] {out_path}")

        # Print summary to console
        _print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
