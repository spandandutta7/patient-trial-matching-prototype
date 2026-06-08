"""
One-time index builder: encodes all trials into the LanceDB vector store.

Run this before the first matching session:
    python pipeline/build_index.py [--max-trials N]

--max-trials N  Limit to first N trials (useful for fast smoke tests; default: all ~60K)

NOTE: If you already have a cache/lancedb/ directory from a previous build,
delete it first — the builder reuses existing tables and won't pick up schema
changes otherwise:
    rm -rf cache/lancedb/ cache/medcpt_nctids.json
"""

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils.data_loader import load_trials
from retrieval.medcpt_retriever import MedCPTRetriever


def main():
    parser = argparse.ArgumentParser(description="Build the MedCPT LanceDB index.")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit to first N trials (default: all ~60K)",
    )
    args = parser.parse_args()

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trials dataset …")
    trials_df = load_trials(nrows=args.max_trials)
    print(f"Loaded {len(trials_df):,} trials.")

    print("\n--- Building MedCPT + LanceDB Index ---")
    medcpt = MedCPTRetriever()
    medcpt.build(trials_df)

    print("\nIndex build complete.")
    print(f"  LanceDB: {config.LANCEDB_DIR}")


if __name__ == "__main__":
    main()
