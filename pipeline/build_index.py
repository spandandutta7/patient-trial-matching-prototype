"""
One-time index builder: creates the BM25 JSON cache and the LanceDB vector store.

Run this before the first matching session:
    python pipeline/build_index.py [--max-trials N]

--max-trials N  Limit to first N trials (useful for fast smoke tests; default: all)
"""

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils.data_loader import load_trials
from retrieval.bm25_retriever import BM25Retriever
from retrieval.medcpt_retriever import MedCPTRetriever


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and LanceDB indices.")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit to first N trials (default: all ~60K)",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only the BM25 index (skip MedCPT encoding)",
    )
    parser.add_argument(
        "--medcpt-only",
        action="store_true",
        help="Build only the MedCPT/LanceDB index (skip BM25)",
    )
    args = parser.parse_args()

    # Ensure cache and output dirs exist
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading trials dataset …")
    trials_df = load_trials(nrows=args.max_trials)
    print(f"Loaded {len(trials_df):,} trials.")

    if not args.medcpt_only:
        print("\n--- Step 1: BM25 Index ---")
        bm25 = BM25Retriever()
        bm25.build(trials_df)

    if not args.bm25_only:
        print("\n--- Step 2: MedCPT + LanceDB Index ---")
        medcpt = MedCPTRetriever()
        medcpt.build(trials_df)

    print("\nIndex build complete.")
    print(f"  BM25 cache: {config.BM25_CACHE_PATH}")
    print(f"  LanceDB:    {config.LANCEDB_DIR}")


if __name__ == "__main__":
    main()
