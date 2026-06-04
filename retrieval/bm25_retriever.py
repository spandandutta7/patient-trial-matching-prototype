"""
BM25 keyword-based retriever.

BM25 index construction:
  - Title tokens × 3 (highest importance)
  - Disease/condition tokens × 2 (medical focus boost)
  - Full combined-text tokens × 1

Index is built from trials_clean.csv and cached as JSON for fast reuse.
"""

import json
import os
from typing import Optional
import pandas as pd
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
import nltk
import config

# Download NLTK tokenizer data if not already present
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def _build_tokenized_corpus(trials_df: pd.DataFrame) -> tuple[list, list]:
    """Tokenize corpus with TrialGPT-style field weighting."""
    tokenized_corpus = []
    corpus_nctids = []

    for _, row in trials_df.iterrows():
        nct_id = str(row.get("nct_id", ""))
        title = str(row.get("title", ""))
        conditions = str(row.get("conditions", ""))
        text = str(row.get("combined_text_for_retrieval", ""))

        # Weight: title × 3, conditions × 2, text × 1 (same as TrialGPT)
        tokens = word_tokenize(title.lower()) * 3

        # Parse pipe-separated conditions and weight each
        for cond in conditions.split("|"):
            cond = cond.strip()
            if cond:
                tokens += word_tokenize(cond.lower()) * 2

        tokens += word_tokenize(text.lower())

        tokenized_corpus.append(tokens)
        corpus_nctids.append(nct_id)

    return tokenized_corpus, corpus_nctids


class BM25Retriever:
    def __init__(self, cache_path: str = None):
        self._cache_path = str(cache_path or config.BM25_CACHE_PATH)
        self._bm25: Optional[BM25Okapi] = None
        self._nctids: Optional[list] = None

    def build(self, trials_df: pd.DataFrame) -> None:
        """Build and cache the BM25 index from a trials DataFrame."""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)

        if os.path.exists(self._cache_path):
            print(f"[BM25] Loading cached index from {self._cache_path}")
            data = json.load(open(self._cache_path))
            tokenized_corpus = data["tokenized_corpus"]
            self._nctids = data["corpus_nctids"]
        else:
            print(f"[BM25] Building index for {len(trials_df):,} trials …")
            tokenized_corpus, self._nctids = _build_tokenized_corpus(trials_df)
            with open(self._cache_path, "w") as f:
                json.dump(
                    {"tokenized_corpus": tokenized_corpus, "corpus_nctids": self._nctids},
                    f,
                )
            print(f"[BM25] Index cached to {self._cache_path}")

        self._bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25] Ready — {len(self._nctids):,} documents indexed.")

    def search(self, conditions: list[str], n: int = None) -> list[list[str]]:
        """Search for each condition string.

        Returns a list aligned with `conditions`, where each element is an
        ordered list of NCT IDs (highest BM25 score first).
        """
        if self._bm25 is None or self._nctids is None:
            raise RuntimeError("Call build() before search().")
        n = n or config.BM25_TOP_N
        results = []
        for condition in conditions:
            tokens = word_tokenize(condition.lower())
            top_nctids = self._bm25.get_top_n(tokens, self._nctids, n=n)
            results.append(top_nctids)
        return results
