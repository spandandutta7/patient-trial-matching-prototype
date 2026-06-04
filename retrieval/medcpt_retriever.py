"""
MedCPT semantic retriever backed by LanceDB.

Mirrors TrialGPT's MedCPT FAISS approach but stores vectors in LanceDB,
which adds metadata filtering and structured retrieval on top of ANN search.

Encoding:
  - Trials    → ncbi/MedCPT-Article-Encoder  (title + text, max 512 tokens)
  - Queries   → ncbi/MedCPT-Query-Encoder    (condition string, max 256 tokens)
  - Representation: [CLS] hidden state (768-dim float32)

Storage:
  - LanceDB table "trials" with schema:
      nct_id, title, conditions, sex, min_age, max_age,
      study_type, phase, combined_text, vector (768-dim)
"""

import json
import os
from typing import Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import lancedb
import pyarrow as pa
import config
from utils.data_loader import parse_age_to_years

_LANCEDB_TABLE = "trials"
_NCTIDS_CACHE = str(config.CACHE_DIR / "medcpt_nctids.json")


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_articles(
    title_text_pairs: list[tuple[str, str]],
    tokenizer,
    model,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode (title, text) pairs using MedCPT-Article-Encoder."""
    all_embeds = []
    for i in tqdm(range(0, len(title_text_pairs), batch_size), desc="Encoding trials"):
        batch = title_text_pairs[i : i + batch_size]
        with torch.no_grad():
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=config.MEDCPT_ARTICLE_MAX_LEN,
            ).to(device)
            embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embeds.append(embeds)
    return np.vstack(all_embeds)


class MedCPTRetriever:
    def __init__(self, db_dir: str = None):
        self._db_dir = str(db_dir or config.LANCEDB_DIR)
        self._db = None
        self._table = None
        self._nctids: Optional[list] = None
        self._query_tokenizer = None
        self._query_model = None
        self._device = _get_device()

    def _open_db(self):
        self._db = lancedb.connect(self._db_dir)

    def _load_query_encoder(self):
        if self._query_model is None:
            print(f"[MedCPT] Loading query encoder on {self._device} …")
            self._query_tokenizer = AutoTokenizer.from_pretrained(
                config.MEDCPT_QUERY_ENCODER
            )
            self._query_model = AutoModel.from_pretrained(
                config.MEDCPT_QUERY_ENCODER
            ).to(self._device)
            self._query_model.eval()

    def build(self, trials_df: pd.DataFrame) -> None:
        """Encode all trials and store in LanceDB. Skip if table already exists."""
        os.makedirs(self._db_dir, exist_ok=True)
        self._open_db()

        if _LANCEDB_TABLE in self._db.table_names():
            print(f"[MedCPT] LanceDB table '{_LANCEDB_TABLE}' already exists — loading.")
            self._table = self._db.open_table(_LANCEDB_TABLE)
            if os.path.exists(_NCTIDS_CACHE):
                self._nctids = json.load(open(_NCTIDS_CACHE))
            else:
                self._nctids = self._table.to_pandas()["nct_id"].tolist()
            print(f"[MedCPT] Loaded {len(self._nctids):,} trial embeddings.")
            return

        print(f"[MedCPT] Encoding {len(trials_df):,} trials with MedCPT-Article-Encoder …")
        print(f"[MedCPT] Device: {self._device}")
        if self._device == "cpu":
            print("[MedCPT] WARNING: CPU encoding is slow (~2-6h for 60K trials). "
                  "Consider using a GPU or reducing dataset size with --max-trials.")

        art_tokenizer = AutoTokenizer.from_pretrained(config.MEDCPT_ARTICLE_ENCODER)
        art_model = AutoModel.from_pretrained(config.MEDCPT_ARTICLE_ENCODER).to(
            self._device
        )
        art_model.eval()

        title_text_pairs = [
            (str(row["title"]), str(row["combined_text_for_retrieval"]))
            for _, row in trials_df.iterrows()
        ]
        nctids = trials_df["nct_id"].tolist()

        embeds = _encode_articles(
            title_text_pairs, art_tokenizer, art_model,
            self._device, config.MEDCPT_BATCH_SIZE
        )

        # Free GPU memory after encoding
        del art_model
        if self._device == "cuda":
            torch.cuda.empty_cache()

        print("[MedCPT] Storing embeddings in LanceDB …")
        records = []
        for i, (nct_id, row) in enumerate(zip(nctids, trials_df.itertuples())):
            min_age = parse_age_to_years(str(getattr(row, "minimum_age", ""))) or -1
            max_age = parse_age_to_years(str(getattr(row, "maximum_age", ""))) or 200
            records.append({
                "nct_id": nct_id,
                "title": str(getattr(row, "title", "")),
                "conditions": str(getattr(row, "conditions", "")),
                "sex": str(getattr(row, "sex", "ALL")),
                "min_age": min_age,
                "max_age": max_age,
                "study_type": str(getattr(row, "study_type", "")),
                "phase": str(getattr(row, "phase", "")),
                "vector": embeds[i].astype(np.float32).tolist(),
            })

        schema = pa.schema([
            pa.field("nct_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("conditions", pa.string()),
            pa.field("sex", pa.string()),
            pa.field("min_age", pa.int32()),
            pa.field("max_age", pa.int32()),
            pa.field("study_type", pa.string()),
            pa.field("phase", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
        ])

        self._table = self._db.create_table(
            _LANCEDB_TABLE, data=records, schema=schema, mode="overwrite"
        )
        self._nctids = nctids
        os.makedirs(os.path.dirname(_NCTIDS_CACHE), exist_ok=True)
        json.dump(nctids, open(_NCTIDS_CACHE, "w"))
        print(f"[MedCPT] LanceDB index built with {len(nctids):,} trials.")

    def load(self) -> None:
        """Open an existing LanceDB table without re-encoding."""
        self._open_db()
        if _LANCEDB_TABLE not in self._db.table_names():
            raise RuntimeError(
                "LanceDB index not found. Run 'python pipeline/build_index.py' first."
            )
        self._table = self._db.open_table(_LANCEDB_TABLE)
        if os.path.exists(_NCTIDS_CACHE):
            self._nctids = json.load(open(_NCTIDS_CACHE))
        else:
            self._nctids = self._table.to_pandas()["nct_id"].tolist()

    def search(
        self,
        conditions: list[str],
        k: int = None,
        sex_filter: str = None,
        min_age: int = None,
        max_age: int = None,
    ) -> list[list[str]]:
        """Encode query conditions and search the LanceDB ANN index.

        Returns a list aligned with `conditions`, each element being an ordered
        list of NCT IDs (most similar first).

        Optional metadata filters (sex_filter, min_age, max_age) narrow the
        search space — this is the structured-retrieval advantage over pure FAISS.
        """
        if self._table is None:
            raise RuntimeError("Call build() or load() before search().")
        k = k or config.MEDCPT_TOP_N

        self._load_query_encoder()

        with torch.no_grad():
            encoded = self._query_tokenizer(
                conditions,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=config.MEDCPT_QUERY_MAX_LEN,
            ).to(self._device)
            query_embeds = (
                self._query_model(**encoded)
                .last_hidden_state[:, 0, :]
                .cpu()
                .float()
                .numpy()
            )

        # Build optional WHERE clause for metadata pre-filtering
        where_clauses = []
        if sex_filter and sex_filter.upper() not in ("ALL", ""):
            where_clauses.append(f"(sex = '{sex_filter.upper()}' OR sex = 'ALL')")
        if min_age is not None:
            where_clauses.append(f"max_age >= {min_age}")
        if max_age is not None:
            where_clauses.append(f"min_age <= {max_age}")
        where_str = " AND ".join(where_clauses) if where_clauses else None

        results = []
        for embed in query_embeds:
            q = self._table.search(embed.tolist(), vector_column_name="vector").limit(k)
            if where_str:
                # prefilter is a hint for some LanceDB versions; fall back gracefully
                try:
                    q = q.where(where_str, prefilter=True)
                except TypeError:
                    q = q.where(where_str)
            df = q.to_pandas()
            results.append(df["nct_id"].tolist())
        return results
