"""
Reciprocal Rank Fusion (RRF) hybrid retrieval.

Combines BM25 (keyword-based) and MedCPT (semantic) ranked lists using the
same RRF formula as TrialGPT:

    score(trial) += (1 / (rank + k)) × (1 / (condition_index + 1))

The condition_index downweighting means the primary condition (index 0)
contributes full weight while secondary conditions contribute less — matching
TrialGPT's prioritisation of the patient's main problem.
"""

import config


def fuse(
    bm25_results: list[list[str]],
    medcpt_results: list[list[str]],
    k: int = None,
    bm25_wt: int = None,
    medcpt_wt: int = None,
) -> list[tuple[str, float]]:
    """Fuse per-condition BM25 and MedCPT ranked lists with RRF.

    Both `bm25_results` and `medcpt_results` are lists indexed by condition:
      [ [nct_id_1, nct_id_2, ...],   # condition 0 results
        [nct_id_1, nct_id_2, ...],   # condition 1 results
        ... ]

    Returns a list of (nct_id, score) tuples sorted by score descending.
    """
    k = k if k is not None else config.RRF_K
    bm25_wt = bm25_wt if bm25_wt is not None else config.BM25_WEIGHT
    medcpt_wt = medcpt_wt if medcpt_wt is not None else config.MEDCPT_WEIGHT

    nctid2score: dict[str, float] = {}

    for condition_idx, (bm25_top, medcpt_top) in enumerate(
        zip(bm25_results, medcpt_results)
    ):
        condition_weight = 1.0 / (condition_idx + 1)  # Primary condition = full weight

        if bm25_wt > 0:
            for rank, nctid in enumerate(bm25_top):
                nctid2score[nctid] = nctid2score.get(nctid, 0.0) + (
                    (1.0 / (rank + k)) * condition_weight
                )

        if medcpt_wt > 0:
            for rank, nctid in enumerate(medcpt_top):
                nctid2score[nctid] = nctid2score.get(nctid, 0.0) + (
                    (1.0 / (rank + k)) * condition_weight
                )

    return sorted(nctid2score.items(), key=lambda x: -x[1])



'''
Example of output from both BM25 and MedCPT for a single condition:
bm25_results = [
  ["NCT-A", "NCT-B", "NCT-C", ...],   # index 0 → "chest pain"   
  ["NCT-D", "NCT-A", ...],            # index 1 → "hypertension"
  ["NCT-E", ...],                     # index 2 → "obesity"
]

Note how it is ordered by each keyword condition
The position of an id in its list is its rank




nctid2score[nctid] += (1.0 / (rank + k)) * condition_weight

1. The reciprocal-rank term 1 / (rank + k) (with k = RRF_K = 20 as default)
This converts a rank into a contribution that decreases as you go down the list

2. The condition weight: 1 / (condition_idx + 1) 
primary condition = full weight, secondary conditions contribute less



If a trial is found by only one method, it never gets the second deposit (mild penalty for lacking cross-method agreement)


'''