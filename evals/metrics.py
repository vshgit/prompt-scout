"""Ranking metrics with binary relevance. Pure functions, no I/O.

`ranked` is a list of ids, best first; `relevant` is the set of correct ids.
All metrics are cut off at k (default 10) and return a float in [0, 1].
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Share of the correct ids that appear in the top k."""
    rel = set(relevant)
    if not rel:
        raise ValueError("a query needs at least one relevant id")
    return len(rel.intersection(ranked[:k])) / len(rel)


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """1.0 if at least one correct id is in the top k, else 0.0."""
    rel = set(relevant)
    if not rel:
        raise ValueError("a query needs at least one relevant id")
    return 1.0 if rel.intersection(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """1 / rank of the first correct id within the top k (0 if none). Its mean is MRR@k."""
    rel = set(relevant)
    if not rel:
        raise ValueError("a query needs at least one relevant id")
    for pos, rid in enumerate(ranked[:k], 1):
        if rid in rel:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Binary nDCG: correct ids near the top count more; 1.0 = all of them first."""
    rel = set(relevant)
    if not rel:
        raise ValueError("a query needs at least one relevant id")
    dcg = sum(1.0 / math.log2(pos + 1) for pos, rid in enumerate(ranked[:k], 1) if rid in rel)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(rel), k) + 1))
    return dcg / ideal


METRICS = {
    "Recall@10": recall_at_k,
    "MRR@10": reciprocal_rank,
    "nDCG@10": ndcg_at_k,
    "Hit@10": hit_at_k,
}


def score_query(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> dict[str, float]:
    rel = set(relevant)
    return {name: fn(ranked, rel, k) for name, fn in METRICS.items()}


def mean_scores(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {name: 0.0 for name in METRICS}
    return {name: sum(q[name] for q in per_query) / len(per_query) for name in METRICS}
