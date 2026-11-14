"""On-disk index and hybrid search (dense vectors + BM25, fused with RRF).

15k x 768 float32 is ~45 MB, so exact cosine search in numpy takes milliseconds;
a vector database would add moving parts without improving anything here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from .dataset import Record
from .embed import Embedder, tokenize

RRF_K = 60


@dataclass
class Hit:
    record: Record
    score: float
    dense_rank: int | None
    keyword_rank: int | None


class Index:
    def __init__(self, records: list[Record], vectors: np.ndarray, embedder_name: str):
        if len(records) != len(vectors):
            raise ValueError("records and vectors length mismatch")
        self.records = records
        self.vectors = vectors.astype(np.float32)
        self.embedder_name = embedder_name
        self.by_id = {r.id: i for i, r in enumerate(records)}
        # snapshot of the text each vector was computed from
        self.hashes = [r.content_hash() for r in records]
        self._bm25 = BM25Okapi([tokenize(r.embed_text()) or ["_"] for r in records])

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with (path / "records.jsonl").open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        meta = {"embedder": self.embedder_name, "count": len(self.records)}
        (path / "meta.json").write_text(json.dumps(meta), "utf-8")

    @classmethod
    def load(cls, path: Path) -> "Index":
        meta = json.loads((path / "meta.json").read_text("utf-8"))
        with (path / "records.jsonl").open(encoding="utf-8") as f:
            records = [Record.from_dict(json.loads(line)) for line in f if line.strip()]
        return cls(records, np.load(path / "vectors.npy"), meta["embedder"])

    # ---------- build ----------
    @classmethod
    def build(cls, records: list[Record], embedder: Embedder,
              previous: "Index | None" = None,
              progress: Callable[[int, int], None] | None = None,
              chunk: int = 500) -> "Index":
        """Embed records, reusing vectors from `previous` when text is unchanged."""
        reuse: dict[str, np.ndarray] = {}
        if previous is not None and previous.embedder_name == embedder.name:
            for h, vec in zip(previous.hashes, previous.vectors):
                reuse[h] = vec

        vectors = np.zeros((len(records), embedder.dim), dtype=np.float32)
        todo = []
        for i, r in enumerate(records):
            h = r.content_hash()
            if h in reuse:
                vectors[i] = reuse[h]
            else:
                todo.append(i)

        for start in range(0, len(todo), chunk):
            idx = todo[start:start + chunk]
            vectors[idx] = embedder.embed_documents([records[i].embed_text() for i in idx])
            if progress:
                progress(min(start + chunk, len(todo)), len(todo))
        return cls(records, vectors, embedder.name)

    # ---------- search ----------
    def _mask(self, with_reference: bool | None, category: str | None) -> np.ndarray:
        mask = np.ones(len(self.records), dtype=bool)
        if with_reference is not None:
            mask &= np.array([r.needs_reference == with_reference for r in self.records])
        if category:
            mask &= np.array([category in r.categories for r in self.records])
        return mask

    def search(self, query: str, embedder: Embedder, limit: int = 8,
               with_reference: bool | None = None, category: str | None = None,
               pool: int = 100) -> list[Hit]:
        """Hybrid search. Dense handles meaning and cross-language queries;
        BM25 rescues exact terms (brand names, 'polaroid', 'ukiyo-e')."""
        if embedder.name != self.embedder_name:
            raise ValueError(f"index built with {self.embedder_name}, query embedder is {embedder.name}")
        mask = self._mask(with_reference, category)
        if not mask.any():
            return []

        dense = self.vectors @ embedder.embed_query(query)
        dense[~mask] = -np.inf
        dense_order = [int(i) for i in np.argsort(-dense)[:pool] if mask[i]]

        q_tokens = tokenize(query)
        keyword_order: list[int] = []
        if q_tokens:
            kw = self._bm25.get_scores(q_tokens)
            kw[~mask] = -np.inf
            keyword_order = [int(i) for i in np.argsort(-kw)[:pool] if mask[i] and kw[i] > 0]

        fused: dict[int, list] = {}
        for rank, i in enumerate(dense_order):
            fused.setdefault(i, [0.0, None, None])
            fused[i][0] += 1.0 / (RRF_K + rank + 1)
            fused[i][1] = rank + 1
        for rank, i in enumerate(keyword_order):
            fused.setdefault(i, [0.0, None, None])
            fused[i][0] += 1.0 / (RRF_K + rank + 1)
            fused[i][2] = rank + 1

        best = sorted(fused.items(), key=lambda kv: -kv[1][0])[:limit]
        return [Hit(self.records[i], s, d, k) for i, (s, d, k) in best]

    def get(self, prompt_id: str) -> Record | None:
        i = self.by_id.get(str(prompt_id))
        return None if i is None else self.records[i]
