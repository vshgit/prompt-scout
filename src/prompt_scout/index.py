"""On-disk index and hybrid multimodal search.

Each prompt has up to two vectors in the same space:
  text vector  - title + description + prompt
  image vector - its sample image
A query (text and/or photo) is compared with both, plus BM25 keyword search on text;
the ranked lists are fused with Reciprocal Rank Fusion.

A few thousand x 768 float32 is tens of MB, so exact search in numpy takes
milliseconds; a vector database would add moving parts without improving anything here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from .dataset import Record
from .embed import Embedder, tokenize

RRF_K = 60
# How much each ranked list counts in the fusion. A photo is best compared with sample
# images; photo-vs-prompt-text is a weaker cross-modal signal, so it counts half.
WEIGHTS = {
    "photo->look": 1.0,
    "photo->meaning": 0.5,
    "text->meaning": 1.0,
    "text->look": 1.0,
    "text->keywords": 1.0,
}
ImageLoader = Callable[[Record], "bytes | None"]


@dataclass
class Hit:
    record: Record
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank


def image_key(r: Record) -> str:
    return f"{r.id}|{r.images[0]}" if r.images else ""


class Index:
    def __init__(self, records: list[Record], text_vectors: np.ndarray, image_vectors: np.ndarray,
                 image_keys: list[str], embedder_name: str):
        n = len(records)
        if len(text_vectors) != n or len(image_vectors) != n or len(image_keys) != n:
            raise ValueError("records and vectors length mismatch")
        self.records = records
        self.text_vectors = text_vectors.astype(np.float32)
        self.image_vectors = image_vectors.astype(np.float32)
        self.image_keys = image_keys  # "" where the image is missing
        self.has_image = np.array([bool(k) for k in image_keys], dtype=bool)
        self.embedder_name = embedder_name
        self.by_id = {r.id: i for i, r in enumerate(records)}
        self.text_hashes = [r.content_hash() for r in records]  # snapshot at build time
        self._bm25 = BM25Okapi([tokenize(r.embed_text()) or ["_"] for r in records])

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "text_vectors.npy", self.text_vectors)
        np.save(path / "image_vectors.npy", self.image_vectors)
        with (path / "records.jsonl").open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        meta = {"embedder": self.embedder_name, "count": len(self.records),
                "with_images": int(self.has_image.sum()), "image_keys": self.image_keys}
        (path / "meta.json").write_text(json.dumps(meta), "utf-8")

    @classmethod
    def load(cls, path: Path) -> "Index":
        meta = json.loads((path / "meta.json").read_text("utf-8"))
        with (path / "records.jsonl").open(encoding="utf-8") as f:
            records = [Record.from_dict(json.loads(line)) for line in f if line.strip()]
        return cls(records, np.load(path / "text_vectors.npy"), np.load(path / "image_vectors.npy"),
                   meta["image_keys"], meta["embedder"])

    # ---------- build ----------
    @classmethod
    def build(cls, records: list[Record], embedder: Embedder, load_image: ImageLoader | None = None,
              previous: "Index | None" = None,
              progress: Callable[[str, int, int], None] | None = None,
              chunk: int = 200) -> "Index":
        """Embed texts and images, reusing vectors from `previous` when nothing changed."""
        text_reuse: dict[str, np.ndarray] = {}
        image_reuse: dict[str, np.ndarray] = {}
        if previous is not None and previous.embedder_name == embedder.name:
            text_reuse = dict(zip(previous.text_hashes, previous.text_vectors))
            image_reuse = {k: v for k, v in zip(previous.image_keys, previous.image_vectors) if k}

        n = len(records)
        text_vectors = np.zeros((n, embedder.dim), dtype=np.float32)
        image_vectors = np.zeros((n, embedder.dim), dtype=np.float32)
        image_keys = [""] * n

        todo_text = []
        for i, r in enumerate(records):
            h = r.content_hash()
            if h in text_reuse:
                text_vectors[i] = text_reuse[h]
            else:
                todo_text.append(i)
        for s in range(0, len(todo_text), chunk):
            idx = todo_text[s:s + chunk]
            text_vectors[idx] = embedder.embed_texts([records[i].embed_text() for i in idx])
            if progress:
                progress("texts", min(s + chunk, len(todo_text)), len(todo_text))

        if load_image is not None:
            todo_img = []
            for i, r in enumerate(records):
                key = image_key(r)
                if key and key in image_reuse:
                    image_vectors[i], image_keys[i] = image_reuse[key], key
                elif key:
                    todo_img.append(i)
            for s in range(0, len(todo_img), chunk):
                batch = [(i, load_image(records[i])) for i in todo_img[s:s + chunk]]
                batch = [(i, b) for i, b in batch if b]
                if batch:
                    vecs = embedder.embed_images([b for _, b in batch])
                    for (i, _), v in zip(batch, vecs):
                        image_vectors[i], image_keys[i] = v, image_key(records[i])
                if progress:
                    progress("images", min(s + chunk, len(todo_img)), len(todo_img))

        return cls(records, text_vectors, image_vectors, image_keys, embedder.name)

    # ---------- search ----------
    def _mask(self, with_reference: bool | None, category: str | None) -> np.ndarray:
        mask = np.ones(len(self.records), dtype=bool)
        if with_reference is not None:
            mask &= np.array([r.needs_reference == with_reference for r in self.records])
        if category:
            mask &= np.array([category in r.categories for r in self.records])
        return mask

    @staticmethod
    def _ranked(scores: np.ndarray, mask: np.ndarray, pool: int) -> list[int]:
        scores = np.where(mask & (scores > 0), scores, -np.inf)  # no similarity = not a match
        return [int(i) for i in np.argsort(-scores)[:pool] if np.isfinite(scores[i])]

    def ranked_lists(self, embedder: Embedder, query: str = "", image: bytes | None = None,
                     with_reference: bool | None = None, category: str | None = None,
                     pool: int = 100) -> dict[str, list[int]]:
        """The ranked lists (record positions, best first), one per signal, before fusion.

        Kept separate from fusion so that evaluation can embed a query once and then
        re-fuse the same lists with other weights or with signals switched off.
        """
        if embedder.name != self.embedder_name:
            raise ValueError(f"index built with {self.embedder_name}, query embedder is {embedder.name}")
        if not query.strip() and image is None:
            raise ValueError("Give a text query, an image, or both.")
        mask = self._mask(with_reference, category)
        if not mask.any():
            return {}

        lists: dict[str, list[int]] = {}
        query_vectors = []
        if image is not None:
            query_vectors.append(("photo", embedder.embed_images([image])[0]))
        if query.strip():
            query_vectors.append(("text", embedder.embed_texts([query], query=True)[0]))
        for src, qv in query_vectors:
            lists[f"{src}->look"] = self._ranked(self.image_vectors @ qv, mask & self.has_image, pool)
            lists[f"{src}->meaning"] = self._ranked(self.text_vectors @ qv, mask, pool)
        if query.strip():
            q_tokens = tokenize(query)
            if q_tokens:
                kw = self._bm25.get_scores(q_tokens)
                lists["text->keywords"] = self._ranked(kw, mask, pool)
        return lists

    def fuse(self, lists: dict[str, list[int]], limit: int = 8,
             weights: dict[str, float] | None = None) -> list[Hit]:
        """Weighted Reciprocal Rank Fusion. A list with weight 0 is switched off."""
        weights = WEIGHTS if weights is None else weights
        fused: dict[int, Hit] = {}
        for name, order in lists.items():
            w = weights.get(name, 1.0)
            if w <= 0:
                continue
            for rank, i in enumerate(order):
                hit = fused.setdefault(i, Hit(self.records[i], 0.0))
                hit.score += w / (RRF_K + rank + 1)
                hit.ranks[name] = rank + 1
        return sorted(fused.values(), key=lambda h: -h.score)[:limit]

    def search(self, embedder: Embedder, query: str = "", image: bytes | None = None,
               limit: int = 8, with_reference: bool | None = None, category: str | None = None,
               pool: int = 100) -> list[Hit]:
        """query: text in any language; image: JPEG/PNG bytes of an example. Either or both."""
        lists = self.ranked_lists(embedder, query, image, with_reference, category, pool)
        return self.fuse(lists, limit)

    def get(self, prompt_id: str) -> Record | None:
        i = self.by_id.get(str(prompt_id))
        return None if i is None else self.records[i]
