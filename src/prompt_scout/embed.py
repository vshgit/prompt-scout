"""Embedders: text and images into one vector space.

Default: gemini-embedding-2 (multimodal). A photo, a sample image and a text query all
become points in the same space, so "photo vs sample images" and "text vs sample images"
are plain nearest-neighbour searches, like image search on a marketplace.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Protocol, Sequence

import numpy as np

DEFAULT_EMBED_MODEL = os.environ.get("PROMPT_SCOUT_EMBED_MODEL", "gemini-embedding-2")
DEFAULT_DIM = int(os.environ.get("PROMPT_SCOUT_EMBED_DIM", "768"))


class Embedder(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: Sequence[str], query: bool = False) -> np.ndarray: ...

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _retryable(exc: Exception) -> bool:
    text = str(exc)
    return any(c in text for c in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE"))


class GeminiEmbedder:
    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dim: int = DEFAULT_DIM,
                 text_batch: int = 100, image_batch: int = 20, client=None):
        if client is None:
            from google import genai
            client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self.client = client
        self.model = model
        self.dim = dim
        self.text_batch = text_batch
        self.image_batch = image_batch
        self.name = f"{model}@{dim}"

    def _config(self, task_type: str | None):
        from google.genai import types
        # gemini-embedding-001 uses task types; gemini-embedding-2 does not take them.
        use_task = task_type if self.model.startswith("gemini-embedding-001") else None
        return types.EmbedContentConfig(task_type=use_task, output_dimensionality=self.dim)

    def _call(self, contents: list, task_type: str | None) -> list[list[float]]:
        for attempt in range(6):
            try:
                resp = self.client.models.embed_content(
                    model=self.model, contents=contents, config=self._config(task_type))
                vals = [e.values for e in resp.embeddings]
                if len(vals) != len(contents):  # one vector per input, never a merged one
                    raise RuntimeError(f"expected {len(contents)} embeddings, got {len(vals)}")
                return vals
            except Exception as exc:
                if attempt == 5 or not _retryable(exc):
                    raise
                time.sleep(min(60, 2 ** attempt * 2))
        raise AssertionError("unreachable")

    def embed_texts(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), self.text_batch):
            out.extend(self._call(list(texts[i:i + self.text_batch]), task))
        return l2_normalize(np.asarray(out, dtype=np.float32).reshape(len(texts), -1))

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(images), self.image_batch):
            # each image is its own Content, so each gets its own vector
            contents = [types.Content(parts=[types.Part.from_bytes(data=b, mime_type="image/jpeg")])
                        for b in images[i:i + self.image_batch]]
            out.extend(self._call(contents, None))
        return l2_normalize(np.asarray(out, dtype=np.float32).reshape(len(images), -1))


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class HashEmbedder:
    """Offline bag-of-words embedder for tests and dry runs.

    "Images" are embedded by reading their bytes as text, so tests can use
    fake images like b"neon cyberpunk street" and still exercise cross-modal search.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash@{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % self.dim] += 1.0
        return v

    def embed_texts(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        return self.embed_texts([b.decode("utf-8", "ignore") for b in images])


def embedder_for(name: str) -> Embedder:
    """Recreate the embedder an index was built with."""
    model, dim = name.rsplit("@", 1)
    if model == "hash":
        return HashEmbedder(int(dim))
    return GeminiEmbedder(model=model, dim=int(dim))
