"""Text embedders. Gemini for real use, a deterministic hashing embedder for tests."""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Protocol, Sequence

import numpy as np

DEFAULT_EMBED_MODEL = os.environ.get("PROMPT_SCOUT_EMBED_MODEL", "gemini-embedding-001")
DEFAULT_DIM = int(os.environ.get("PROMPT_SCOUT_EMBED_DIM", "768"))


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class GeminiEmbedder:
    """gemini-embedding-001: multilingual, so Russian queries match English prompts.

    Truncated dimensions (<3072) are not normalized by the API, so we normalize.
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dim: int = DEFAULT_DIM,
                 batch_size: int = 100, client=None):
        if client is None:
            from google import genai
            client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.name = f"{model}@{dim}"

    def _embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        from google.genai import types

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            for attempt in range(6):
                try:
                    resp = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type, output_dimensionality=self.dim),
                    )
                    out.extend(e.values for e in resp.embeddings)
                    break
                except Exception as exc:  # rate limits and transient errors
                    if attempt == 5 or not _retryable(exc):
                        raise
                    time.sleep(min(60, 2 ** attempt * 2))
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _retryable(exc: Exception) -> bool:
    text = str(exc)
    return any(code in text for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class HashEmbedder:
    """Bag-of-words hashing embedder. Offline, deterministic; for tests and dry runs."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash@{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vec(text)[None, :])[0]
