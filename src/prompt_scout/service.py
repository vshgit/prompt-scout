"""Tool logic, independent of MCP so it can be tested and reused from the CLI."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from . import net
from .dataset import fill_placeholders
from .embed import Embedder
from .images import cache_path, thumbnail, to_jpeg
from .index import Hit, Index
from .llm import LLM

PREVIEW_CHARS = 280


def hit_to_dict(hit: Hit, number: int) -> dict:
    r = hit.record
    prompt = fill_placeholders(r.prompt)
    return {
        "n": number,
        "id": r.id,
        "title": r.title,
        "description": r.description,
        "prompt_preview": prompt[:PREVIEW_CHARS] + ("…" if len(prompt) > PREVIEW_CHARS else ""),
        "needs_reference": r.needs_reference,
        "image": r.images[0] if r.images else None,
        "categories": r.categories,
        "source_model": r.source_model,
        "source_url": r.source_url,
        "score": round(hit.score, 4),
        "matched_by": list(hit.ranks),
    }


def read_image(path_or_url: str, max_bytes: int = 15 * 1024 * 1024) -> bytes:
    """Load an example image and normalize it to a JPEG of at most 640 px."""
    if path_or_url.startswith(("http://", "https://")):
        data = net.get(path_or_url, timeout=60, max_bytes=max_bytes + 1)
    else:
        p = Path(path_or_url).expanduser()
        if not p.exists():
            raise ValueError(f"File not found: {p}")
        guessed = mimetypes.guess_type(p.name)[0] or ""
        if guessed and not guessed.startswith("image/"):
            raise ValueError(f"Not an image: {p.name}")
        data = p.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("Image is larger than 15 MB")
    try:
        return to_jpeg(data)
    except Exception as exc:
        raise ValueError(f"Could not read the image: {exc}") from exc


@dataclass
class SearchResult:
    query: str
    query_image: bytes | None
    results: list[dict]
    preview_path: Path | None = None


class PromptScout:
    def __init__(self, index: Index, embedder: Embedder, llm: LLM | None = None,
                 cache_dir: Path | None = None, results_dir: Path | None = None):
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.cache_dir = cache_dir
        self.results_dir = results_dir

    def search(self, query: str = "", image: str | None = None, limit: int = 8,
               with_reference: bool | None = None, category: str | None = None,
               write_preview: bool = True) -> SearchResult:
        limit = max(1, min(limit, 24))
        img = read_image(image) if image else None
        hits = self.index.search(self.embedder, query=query, image=img, limit=limit,
                                 with_reference=with_reference, category=category)
        res = SearchResult(query, img, [hit_to_dict(h, n) for n, h in enumerate(hits, 1)])
        if write_preview and self.results_dir is not None and res.results:
            from .preview import render
            prompts = {r["id"]: self.full_prompt(r["id"]) for r in res.results}
            res.preview_path = render(res.results, prompts, query, img,
                                      self.cache_dir or Path("."), self.results_dir)
        return res

    def thumbnails(self, results: list[dict]) -> list[tuple[int, bytes]]:
        out = []
        if self.cache_dir is None:
            return out
        for r in results:
            p = cache_path(self.cache_dir, r["id"])
            if p.exists():
                out.append((r["n"], thumbnail(p)))
        return out

    def full_prompt(self, prompt_id: str) -> str:
        r = self.index.get(prompt_id)
        if r is None:
            raise ValueError(f"No prompt with id {prompt_id}")
        return fill_placeholders(r.prompt)

    def get_prompt(self, prompt_id: str) -> dict:
        r = self.index.get(prompt_id)
        if r is None:
            raise ValueError(f"No prompt with id {prompt_id}")
        d = r.to_dict()
        d["prompt"] = fill_placeholders(r.prompt)
        return d

    def adapt(self, prompt_id: str = "", wish: str = "", keep: str = "", prompt_text: str = "") -> dict:
        """Identity-neutral version of a library prompt (by id) or of any prompt text."""
        if self.llm is None:
            raise RuntimeError("This tool needs GEMINI_API_KEY (LLM is not configured).")
        if prompt_text.strip():
            source, needs_ref, meta = prompt_text, False, {"source_id": None, "source_title": None,
                                                          "source_url": None, "license": None}
        else:
            r = self.index.get(prompt_id)
            if r is None:
                raise ValueError(f"No prompt with id {prompt_id!r}. Give prompt_id or prompt_text.")
            source, needs_ref = r.prompt, r.needs_reference
            meta = {"source_id": r.id, "source_title": r.title, "source_url": r.source_url, "license": r.license}
        out = self.llm.adapt(source, wish=wish, keep=keep, needs_reference=needs_ref)
        return {**meta, **out}

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.index.records:
            for c in r.categories:
                counts[c] = counts.get(c, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
