"""Tool logic, independent of MCP so it can be tested and reused from the CLI."""

from __future__ import annotations

from .dataset import fill_placeholders
from .embed import Embedder
from .index import Hit, Index
from .llm import LLM, read_image

PREVIEW_CHARS = 280


def hit_to_dict(hit: Hit) -> dict:
    r = hit.record
    prompt = fill_placeholders(r.prompt)
    return {
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
        "matched_by": [m for m, v in (("meaning", hit.dense_rank), ("keywords", hit.keyword_rank)) if v],
    }


class PromptScout:
    def __init__(self, index: Index, embedder: Embedder, llm: LLM | None = None):
        self.index = index
        self.embedder = embedder
        self.llm = llm

    def _need_llm(self) -> LLM:
        if self.llm is None:
            raise RuntimeError("This tool needs GEMINI_API_KEY (LLM is not configured).")
        return self.llm

    def search(self, query: str, limit: int = 8, with_reference: bool | None = None,
               category: str | None = None) -> list[dict]:
        limit = max(1, min(limit, 25))
        hits = self.index.search(query, self.embedder, limit=limit,
                                 with_reference=with_reference, category=category)
        return [hit_to_dict(h) for h in hits]

    def search_by_image(self, image: str, limit: int = 8, with_reference: bool | None = None,
                        category: str | None = None, extra: str = "") -> dict:
        img = read_image(image)
        described = self._need_llm().describe_image(img.data, img.mime_type)
        query = described.get("search_query") or described.get("style_description", "")
        if extra.strip():
            query = f"{query}. {extra.strip()}"
        return {
            "style_description": described.get("style_description", ""),
            "search_query": query,
            "results": self.search(query, limit, with_reference, category),
        }

    def get_prompt(self, prompt_id: str) -> dict:
        r = self.index.get(prompt_id)
        if r is None:
            raise ValueError(f"No prompt with id {prompt_id}")
        d = r.to_dict()
        d["prompt"] = fill_placeholders(r.prompt)
        return d

    def adapt(self, prompt_id: str, wish: str = "") -> dict:
        r = self.index.get(prompt_id)
        if r is None:
            raise ValueError(f"No prompt with id {prompt_id}")
        out = self._need_llm().adapt(r, wish)
        return {
            "source_id": r.id,
            "source_title": r.title,
            "source_url": r.source_url,
            "license": r.license,
            "adapted_prompt": out.get("adapted_prompt", ""),
            "changes": out.get("changes", []),
            "extra_references": out.get("extra_references", []),
            "warnings": out.get("warnings", []),
        }

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.index.records:
            for c in r.categories:
                counts[c] = counts.get(c, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
