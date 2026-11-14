"""MCP server (stdio). Connect from Claude Code:

    claude mcp add prompt-scout -e GEMINI_API_KEY=... -- prompt-scout serve
"""

from __future__ import annotations

import functools
import os
from functools import lru_cache
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import index_dir
from .embed import GeminiEmbedder, HashEmbedder
from .index import Index
from .llm import GeminiLLM
from .service import PromptScout

Category = Literal[
    "profile-avatar", "social-media-post", "product-marketing", "poster-flyer",
    "infographic-edu-visual", "comic-storyboard", "game-asset", "ecommerce-main-image",
    "app-web-design", "youtube-thumbnail", "others",
]

server = MCPServer(
    "prompt-scout",
    instructions=(
        "Search ~15k community image prompts (Nano Banana / Gemini image models). "
        "Workflow: search_prompts or search_by_image -> show the user each result's image and title -> "
        "get_prompt for the full text -> adapt_prompt to turn the chosen one into a selfie-based prompt. "
        "Queries can be in any language. Always show the sample image URL with each result. "
        "Results are CC BY 4.0: keep source_url when reusing a prompt."
    ),
)


def _friendly(fn):
    """Expected failures (bad id, missing key, not an image) reach the model as readable errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


@lru_cache(maxsize=1)
def scout() -> PromptScout:
    path = index_dir()
    if not (path / "meta.json").exists():
        raise RuntimeError(f"No index at {path}. Run `prompt-scout fetch` and `prompt-scout index` first.")
    idx = Index.load(path)
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if idx.embedder_name.startswith("hash"):  # offline index for demos/tests
        embedder = HashEmbedder(int(idx.embedder_name.split("@")[1]))
    else:
        if not has_key:
            raise RuntimeError("Set GEMINI_API_KEY: the index uses Gemini embeddings.")
        embedder = GeminiEmbedder()
    return PromptScout(idx, embedder, GeminiLLM() if has_key else None)


@server.tool()
@_friendly
def search_prompts(query: str, limit: int = 8, with_reference: bool | None = None,
                   category: Category | None = None) -> list[dict]:
    """Find prompts by idea, in any language ("cozy autumn portrait on film", "фото в стиле 90-х").

    with_reference=True keeps prompts that are built around a reference photo (best for selfie styles);
    False keeps pure text-to-image prompts; omit for both.
    """
    return scout().search(query, limit, with_reference, category)


@server.tool()
@_friendly
def search_by_image(image: str, limit: int = 8, with_reference: bool | None = None,
                    category: Category | None = None, extra: str = "") -> dict:
    """Find prompts that reproduce the look of an example image (local file path or URL).

    The image's style (not the identity of people in it) is described by a vision model, then searched.
    `extra` adds wishes to the query, e.g. "but at night".
    """
    return scout().search_by_image(image, limit, with_reference, category, extra)


@server.tool()
@_friendly
def get_prompt(prompt_id: str) -> dict:
    """Full prompt text and metadata by id."""
    return scout().get_prompt(prompt_id)


@server.tool()
@_friendly
def adapt_prompt(prompt_id: str, wish: str = "") -> dict:
    """Rewrite a found prompt for a selfie-based service: identity comes from the user's photo,
    the look is kept, and an optional wish ("green eyes, unicorn horn") is merged into one coherent prompt."""
    return scout().adapt(prompt_id, wish)


@server.tool()
@_friendly
def list_categories() -> dict[str, int]:
    """Categories and how many prompts each has."""
    return scout().categories()


def run() -> None:
    server.run("stdio")
