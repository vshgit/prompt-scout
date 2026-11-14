"""MCP server (stdio). Connect from Claude Code:

    claude mcp add prompt-scout -e GEMINI_API_KEY=... -- prompt-scout serve
"""

from __future__ import annotations

import functools
import json
import os
import webbrowser
from functools import lru_cache
from typing import Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import images_dir, index_dir, load_config, results_dir
from .embed import embedder_for
from .index import Index
from .llm import GeminiLLM
from .service import PromptScout, SearchResult

Category = Literal[
    "profile-avatar", "social-media-post", "product-marketing", "poster-flyer",
    "infographic-edu-visual", "comic-storyboard", "game-asset", "ecommerce-main-image",
    "app-web-design", "youtube-thumbnail", "others",
]

server = MCPServer(
    "prompt-scout",
    instructions=(
        "Search a library of community image prompts (Nano Banana / Gemini image models) by text, "
        "by example photo, or both. Results come back numbered (n=1..N) with thumbnails, and an HTML page "
        "with the same numbered cards opens for the user. Workflow: search_prompts -> show the user a short "
        "numbered list (number, title, one line on why it fits) -> the user picks a number -> get_prompt or "
        "adapt_prompt with that result's id. adapt_prompt is the default next step: it makes the prompt "
        "identity-neutral for selfie generation. If a tool fails, show the user the error text; do not "
        "silently do the tool's job yourself. Queries can be in any language. "
        "Results are CC BY 4.0: keep source_url when reusing a prompt."
    ),
)


def _friendly(fn):
    """Expected failures (bad id, missing key, not an image) reach the model as readable errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # local tool: show the real reason instead of a generic failure
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    return wrapper


@lru_cache(maxsize=1)
def scout() -> PromptScout:
    path = index_dir()
    if not (path / "meta.json").exists():
        raise RuntimeError(f"No index at {path}. Run `prompt-scout fetch` and `prompt-scout index` first.")
    idx = Index.load(path)
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if not idx.embedder_name.startswith("hash") and not has_key:
        raise RuntimeError("Set GEMINI_API_KEY: the index uses Gemini embeddings.")
    return PromptScout(idx, embedder_for(idx.embedder_name), GeminiLLM() if has_key else None,
                       cache_dir=images_dir(), results_dir=results_dir())


def _present(res: SearchResult) -> list:
    cfg = load_config()
    if res.preview_path and cfg.open_browser:
        try:
            webbrowser.open(res.preview_path.resolve().as_uri())
        except Exception:
            pass
    summary = {
        "results": res.results,
        "preview_page": str(res.preview_path) if res.preview_path else None,
        "note": "Thumbnails follow in result order, each preceded by its number."
        if cfg.thumbnails_in_chat else None,
    }
    out: list = [json.dumps(summary, ensure_ascii=False, indent=1)]
    if cfg.thumbnails_in_chat:
        for n, thumb in scout().thumbnails(res.results):
            out.extend([f"#{n}", Image(data=thumb, format="jpeg")])
    return out


@server.tool()
@_friendly
def search_prompts(query: str = "", image: str = "", limit: int = 8,
                   with_reference: bool | None = None, category: Category | None = None) -> list[str | Image]:
    """Find prompts by idea, by example photo, or both.

    query: idea in any language ("голливудский портрет", "film flash party photo").
    image: path to a local image file or an image URL; finds prompts whose sample results LOOK like it.
    Use both to steer: image of a style + query "but at night".
    with_reference=True keeps prompts built around a reference photo (best for selfie styles).
    """
    return _present(scout().search(query, image or None, limit, with_reference, category))


@server.tool()
@_friendly
def get_prompt(prompt_id: str) -> dict:
    """Full prompt text and metadata by id."""
    return scout().get_prompt(prompt_id)


@server.tool()
@_friendly
def adapt_prompt(prompt_id: str = "", wish: str = "", keep: str = "", prompt_text: str = "") -> dict:
    """Make a prompt identity-neutral for a selfie service (ImagineMe default): gender, age, ethnicity,
    face, body and natural hair are removed so they come from the user's reference photos; the look stays.
    Appearance details that are part of the style (pink hair, character make-up) are kept and listed in
    kept_as_style; doubtful ones are removed and listed in removed.

    prompt_id: a library result id. Or prompt_text: any prompt (e.g. one of the user's own styles).
    keep: details to keep despite neutralizing, e.g. "pink hair, freckles".
    wish: extra edits merged into one coherent prompt, e.g. "horizontal 16:9, green instead of yellow".
    Show the user adapted_prompt in full, then removed / kept_as_style / changes / warnings briefly.
    """
    return scout().adapt(prompt_id, wish=wish, keep=keep, prompt_text=prompt_text)


@server.tool()
@_friendly
def list_categories() -> dict[str, int]:
    """Indexed categories and how many prompts each has."""
    return scout().categories()


def run() -> None:
    server.run("stdio")
