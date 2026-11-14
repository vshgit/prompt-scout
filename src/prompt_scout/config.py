"""Settings: paths and the user's library config (~/.prompt-scout/config.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ALL_CATEGORIES = [
    "profile-avatar", "social-media-post", "product-marketing", "poster-flyer",
    "infographic-edu-visual", "comic-storyboard", "game-asset", "ecommerce-main-image",
    "app-web-design", "youtube-thumbnail", "others",
]

DEFAULT_CONFIG = """\
# Prompt Scout settings. After editing, run `prompt-scout index` again:
# only new or changed prompts are embedded, removed ones simply drop out.

[library]
# Which categories of the YouMind library to index. Prompt counts (Nov 2026):
#   profile-avatar 2158 | social-media-post 9876 | product-marketing 5781
#   others 1107 | poster-flyer 1049 | game-asset 732 | comic-storyboard 705
#   infographic-edu-visual 607 | ecommerce-main-image 583
#   app-web-design 240 | youtube-thumbnail 231
# A prompt can be in several categories; it is indexed once.
categories = [
  "profile-avatar",
  # "social-media-post",
]

# Prompt ids you never want to see (e.g. "553"). Shown on every result card.
exclude_ids = []

# Download sample images and index them, so you can search by photo and by look.
images = true

[preview]
# Open an HTML page with numbered result cards after each search.
open_browser = true
# Also send small thumbnails into the chat.
thumbnails_in_chat = true
"""


@dataclass
class Config:
    categories: list[str] = field(default_factory=lambda: ["profile-avatar"])
    exclude_ids: list[str] = field(default_factory=list)
    images: bool = True
    open_browser: bool = True
    thumbnails_in_chat: bool = True

    @classmethod
    def from_toml(cls, text: str) -> "Config":
        data = tomllib.loads(text)
        lib, prev = data.get("library", {}), data.get("preview", {})
        cats = [str(c) for c in lib.get("categories", ["profile-avatar"])]
        unknown = [c for c in cats if c not in ALL_CATEGORIES]
        if unknown:
            raise ValueError(f"Unknown categories in config: {unknown}. Known: {ALL_CATEGORIES}")
        return cls(
            categories=cats,
            exclude_ids=[str(i) for i in lib.get("exclude_ids", [])],
            images=bool(lib.get("images", True)),
            open_browser=bool(prev.get("open_browser", True)),
            thumbnails_in_chat=bool(prev.get("thumbnails_in_chat", True)),
        )


def home_dir() -> Path:
    return Path(os.environ.get("PROMPT_SCOUT_HOME", "~/.prompt-scout")).expanduser()


def config_path() -> Path:
    return home_dir() / "config.toml"


def references_dir() -> Path:
    return home_dir() / "references"


def index_dir() -> Path:
    return home_dir() / "index"


def images_dir() -> Path:
    return home_dir() / "images"


def results_dir() -> Path:
    return home_dir() / "results"


def write_default_config(overwrite: bool = False) -> Path:
    path = config_path()
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG, "utf-8")
    return path


def load_config() -> Config:
    path = config_path()
    return Config.from_toml(path.read_text("utf-8")) if path.exists() else Config()
