import os
from pathlib import Path


def home_dir() -> Path:
    return Path(os.environ.get("PROMPT_SCOUT_HOME", "~/.prompt-scout")).expanduser()


def references_dir() -> Path:
    return home_dir() / "references"


def index_dir() -> Path:
    return home_dir() / "index"
