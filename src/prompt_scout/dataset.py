"""Loading the public prompt library into normalized records.

Source: YouMind-OpenLab prompt library (CC BY 4.0), published as category JSON
files in the `references/` folder of
https://github.com/YouMind-OpenLab/nano-banana-pro-prompts-recommend-skill .
A prompt can appear in several categories; records are merged by id.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_RAW = (
    "https://raw.githubusercontent.com/YouMind-OpenLab/"
    "nano-banana-pro-prompts-recommend-skill/main/references"
)
SOURCE_PAGE = "https://youmind.com/en-US/nano-banana-pro-prompts?id={id}"
LICENSE = "CC BY 4.0 (YouMind OpenLab prompt library)"
SOURCE_MODEL = "nano-banana-pro"

# Raycast-style placeholders: {argument name="x" default="y"} -> y
_ARG_RE = re.compile(r'\{argument\s+name="[^"]*"\s+default="([^"]*)"\}')


@dataclass
class Record:
    id: str
    title: str
    description: str
    prompt: str
    needs_reference: bool
    images: list[str]
    categories: list[str] = field(default_factory=list)
    source_model: str = SOURCE_MODEL
    source_url: str = ""
    license: str = LICENSE

    def embed_text(self, max_prompt_chars: int = 2000) -> str:
        """Text used for embedding and keyword search."""
        return f"{self.title}\n{self.description}\n{self.prompt[:max_prompt_chars]}"

    def content_hash(self) -> str:
        return hashlib.sha256(self.embed_text().encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        return cls(**d)


def fill_placeholders(text: str) -> str:
    """Replace Raycast placeholders with their default values."""
    return _ARG_RE.sub(lambda m: m.group(1), text)


def _parse_media(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except (ValueError, SyntaxError):
            return [value]
    return []


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize(raw: dict, category: str) -> Record:
    rid = str(raw["id"])
    return Record(
        id=rid,
        title=(raw.get("title") or "").strip(),
        description=(raw.get("description") or "").strip(),
        prompt=(raw.get("content") or "").strip(),
        needs_reference=_parse_bool(raw.get("needReferenceImages", False)),
        images=_parse_media(raw.get("sourceMedia")),
        categories=[category],
        source_url=SOURCE_PAGE.format(id=rid),
    )


def load_records(references_dir: Path) -> list[Record]:
    """Read every category file in a `references/` folder and merge by id."""
    manifest = json.loads((references_dir / "manifest.json").read_text("utf-8"))
    by_id: dict[str, Record] = {}
    for cat in manifest["categories"]:
        path = references_dir / cat["file"]
        if not path.exists():
            continue
        for raw in json.loads(path.read_text("utf-8")):
            rec = normalize(raw, cat["slug"])
            if not rec.prompt:
                continue
            if rec.id in by_id:
                if cat["slug"] not in by_id[rec.id].categories:
                    by_id[rec.id].categories.append(cat["slug"])
            else:
                by_id[rec.id] = rec
    return sorted(by_id.values(), key=lambda r: int(r.id) if r.id.isdigit() else r.id)


def select(records: list[Record], categories: list[str], exclude_ids: list[str]) -> list[Record]:
    """Keep records in any of the chosen categories, minus excluded ids."""
    cats, excluded = set(categories), set(exclude_ids)
    return [r for r in records if r.id not in excluded and cats.intersection(r.categories)]


def fetch_references(dest: Path, base_url: str = REPO_RAW) -> Path:
    """Download manifest + category files from GitHub into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> bytes:
        with urllib.request.urlopen(f"{base_url}/{name}", timeout=120) as resp:
            return resp.read()

    manifest_bytes = get("manifest.json")
    (dest / "manifest.json").write_bytes(manifest_bytes)
    for cat in json.loads(manifest_bytes)["categories"]:
        (dest / cat["file"]).write_bytes(get(cat["file"]))
    return dest
