"""Loading and checking the labelled query set (evals/queries.jsonl).

One JSON object per line:
  id        unique string
  lang      "en" | "ru"  (language of the text part; free label, used for the breakdown)
  text      text query, may be empty for a photo-only query
  image     path or URL of an example photo, or null
  relevant  3-10 library ids that are correct answers (strings); may be empty while status is draft
  status    "draft" (labelled from the prompt text only) | "verified" (checked by the owner)
  note      optional free text
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MIN_RELEVANT, MAX_RELEVANT = 3, 10  # 5 at first; review can add more that truly fit


@dataclass
class Query:
    id: str
    text: str
    image: str | None
    relevant: list[str]
    lang: str = "en"
    status: str = "draft"
    note: str = ""

    @property
    def kind(self) -> str:
        if self.text.strip() and self.image:
            return "mixed"
        return "photo" if self.image else "text"


@dataclass
class Problems:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse(lines: list[str]) -> tuple[list[Query], Problems]:
    queries: list[Query] = []
    problems = Problems()
    seen: set[str] = set()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.errors.append(f"line {n}: not valid JSON ({exc.msg})")
            continue
        qid = str(raw.get("id", "")).strip()
        where = f"line {n} ({qid or '?'})"
        text, image = str(raw.get("text") or ""), raw.get("image") or None
        relevant = [str(r) for r in raw.get("relevant", [])]
        if not qid:
            problems.errors.append(f"line {n}: missing id")
            continue
        if qid in seen:
            problems.errors.append(f"{where}: duplicate id")
            continue
        seen.add(qid)
        if not text.strip() and not image:
            problems.errors.append(f"{where}: needs text, image, or both")
            continue
        status = str(raw.get("status", "draft"))
        if not relevant:
            if status == "verified":
                problems.errors.append(f"{where}: no relevant ids")
            else:  # a new query waiting for the labelling page: kept, skipped by the runner
                problems.warnings.append(f"{where}: not labelled yet")
                queries.append(Query(qid, text, image, [], str(raw.get("lang", "en")), status,
                                     str(raw.get("note", ""))))
            continue
        if len(set(relevant)) != len(relevant):
            problems.errors.append(f"{where}: repeated ids in relevant")
            continue
        if not MIN_RELEVANT <= len(relevant) <= MAX_RELEVANT:
            problems.warnings.append(f"{where}: {len(relevant)} relevant ids (expected {MIN_RELEVANT}-{MAX_RELEVANT})")
        queries.append(Query(qid, text, image, relevant, str(raw.get("lang", "en")),
                             status, str(raw.get("note", ""))))
    return queries, problems


def load(path: Path) -> tuple[list[Query], Problems]:
    return parse(path.read_text("utf-8").splitlines())
