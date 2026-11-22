"""Add a query to evals/queries.jsonl without editing JSON by hand.

    python -m evals.add --image "https://images.unsplash.com/photo-..." --lang en
    python -m evals.add --image "https://..." --text "same look, but in winter" --lang en
    python -m evals.add --text "свадебный портрет" --lang ru

The query is added as a draft with no answers; label it on the page from `python -m evals.review`.
Photos should come from outside the library (not its own sample images) and be shareable,
e.g. Unsplash or Wikimedia Commons URLs: the query file is public.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .queries import load


def next_id(ids: list[str]) -> str:
    nums = [int(i[1:]) for i in ids if i.startswith("q") and i[1:].isdigit()]
    return f"q{max(nums, default=0) + 1:02d}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.add", description=__doc__.split("\n\n")[0])
    p.add_argument("--text", default="", help="text part of the query")
    p.add_argument("--image", help="example photo: URL (preferred, the file is public) or file path")
    p.add_argument("--lang", choices=["en", "ru"], default="en", help="language of the text part")
    p.add_argument("--note", default="", help="free note, e.g. where the photo is from")
    p.add_argument("--queries", type=Path, default=Path(__file__).with_name("queries.jsonl"))
    p.add_argument("--no-check", action="store_true", help="do not try to load the photo")
    args = p.parse_args(argv)

    if not args.text.strip() and not args.image:
        p.error("give --text, --image, or both")
    if args.image and not args.no_check:
        from prompt_scout.service import read_image
        read_image(args.image)  # fails loudly now rather than in the middle of an evaluation run

    queries, problems = load(args.queries) if args.queries.exists() else ([], None)
    if problems and problems.errors:
        print("Query file has errors:\n  " + "\n  ".join(problems.errors), file=sys.stderr)
        return 2
    qid = next_id([q.id for q in queries])
    row = {"id": qid, "text": args.text, "image": args.image or None, "relevant": [],
           "lang": args.lang, "status": "draft", "note": args.note}
    text = args.queries.read_text("utf-8") if args.queries.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    args.queries.write_text(text + json.dumps(row, ensure_ascii=False) + "\n", "utf-8")
    kind = "mixed" if args.text.strip() and args.image else ("photo" if args.image else "text")
    print(f"Added {qid} ({kind}). Label it: python -m evals.review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
