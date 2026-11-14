"""prompt-scout fetch | index | search | serve"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import index_dir, references_dir


def _embedder(kind: str):
    from .embed import GeminiEmbedder, HashEmbedder
    return HashEmbedder() if kind == "hash" else GeminiEmbedder()


def cmd_fetch(args) -> None:
    from .dataset import fetch_references
    dest = fetch_references(Path(args.dest) if args.dest else references_dir())
    print(f"Downloaded library to {dest}")


def cmd_index(args) -> None:
    from .dataset import load_records
    from .index import Index

    src = Path(args.references) if args.references else references_dir()
    out = index_dir()
    records = load_records(src)
    if args.limit:
        records = records[: args.limit]
    previous = Index.load(out) if (out / "meta.json").exists() and not args.rebuild else None
    embedder = _embedder(args.embedder)

    def progress(done: int, total: int) -> None:
        print(f"  embedded {done}/{total}", file=sys.stderr)

    idx = Index.build(records, embedder, previous=previous, progress=progress)
    idx.save(out)
    print(f"Indexed {len(records)} prompts with {embedder.name} -> {out}")


def cmd_search(args) -> None:
    from .index import Index
    from .service import PromptScout

    idx = Index.load(index_dir())
    embedder = _embedder("hash" if idx.embedder_name.startswith("hash") else "gemini")
    wr = {"yes": True, "no": False}.get(args.with_reference)
    results = PromptScout(idx, embedder).search(args.query, args.limit, wr, args.category)
    for r in results:
        print(f"[{r['id']}] {r['title']}  ({', '.join(r['matched_by'])}, ref={r['needs_reference']})")
        print(f"    {r['image']}")
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_serve(args) -> None:
    from .server import run
    run()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="prompt-scout")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the prompt library from GitHub")
    f.add_argument("--dest")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("index", help="embed prompts (incremental: unchanged prompts are reused)")
    i.add_argument("--references", help="folder with manifest.json and category files")
    i.add_argument("--embedder", choices=["gemini", "hash"], default="gemini")
    i.add_argument("--limit", type=int, help="index only the first N prompts (for a cheap trial)")
    i.add_argument("--rebuild", action="store_true")
    i.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="search from the terminal")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=8)
    s.add_argument("--with-reference", choices=["yes", "no"])
    s.add_argument("--category")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    sv = sub.add_parser("serve", help="run the MCP server over stdio")
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
