"""prompt-scout init | fetch | index | search | serve"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .config import (config_path, images_dir, index_dir, load_config, references_dir,
                     results_dir, write_default_config)


def _log(stage: str, done: int, total: int) -> None:
    print(f"  {stage}: {done}/{total}", file=sys.stderr)


def cmd_init(args) -> None:
    path = write_default_config(overwrite=args.force)
    print(f"Config: {path}\nEdit categories there, then run `prompt-scout fetch` and `prompt-scout index`.")


def cmd_fetch(args) -> None:
    from .dataset import fetch_references
    dest = fetch_references(Path(args.dest) if args.dest else references_dir())
    print(f"Downloaded library to {dest}")


def cmd_index(args) -> None:
    from .dataset import load_records, select
    from .embed import GeminiEmbedder, HashEmbedder
    from .images import cache_path, download_all
    from .index import Index

    write_default_config()
    cfg = load_config()
    src = Path(args.references) if args.references else references_dir()
    records = select(load_records(src), cfg.categories, cfg.exclude_ids)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} prompts in categories {cfg.categories} "
          f"(excluded ids: {len(cfg.exclude_ids)})", file=sys.stderr)

    load_image = None
    if cfg.images and not args.no_images:
        cache = images_dir()
        print("Downloading sample images (cached ones are skipped)...", file=sys.stderr)
        got = download_all(((r.id, r.images[0]) for r in records if r.images), cache,
                           progress=lambda d, t: _log("images downloaded", d, t))
        print(f"  {len(got)} images available", file=sys.stderr)
        load_image = lambda r: cache_path(cache, r.id).read_bytes() if r.id in got else None  # noqa: E731

    out = index_dir()
    previous = Index.load(out) if (out / "meta.json").exists() and not args.rebuild else None
    embedder = HashEmbedder() if args.embedder == "hash" else GeminiEmbedder()
    idx = Index.build(records, embedder, load_image=load_image, previous=previous, progress=_log)
    idx.save(out)
    print(f"Indexed {len(records)} prompts ({int(idx.has_image.sum())} with images) "
          f"with {embedder.name} -> {out}")


def cmd_search(args) -> None:
    from .embed import embedder_for
    from .index import Index
    from .service import PromptScout

    idx = Index.load(index_dir())
    scout = PromptScout(idx, embedder_for(idx.embedder_name),
                        cache_dir=images_dir(), results_dir=results_dir())
    wr = {"yes": True, "no": False}.get(args.with_reference)
    res = scout.search(args.query or "", args.image, args.limit, wr, args.category)
    for r in res.results:
        print(f"{r['n']:>2}. [{r['id']}] {r['title']}  ({', '.join(r['matched_by'])})")
    if res.preview_path:
        print(f"\nPreview: {res.preview_path}")
        if args.open:
            webbrowser.open(res.preview_path.resolve().as_uri())


def cmd_serve(args) -> None:
    from .server import run
    run()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="prompt-scout")
    sub = p.add_subparsers(dest="cmd", required=True)

    i0 = sub.add_parser("init", help=f"create the config file ({config_path()})")
    i0.add_argument("--force", action="store_true", help="overwrite an existing config")
    i0.set_defaults(func=cmd_init)

    f = sub.add_parser("fetch", help="download the prompt library from GitHub")
    f.add_argument("--dest")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("index", help="embed prompts and images (incremental)")
    i.add_argument("--references", help="folder with manifest.json and category files")
    i.add_argument("--embedder", choices=["gemini", "hash"], default="gemini")
    i.add_argument("--limit", type=int, help="index only the first N prompts (cheap trial)")
    i.add_argument("--no-images", action="store_true", help="text only, skip sample images")
    i.add_argument("--rebuild", action="store_true", help="ignore the previous index")
    i.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="search from the terminal")
    s.add_argument("query", nargs="?", default="")
    s.add_argument("--image", help="example image: file path or URL")
    s.add_argument("--limit", type=int, default=8)
    s.add_argument("--with-reference", choices=["yes", "no"])
    s.add_argument("--category")
    s.add_argument("--open", action="store_true", help="open the results page in the browser")
    s.set_defaults(func=cmd_search)

    sv = sub.add_parser("serve", help="run the MCP server over stdio")
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    if args.cmd == "search" and not args.query and not args.image:
        p.error("give a query, --image, or both")
    args.func(args)


if __name__ == "__main__":
    main()
