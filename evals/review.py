"""Labelling page: check and extend the correct answers of every query by eye.

    python -m evals.review            # writes evals/review.html and opens it

For every query the page shows a pool of candidates with their sample images:
the ids already labelled (pre-checked), the hybrid top 10, and the top 5 of each
signal on its own. Pooling the output of several systems (as TREC does) keeps the
labels from favouring one signal. Tick what really fits, mark the query as checked,
and press "Download queries.jsonl": the file has the same rows with the new
`relevant` ids and `status: "verified"` for checked queries. Put it in place of
evals/queries.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

from prompt_scout.index import Index

from .queries import MAX_RELEVANT, MIN_RELEVANT, load
from .run import NAMES, SIGNALS

TOP_HYBRID, TOP_SIGNAL = 10, 5


def pool_for(index: Index, lists: dict[str, list[int]], labelled: list[str]) -> list[tuple[str, list[str]]]:
    """Candidate ids with the reasons they are in the pool, labelled ones first."""
    why: dict[str, list[str]] = {rid: ["labelled"] for rid in labelled if rid in index.by_id}
    for pos, h in enumerate(index.fuse(lists, limit=TOP_HYBRID), 1):
        why.setdefault(h.record.id, []).append(f"hybrid #{pos}")
    for s in SIGNALS:
        for pos, i in enumerate(lists.get(s, [])[:TOP_SIGNAL], 1):
            why.setdefault(index.records[i].id, []).append(f"{NAMES[s]} #{pos}")
    return list(why.items())


def card(index: Index, rid: str, reasons: list[str], checked: bool, cache: Path) -> dict:
    r = index.records[index.by_id[rid]]
    img = cache / f"{rid}.jpg"
    return {"id": rid, "title": r.title, "description": r.description, "prompt": r.prompt[:1500],
            "image": img.resolve().as_uri() if img.exists() else (r.images[0] if r.images else ""),
            "why": reasons, "checked": checked, "url": r.source_url}


PAGE = """<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разметка запросов</title>
<style>
:root{color-scheme:light dark;--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e4e2dc;--ok:#2f7d55;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--ink:#eceae4;--muted:#9d9b95;--line:#33322f;--ok:#7fc8a0;--warn:#e0a060}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--ink)}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.bar button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--ink);background:var(--ink);color:var(--bg);cursor:pointer}
.muted{color:var(--muted);font-size:13px}
section{padding:20px 24px;border-bottom:1px solid var(--line)}
h2{font-size:18px;margin:0 0 4px}.qhead{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
.count{font-size:13px}.count.bad{color:var(--warn)}.count.good{color:var(--ok)}
label.done{font-size:14px;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-top:12px}
.c{background:var(--card);border:2px solid var(--line);border-radius:12px;overflow:hidden;cursor:pointer;display:flex;flex-direction:column}
.c.on{border-color:var(--ok)}.c img{width:100%;aspect-ratio:4/5;object-fit:cover;display:block;background:var(--line)}
.c .b{padding:8px 10px;font-size:13px;display:flex;flex-direction:column;gap:4px}
.c .t{font-weight:600}.c .why{color:var(--muted);font-size:11px}
.c details{font-size:12px;color:var(--muted)}.c details p{white-space:pre-wrap;max-height:220px;overflow:auto;margin:4px 0}
.c .tick{font-weight:700;color:var(--ok)}
</style>
<div class="bar"><button id="dl">Скачать queries.jsonl</button>
<span id="progress" class="muted"></span>
<span class="muted">Клик по карточке = подходит / не подходит. Нужно от __MIN__ до __MAX__ отмеченных.</span></div>
<div id="root"></div>
<script>
const DATA = __DATA__;
const KEY = "prompt-scout-review-v1";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
for (const q of DATA) {
  const s = saved[q.query.id];
  if (s) { q.done = s.done; for (const c of q.cards) c.checked = s.ids.includes(c.id); }
}
function save() {
  const s = {};
  for (const q of DATA) s[q.query.id] = {done: q.done, ids: q.cards.filter(c => c.checked).map(c => c.id)};
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
}
const esc = t => String(t ?? "").replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
function render() {
  const root = document.getElementById("root"); root.innerHTML = "";
  let done = 0;
  DATA.forEach((q, qi) => {
    if (q.done) done++;
    const n = q.cards.filter(c => c.checked).length;
    const okN = n >= __MIN__ && n <= __MAX__;
    const sec = document.createElement("section");
    sec.innerHTML = `<div class="qhead"><h2>${esc(q.query.id)} · ${esc(q.query.text || "(фото)")}</h2>
      <span class="count ${okN ? "good" : "bad"}">отмечено ${n}</span>
      <label class="done"><input type="checkbox" ${q.done ? "checked" : ""}> проверено</label></div>
      ${q.query.image ? `<div class="muted">фото: ${esc(q.query.image)}</div>` : ""}
      <div class="grid"></div>`;
    sec.querySelector(".done input").onchange = e => { q.done = e.target.checked; save(); render(); };
    const grid = sec.querySelector(".grid");
    q.cards.forEach(c => {
      const el = document.createElement("div");
      el.className = "c" + (c.checked ? " on" : "");
      el.innerHTML = `<img loading="lazy" src="${esc(c.image)}" alt="">
        <div class="b"><div>${c.checked ? '<span class="tick">✓ </span>' : ""}<span class="t">${esc(c.title)}</span></div>
        <div class="muted">id ${esc(c.id)} · <a href="${esc(c.url)}" target="_blank">источник</a></div>
        <div class="why">${esc(c.why.join(", "))}</div>
        <details><summary>промпт</summary><p>${esc(c.prompt)}</p></details></div>`;
      el.onclick = e => { if (e.target.closest("details,a")) return; c.checked = !c.checked; save(); render(); };
      grid.appendChild(el);
    });
    root.appendChild(sec);
  });
  document.getElementById("progress").textContent = `проверено ${done} из ${DATA.length}`;
}
document.getElementById("dl").onclick = () => {
  const lines = DATA.map(q => {
    const row = {...q.query};
    if (q.done) { row.relevant = q.cards.filter(c => c.checked).map(c => c.id); row.status = "verified"; }
    return JSON.stringify(row);
  });
  const blob = new Blob([lines.join("\\n") + "\\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "queries.jsonl"; a.click();
};
render();
</script></html>"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.review", description=__doc__.split("\n\n")[0])
    p.add_argument("--queries", type=Path, default=Path(__file__).with_name("queries.jsonl"))
    p.add_argument("--index", type=Path, help="index folder (default: the configured one)")
    p.add_argument("--images", type=Path, help="sample image cache (default: the configured one)")
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("review.html"))
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    from prompt_scout.config import images_dir, index_dir
    from prompt_scout.embed import embedder_for
    from prompt_scout.service import read_image

    queries, problems = load(args.queries)
    if problems.errors:
        print("Query file has errors:\n  " + "\n  ".join(problems.errors), file=sys.stderr)
        return 2
    index = Index.load(args.index or index_dir())
    embedder = embedder_for(index.embedder_name)
    cache = args.images or images_dir()

    data = []
    for q in queries:
        lists = index.ranked_lists(embedder, q.text, read_image(q.image) if q.image else None)
        cards = [card(index, rid, why, rid in q.relevant, cache) for rid, why in pool_for(index, lists, q.relevant)]
        data.append({"query": asdict(q), "done": q.status == "verified", "cards": cards})

    page = (PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__MIN__", str(MIN_RELEVANT)).replace("__MAX__", str(MAX_RELEVANT)))
    args.out.write_text(page, "utf-8")
    total = sum(len(d["cards"]) for d in data)
    print(f"{len(data)} queries, {total} cards -> {args.out}")
    if not args.no_open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
