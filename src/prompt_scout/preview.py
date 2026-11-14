"""HTML contact sheet for search results: numbered cards, sample images, copy buttons.

Written to ~/.prompt-scout/results/, images are referenced from the local cache
(or the original URL if not cached), so the page works offline.
"""

from __future__ import annotations

import base64
import html
import json
import os
from datetime import datetime
from pathlib import Path

MATCH_LABELS = {
    "photo->look": "фото ↔ картинка",
    "photo->meaning": "фото ↔ промпт",
    "text->look": "текст ↔ картинка",
    "text->meaning": "текст ↔ промпт",
    "text->keywords": "слова",
}

CSS = """
:root{color-scheme:light dark;--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--muted:#6b6a66;--line:#e4e2dc;--accent:#2f5d50}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--ink:#eceae4;--muted:#9d9b95;--line:#33322f;--accent:#7fb8a4}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:28px 32px 8px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
header img{width:96px;height:96px;object-fit:cover;border-radius:10px;border:1px solid var(--line)}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:20px;padding:20px 32px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.imgwrap{position:relative;aspect-ratio:4/5;background:var(--line)}
.imgwrap img{width:100%;height:100%;object-fit:cover;display:block}
.num{position:absolute;top:10px;left:10px;background:var(--ink);color:var(--bg);font-weight:700;border-radius:999px;min-width:32px;height:32px;display:grid;place-items:center;padding:0 8px}
.body{padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px;flex:1}
.title{font-weight:600}.id{color:var(--muted);font-size:12px}.id a{color:var(--accent)}
.desc{color:var(--muted);font-size:13px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.chips{display:flex;gap:6px;flex-wrap:wrap}.chip{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--muted)}
.chip.ref{border-color:var(--accent);color:var(--accent)}
.actions{display:flex;gap:8px;margin-top:auto;flex-wrap:wrap}
button,a.btn{font:inherit;font-size:13px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:8px;padding:6px 10px;cursor:pointer;text-decoration:none}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--ink);color:var(--bg);padding:8px 14px;border-radius:8px;opacity:0;transition:.2s}
.toast.on{opacity:1}
"""

JS = """
function copyText(t){
  const done=()=>{const el=document.getElementById('toast');el.classList.add('on');setTimeout(()=>el.classList.remove('on'),1200)};
  if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(t).then(done).catch(()=>fallback(t,done))}else fallback(t,done)
}
function fallback(t,done){const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();
  try{document.execCommand('copy');done()}catch(e){}a.remove()}
"""


def _img_src(rec: dict, cache_dir: Path, page_dir: Path) -> str:
    cached = cache_dir / f"{rec['id']}.jpg"
    if cached.exists():
        return html.escape(os.path.relpath(cached, page_dir))
    return html.escape(rec.get("image") or "")


def render(results: list[dict], prompts: dict[str, str], query: str, query_image: bytes | None,
           cache_dir: Path, out_dir: Path) -> Path:
    """results: dicts from service.hit_to_dict; prompts: id -> full prompt text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"search-{stamp}.html"

    head_img = ""
    if query_image:
        head_img = f'<img src="data:image/jpeg;base64,{base64.b64encode(query_image).decode()}" alt="">'
    title = html.escape(query) if query else "Поиск по фото"

    cards = []
    for n, r in enumerate(results, 1):
        chips = "".join(f'<span class="chip">{MATCH_LABELS.get(m, m)}</span>' for m in r["matched_by"])
        if r["needs_reference"]:
            chips = '<span class="chip ref">с фото-референсом</span>' + chips
        cards.append(f"""
<article class="card">
  <div class="imgwrap"><img loading="lazy" src="{_img_src(r, cache_dir, out_dir)}" alt=""><div class="num">{n}</div></div>
  <div class="body">
    <div class="title">{html.escape(r['title'])}</div>
    <div class="id">id {html.escape(r['id'])} · {html.escape(', '.join(r['categories']))} · <a href="{html.escape(r['source_url'])}" target="_blank" rel="noopener">источник</a></div>
    <div class="desc">{html.escape(r['description'])}</div>
    <div class="chips">{chips}</div>
    <div class="actions">
      <button class="primary" onclick='copyText({json.dumps(prompts.get(r["id"], ""))})'>Копировать промпт</button>
      <button onclick='copyText({json.dumps("#" + str(n) + " id " + r["id"])})'>№ и id</button>
    </div>
  </div>
</article>""")

    page = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Prompt Scout — {title}</title>
<style>{CSS}</style></head><body>
<header>{head_img}<div><h1>{title}</h1>
<div class="sub">{len(results)} вариантов · {datetime.now():%d.%m.%Y %H:%M} · скажите в чате номер, например «№2 — адаптируй»</div></div></header>
<main>{''.join(cards)}</main><div id="toast" class="toast">Скопировано</div>
<script>{JS}</script></body></html>"""
    path.write_text(page, "utf-8")
    (out_dir / "latest.html").write_text(page, "utf-8")  # same folder, so relative paths still work
    return path
