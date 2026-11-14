"""Sample images: download once, shrink, cache on disk as JPEG.

Cached files feed three things: image embeddings, thumbnails in chat, and the HTML preview.
"""

from __future__ import annotations

import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image as PILImage

MAX_SIDE = 640
THUMB_SIDE = 256

Fetcher = Callable[[str], bytes]


def http_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "prompt-scout"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read(30 * 1024 * 1024)


def to_jpeg(data: bytes, max_side: int = MAX_SIDE, quality: int = 85) -> bytes:
    img = PILImage.open(io.BytesIO(data))
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def cache_path(cache_dir: Path, record_id: str) -> Path:
    return cache_dir / f"{record_id}.jpg"


def download_all(items: Iterable[tuple[str, str]], cache_dir: Path, fetch: Fetcher = http_fetch,
                 workers: int = 8, progress: Callable[[int, int], None] | None = None) -> dict[str, Path]:
    """items: (record_id, url). Returns record_id -> cached path for every image available.
    Already cached files are not downloaded again; failures are skipped."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = list(items)
    result: dict[str, Path] = {}
    todo = []
    for rid, url in items:
        p = cache_path(cache_dir, rid)
        if p.exists():
            result[rid] = p
        else:
            todo.append((rid, url))

    def one(item: tuple[str, str]) -> tuple[str, Path | None]:
        rid, url = item
        try:
            data = to_jpeg(fetch(url))
        except Exception:
            return rid, None
        p = cache_path(cache_dir, rid)
        p.write_bytes(data)
        return rid, p

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rid, p in pool.map(one, todo):
            done += 1
            if p is not None:
                result[rid] = p
            if progress and (done % 200 == 0 or done == len(todo)):
                progress(done, len(todo))
    return result


def thumbnail(path: Path) -> bytes:
    return to_jpeg(path.read_bytes(), THUMB_SIDE, quality=80)
