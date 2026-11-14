import io
import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from prompt_scout.dataset import load_records
from prompt_scout.embed import HashEmbedder
from prompt_scout.images import to_jpeg
from prompt_scout.index import Index

ITEMS = {
    "profile-avatar": [
        {"id": "1", "title": "Vintage film portrait", "description": "Warm 35mm film look portrait",
         "content": "Portrait of {argument name=\"person\" default=\"the woman from image 1\"} on Kodak Portra film, golden hour",
         "sourceMedia": "['https://x/1.jpg']", "needReferenceImages": "True"},
        {"id": "2", "title": "Cyberpunk neon street", "description": "Neon city at night",
         "content": "A person in a rainy cyberpunk street with pink and cyan neon signs",
         "sourceMedia": "['https://x/2.jpg']", "needReferenceImages": "False"},
    ],
    "poster-flyer": [
        {"id": "3", "title": "Ukiyo-e poster", "description": "Edo-period woodblock style",
         "content": "Ukiyo-e woodblock print of a modern Tokyo scene, flat colors",
         "sourceMedia": "['https://x/3.jpg']", "needReferenceImages": "False"},
        {"id": "2", "title": "Cyberpunk neon street", "description": "Neon city at night",
         "content": "A person in a rainy cyberpunk street with pink and cyan neon signs",
         "sourceMedia": "['https://x/2.jpg']", "needReferenceImages": "False"},
        {"id": "4", "title": "Empty", "description": "", "content": "", "sourceMedia": "[]",
         "needReferenceImages": "False"},
        {"id": "5", "title": "Text-only poster", "description": "no image", "content": "Minimal typographic poster",
         "sourceMedia": "[]", "needReferenceImages": "False"},
    ],
}

# Fake "sample images" for the hash embedder: their bytes are read as text,
# deliberately different from the prompt text, so we can tell image hits from text hits.
FAKE_IMAGES = {"1": b"warm grain sepia", "2": b"magenta glow wet asphalt", "3": b"wave mount fuji indigo"}


@pytest.fixture
def refs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "references"
    d.mkdir()
    cats = []
    for slug, items in ITEMS.items():
        (d / f"{slug}.json").write_text(json.dumps(items))
        cats.append({"slug": slug, "title": slug, "file": f"{slug}.json", "count": len(items)})
    (d / "manifest.json").write_text(json.dumps({"categories": cats}))
    return d


@pytest.fixture
def records(refs_dir):
    return load_records(refs_dir)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def index(records, embedder):
    return Index.build(records, embedder, load_image=lambda r: FAKE_IMAGES.get(r.id))


def make_png(color) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 80), color).save(buf, format="PNG")
    return buf.getvalue()


COLORS = {"1": (200, 150, 60), "2": (230, 0, 160), "3": (20, 40, 120)}


class ExactImageEmbedder(HashEmbedder):
    """Identical image bytes -> identical vector; different images -> unrelated vectors."""

    def embed_images(self, images):
        import hashlib
        return self.embed_texts([hashlib.md5(b).hexdigest() for b in images])


@pytest.fixture
def img_embedder():
    return ExactImageEmbedder()


@pytest.fixture
def real_image_env(tmp_path, records, img_embedder):
    embedder = img_embedder
    """Index whose image vectors come from real cached JPEGs, as in production."""
    cache = tmp_path / "images"
    cache.mkdir()
    sources = {}
    for rid, color in COLORS.items():
        png = make_png(color)
        sources[rid] = tmp_path / f"src-{rid}.png"
        sources[rid].write_bytes(png)
        (cache / f"{rid}.jpg").write_bytes(to_jpeg(png))
    idx = Index.build(records, embedder,
                      load_image=lambda r: (cache / f"{r.id}.jpg").read_bytes() if r.id in COLORS else None)
    return idx, cache, sources
