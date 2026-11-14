import json
from pathlib import Path

import pytest

from prompt_scout.dataset import load_records
from prompt_scout.embed import HashEmbedder
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
        # duplicate of id 2 in a second category
        {"id": "2", "title": "Cyberpunk neon street", "description": "Neon city at night",
         "content": "A person in a rainy cyberpunk street with pink and cyan neon signs",
         "sourceMedia": "['https://x/2.jpg']", "needReferenceImages": "False"},
        {"id": "4", "title": "Empty", "description": "", "content": "", "sourceMedia": "[]",
         "needReferenceImages": "False"},
    ],
}


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
    return Index.build(records, embedder)
