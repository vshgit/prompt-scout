import numpy as np
import pytest

from prompt_scout.embed import HashEmbedder
from prompt_scout.index import Index


def ids(hits):
    return [h.record.id for h in hits]


def test_text_query_finds_by_prompt_text(index, embedder):
    hits = index.search(embedder, query="neon cyberpunk night", limit=3)
    assert hits[0].record.id == "2"
    assert {"text->meaning", "text->keywords"} <= set(hits[0].ranks)


def test_text_query_can_match_by_look(index, embedder):
    # words that exist only in the sample image, not in any prompt text
    hits = index.search(embedder, query="magenta glow asphalt")
    assert hits[0].record.id == "2" and hits[0].ranks.get("text->look") == 1


def test_image_query_matches_sample_images(index, embedder):
    hits = index.search(embedder, image=b"wave fuji indigo")
    assert hits[0].record.id == "3" and hits[0].ranks.get("photo->look") == 1


def test_image_and_text_together(index, embedder):
    hits = index.search(embedder, query="ukiyo-e", image=b"warm sepia")
    assert {"1", "3"} <= set(ids(hits[:2]))


def test_records_without_image_never_in_look_list(index, embedder):
    hits = index.search(embedder, image=b"minimal typographic poster", limit=10)
    five = [h for h in hits if h.record.id == "5"]
    assert all("photo->look" not in h.ranks for h in five)


def test_filters(index, embedder):
    assert ids(index.search(embedder, query="portrait", with_reference=True)) == ["1"]
    assert set(ids(index.search(embedder, query="street", category="poster-flyer"))) <= {"2", "3", "5"}
    assert index.search(embedder, query="x", category="nope") == []


def test_empty_query_rejected(index, embedder):
    with pytest.raises(ValueError):
        index.search(embedder, query="  ")


def test_embedder_mismatch_is_rejected(index):
    with pytest.raises(ValueError):
        index.search(HashEmbedder(dim=64), query="x")


def test_save_load_roundtrip(index, embedder, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert ids(loaded.search(embedder, image=b"wave fuji")) [0] == "3"
    assert np.allclose(loaded.image_vectors, index.image_vectors)
    assert loaded.image_keys == index.image_keys


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.texts = self.images = 0

    def embed_texts(self, texts, query=False):
        self.texts += len(texts)
        return super().embed_texts(texts, query)

    def embed_images(self, images):
        self.images += len(images)
        return super().embed_images(images)


def test_incremental_build_reuses_unchanged(records, index):
    from tests.conftest import FAKE_IMAGES
    records[0].title = "Changed title"
    emb = CountingEmbedder()
    rebuilt = Index.build(records, emb, load_image=lambda r: FAKE_IMAGES.get(r.id), previous=index)
    assert emb.texts == 1 and emb.images == 0
    assert np.allclose(rebuilt.image_vectors, index.image_vectors)


def test_build_without_images(records, embedder):
    idx = Index.build(records, embedder, load_image=None)
    assert not idx.has_image.any()
    assert idx.search(embedder, query="ukiyo-e")[0].record.id == "3"
