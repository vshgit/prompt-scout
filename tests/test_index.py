import numpy as np
import pytest

from prompt_scout.embed import HashEmbedder
from prompt_scout.index import Index


def test_finds_relevant_first(index, embedder):
    hits = index.search("neon cyberpunk night", embedder, limit=3)
    assert hits[0].record.id == "2"
    assert hits[0].dense_rank == 1 and hits[0].keyword_rank == 1


def test_reference_filter(index, embedder):
    hits = index.search("portrait", embedder, with_reference=True)
    assert [h.record.id for h in hits] == ["1"]
    hits = index.search("portrait", embedder, with_reference=False)
    assert "1" not in [h.record.id for h in hits]


def test_category_filter(index, embedder):
    hits = index.search("street", embedder, category="poster-flyer")
    assert {h.record.id for h in hits} <= {"2", "3"}


def test_no_match_in_filter_returns_empty(index, embedder):
    assert index.search("anything", embedder, category="nope") == []


def test_save_load_roundtrip(index, embedder, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert [r.id for r in loaded.records] == [r.id for r in index.records]
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.search("ukiyo-e", embedder)[0].record.id == "3"


def test_embedder_mismatch_is_rejected(index):
    with pytest.raises(ValueError):
        index.search("x", HashEmbedder(dim=64))


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.seen = 0

    def embed_documents(self, texts):
        self.seen += len(texts)
        return super().embed_documents(texts)


def test_incremental_build_reuses_unchanged(records, index):
    records[0].title = "Changed title"
    emb = CountingEmbedder()
    rebuilt = Index.build(records, emb, previous=index)
    assert emb.seen == 1  # only the changed record is re-embedded
    assert np.allclose(rebuilt.vectors[1:], index.vectors[1:])
