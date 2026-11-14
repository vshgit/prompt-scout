"""Gemini adapters against a fake client: checks request shapes built with the real SDK types."""

import json
from types import SimpleNamespace

import numpy as np

from prompt_scout.embed import GeminiEmbedder
from prompt_scout.llm import GeminiLLM


class FakeModels:
    def __init__(self, fail_first: int = 0):
        self.calls = []
        self.fail_first = fail_first

    def embed_content(self, model, contents, config):
        self.calls.append(("embed", model, list(contents), config))
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        vals = [[3.0, 4.0] + [0.0] * (config.output_dimensionality - 2) for _ in contents]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vals])

    def generate_content(self, model, contents, config):
        self.calls.append(("gen", model, contents, config))
        return SimpleNamespace(text=json.dumps({"style_description": "s", "search_query": "q"}))


def client(**kw):
    return SimpleNamespace(models=FakeModels(**kw))


def test_embedder_batches_normalizes_and_sets_task(monkeypatch):
    c = client()
    e = GeminiEmbedder(dim=8, batch_size=2, client=c)
    m = e.embed_documents(["a", "b", "c"])
    assert m.shape == (3, 8)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert [len(call[2]) for call in c.models.calls] == [2, 1]
    assert c.models.calls[0][3].task_type == "RETRIEVAL_DOCUMENT"
    e.embed_query("q")
    assert c.models.calls[-1][3].task_type == "RETRIEVAL_QUERY"


def test_embedder_retries_rate_limit(monkeypatch):
    monkeypatch.setattr("prompt_scout.embed.time.sleep", lambda s: None)
    c = client(fail_first=2)
    out = GeminiEmbedder(dim=4, client=c).embed_documents(["a"])
    assert out.shape == (1, 4) and len(c.models.calls) == 3


def test_llm_describe_image_sends_bytes_and_json_mode():
    c = client()
    out = GeminiLLM(model="m", client=c).describe_image(b"img", "image/png")
    assert out["search_query"] == "q"
    _, model, contents, config = c.models.calls[0]
    assert model == "m" and config.response_mime_type == "application/json"
    assert contents[0].inline_data.data == b"img"
