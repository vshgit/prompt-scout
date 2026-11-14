"""Gemini adapters against a fake client: request shapes built with the real SDK types."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from prompt_scout.embed import GeminiEmbedder, embedder_for, HashEmbedder
from prompt_scout.llm import GeminiLLM, LLMError


class FakeModels:
    def __init__(self, fail_first=0, merge=False, gen_text=None, gen_error=False):
        self.calls, self.fail_first, self.merge = [], fail_first, merge
        self.gen_text = gen_text if gen_text is not None else json.dumps({"adapted_prompt": "p", "removed": ["woman"]})
        self.gen_error = gen_error

    def embed_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        # Like the real API: gemini-embedding-2 merges plain strings into ONE vector,
        # separate Content objects get separate vectors.
        plain = all(isinstance(c, str) for c in contents)
        n = 1 if self.merge or (plain and model == "gemini-embedding-2") else len(contents)
        vals = [[3.0, 4.0] + [0.0] * (config.output_dimensionality - 2)] * n
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vals])

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.gen_error:
            raise RuntimeError("404 NOT_FOUND models/x")
        return SimpleNamespace(text=self.gen_text, candidates=[SimpleNamespace(finish_reason="SAFETY")],
                               prompt_feedback=None)


def client(**kw):
    return SimpleNamespace(models=FakeModels(**kw))


def test_v2_texts_one_vector_each_with_prefixes():
    c = client()
    m = GeminiEmbedder(model="gemini-embedding-2", dim=8, text_batch=2, client=c).embed_texts(
        ["Film portrait\nwarm light", "b", "c"])
    assert m.shape == (3, 8) and np.allclose(np.linalg.norm(m, axis=1), 1)
    assert [len(x[1]) for x in c.models.calls] == [2, 1]
    first = c.models.calls[0]
    assert first[2].task_type is None
    assert first[1][0].parts[0].text == "title: Film portrait | text: warm light"


def test_v2_query_prefix():
    c = client()
    GeminiEmbedder(model="gemini-embedding-2", dim=4, client=c).embed_texts(["neon"], query=True)
    assert c.models.calls[0][1][0].parts[0].text == "task: search result | query: neon"


def test_v1_sends_plain_strings():
    c = client()
    GeminiEmbedder(model="gemini-embedding-001", dim=4, client=c).embed_texts(["a", "b"])
    assert c.models.calls[0][1] == ["a", "b"]


def test_task_type_kept_for_001():
    c = client()
    GeminiEmbedder(model="gemini-embedding-001", dim=4, client=c).embed_texts(["q"], query=True)
    assert c.models.calls[0][2].task_type == "RETRIEVAL_QUERY"


def test_each_image_is_its_own_content():
    c = client()
    out = GeminiEmbedder(dim=4, image_batch=2, client=c).embed_images([b"a", b"b", b"c"])
    assert out.shape == (3, 4)
    first = c.models.calls[0][1]
    assert len(first) == 2 and first[0].parts[0].inline_data.data == b"a"
    assert first[0].parts[0].inline_data.mime_type == "image/jpeg"


def test_merged_embedding_is_detected():
    with pytest.raises(RuntimeError):
        GeminiEmbedder(dim=4, client=client(merge=True)).embed_images([b"a", b"b"])


def test_retries_rate_limit(monkeypatch):
    monkeypatch.setattr("prompt_scout.embed.time.sleep", lambda s: None)
    c = client(fail_first=2)
    assert GeminiEmbedder(dim=4, client=c).embed_texts(["a"]).shape == (1, 4)
    assert len(c.models.calls) == 3


def test_llm_adapt_structured_output():
    c = client()
    out = GeminiLLM(model="m", client=c).adapt("A young woman", keep="pink hair")
    assert out["adapted_prompt"] == "p" and out["removed"] == ["woman"] and out["kept_as_style"] == []
    _, contents, config = c.models.calls[0]
    assert config.response_mime_type == "application/json" and config.response_schema is not None
    assert "pink hair" in contents


@pytest.mark.parametrize("kw, text", [
    ({"gen_error": True}, "request failed"),
    ({"gen_text": ""}, "no text"),
    ({"gen_text": "not json"}, "malformed"),
])
def test_llm_errors_are_readable(kw, text):
    with pytest.raises(LLMError, match=text):
        GeminiLLM(model="m", client=client(**kw)).adapt("x")


def test_embedder_for_restores_from_index_name():
    assert isinstance(embedder_for("hash@256"), HashEmbedder)
