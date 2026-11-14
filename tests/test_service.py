import pytest

from prompt_scout.llm import adapt_user_message
from prompt_scout.service import PromptScout


class FakeLLM:
    def __init__(self):
        self.adapt_calls = []

    def describe_image(self, image, mime_type):
        assert image == b"\x89PNG fake" and mime_type == "image/png"
        return {"style_description": "Neon night street.", "search_query": "cyberpunk neon rainy street"}

    def adapt(self, record, wish):
        self.adapt_calls.append((record.id, wish))
        return {"adapted_prompt": "Keep the person from reference image 1 ...", "changes": ["убрано описание лица"],
                "extra_references": [], "warnings": []}


@pytest.fixture
def scout(index, embedder):
    return PromptScout(index, embedder, FakeLLM())


def test_search_returns_preview_and_image(scout):
    res = scout.search("ukiyo-e woodblock", limit=2)
    top = res[0]
    assert top["id"] == "3" and top["image"] == "https://x/3.jpg"
    assert set(top["matched_by"]) == {"meaning", "keywords"}
    assert "{argument" not in scout.search("film portrait")[0]["prompt_preview"]


def test_limit_is_clamped(scout):
    assert len(scout.search("street", limit=0)) == 1


def test_search_by_image_uses_description(scout, tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG fake")
    out = scout.search_by_image(str(img), extra="at night")
    assert out["search_query"].endswith("at night")
    assert out["results"][0]["id"] == "2"


def test_search_by_image_rejects_non_image(scout, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    with pytest.raises(ValueError):
        scout.search_by_image(str(f))


def test_adapt_passes_wish_and_attribution(scout):
    out = scout.adapt("1", "зелёные глаза")
    assert scout.llm.adapt_calls == [("1", "зелёные глаза")]
    assert out["source_url"].endswith("id=1") and "CC BY" in out["license"]


def test_unknown_id(scout):
    with pytest.raises(ValueError):
        scout.get_prompt("999")


def test_llm_required_for_image_and_adapt(index, embedder):
    s = PromptScout(index, embedder, llm=None)
    with pytest.raises(RuntimeError):
        s.adapt("1")


def test_adapt_message_fills_placeholders(records):
    msg = adapt_user_message(records[0], "")
    assert "the woman from image 1" in msg and "(none)" in msg


def test_categories(scout):
    assert scout.categories() == {"profile-avatar": 2, "poster-flyer": 2}
