import pytest

from prompt_scout.llm import adapt_user_message
from prompt_scout.service import PromptScout


class FakeLLM:
    def __init__(self):
        self.calls = []

    def adapt(self, prompt, wish="", keep="", needs_reference=False):
        self.calls.append((prompt, wish, keep, needs_reference))
        return {"adapted_prompt": "The person from the reference photos ...", "removed": ["woman"],
                "kept_as_style": [], "changes": [], "extra_references": [], "warnings": []}


@pytest.fixture
def scout(real_image_env, img_embedder, tmp_path):
    idx, cache, _ = real_image_env
    return PromptScout(idx, img_embedder, FakeLLM(), cache_dir=cache, results_dir=tmp_path / "results")


def test_photo_search_finds_same_look_first(scout, real_image_env):
    _, _, sources = real_image_env
    res = scout.search(image=str(sources["2"]), limit=3)
    assert res.results[0]["id"] == "2"
    assert res.results[0]["n"] == 1 and "photo->look" in res.results[0]["matched_by"]


def test_results_are_numbered_and_have_preview(scout):
    res = scout.search("ukiyo-e woodblock neon", limit=2)
    assert [r["n"] for r in res.results] == [1, 2]
    assert res.preview_path.exists() and (res.preview_path.parent / "latest.html").exists()
    page = res.preview_path.read_text()
    assert "Ukiyo-e poster" in page and "../images/3.jpg" in page
    assert "Ukiyo-e woodblock print of a modern Tokyo scene" in page  # full prompt for the copy button


def test_placeholders_filled_in_preview(scout):
    res = scout.search("film portrait golden hour", limit=1)
    assert "{argument" not in res.results[0]["prompt_preview"]


def test_thumbnails_only_for_cached(scout):
    res = scout.search("poster", limit=5)
    thumbs = scout.thumbnails(res.results)
    assert {n for n, _ in thumbs} <= {r["n"] for r in res.results if r["id"] in {"1", "2", "3"}}
    assert all(b[:2] == b"\xff\xd8" for _, b in thumbs)  # JPEG


def test_rejects_non_image(scout, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    with pytest.raises(ValueError):
        scout.search(image=str(f))
    with pytest.raises(ValueError):
        scout.search(image=str(tmp_path / "missing.png"))


def test_adapt_library_prompt_passes_wish_keep_and_attribution(scout):
    out = scout.adapt("1", wish="horizontal", keep="pink hair")
    prompt, wish, keep, needs_ref = scout.llm.calls[0]
    assert "Kodak Portra" in prompt and (wish, keep, needs_ref) == ("horizontal", "pink hair", True)
    assert out["source_url"].endswith("id=1") and "CC BY" in out["license"]
    assert out["removed"] == ["woman"]


def test_adapt_own_prompt_text(scout):
    out = scout.adapt(prompt_text="A young woman with long dark hair, film flash")
    assert scout.llm.calls[0][0].startswith("A young woman") and out["source_id"] is None


def test_adapt_needs_id_or_text(scout):
    with pytest.raises(ValueError):
        scout.adapt()


def test_adapt_needs_llm(real_image_env, img_embedder):
    idx, _, _ = real_image_env
    with pytest.raises(RuntimeError):
        PromptScout(idx, img_embedder).adapt("1")


def test_unknown_id(scout):
    with pytest.raises(ValueError):
        scout.get_prompt("999")


def test_adapt_message_fills_placeholders_and_marks_empty(records):
    msg = adapt_user_message(records[0].prompt)
    assert "the woman from image 1" in msg and "(none)" in msg and "(nothing)" in msg
