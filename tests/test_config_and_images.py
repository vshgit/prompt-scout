import pytest

from prompt_scout.config import DEFAULT_CONFIG, Config
from prompt_scout.dataset import select
from prompt_scout.images import download_all, to_jpeg
from tests.conftest import make_png


def test_default_config_is_portraits_only():
    cfg = Config.from_toml(DEFAULT_CONFIG)
    assert cfg.categories == ["profile-avatar"] and cfg.images and cfg.exclude_ids == []


def test_unknown_category_is_an_error():
    with pytest.raises(ValueError):
        Config.from_toml('[library]\ncategories = ["portraits"]')


def test_select_by_category_and_exclusion(records):
    assert [r.id for r in select(records, ["profile-avatar"], [])] == ["1", "2"]
    assert [r.id for r in select(records, ["profile-avatar"], ["2"])] == ["1"]
    assert [r.id for r in select(records, ["poster-flyer"], [])] == ["2", "3", "5"]


def test_download_caches_and_skips_failures(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "bad" in url:
            raise OSError("404")
        return make_png((10, 20, 30))

    got = download_all([("1", "https://x/1.png"), ("2", "https://x/bad.png")], tmp_path, fetch=fetch, workers=2)
    assert set(got) == {"1"} and got["1"].read_bytes()[:2] == b"\xff\xd8"
    download_all([("1", "https://x/1.png")], tmp_path, fetch=fetch)
    assert calls.count("https://x/1.png") == 1  # second run uses the cache


def test_to_jpeg_shrinks():
    from io import BytesIO
    from PIL import Image
    big = BytesIO()
    Image.new("RGB", (2000, 1000)).save(big, format="PNG")
    out = Image.open(BytesIO(to_jpeg(big.getvalue())))
    assert max(out.size) == 640
