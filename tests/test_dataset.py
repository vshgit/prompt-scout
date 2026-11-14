from prompt_scout.dataset import fill_placeholders


def test_merges_duplicates_and_skips_empty(records):
    ids = [r.id for r in records]
    assert ids == ["1", "2", "3", "5"]
    neon = next(r for r in records if r.id == "2")
    assert neon.categories == ["profile-avatar", "poster-flyer"]


def test_parses_flags_and_media(records):
    film = records[0]
    assert film.needs_reference is True
    assert film.images == ["https://x/1.jpg"]
    assert film.source_url.endswith("id=1")


def test_fill_placeholders():
    text = 'A {argument name="person" default="the woman from image 1"} smiling'
    assert fill_placeholders(text) == "A the woman from image 1 smiling"
