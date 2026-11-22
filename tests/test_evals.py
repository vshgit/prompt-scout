"""The evaluation code: metrics, query file checks, ablation and the runner. No network."""

import json
import math

import pytest

from evals import run as evals_run
from evals.metrics import hit_at_k, mean_scores, ndcg_at_k, recall_at_k, reciprocal_rank, score_query
from evals.queries import parse
from tests.conftest import FAKE_IMAGES, make_png


# ---------- metrics, checked against hand-computed values ----------
RANKED = ["a", "x", "b", "y", "c"]
REL = {"a", "b", "z"}


def test_recall_and_hit():
    assert recall_at_k(RANKED, REL) == pytest.approx(2 / 3)
    assert hit_at_k(RANKED, REL) == 1.0
    assert recall_at_k(RANKED, REL, k=1) == pytest.approx(1 / 3)
    assert hit_at_k(["x", "y"], REL) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, REL) == 1.0
    assert reciprocal_rank(["x", "y", "b"], REL) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg_by_hand():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)                    # hits at ranks 1 and 3
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant ids, all first
    assert ndcg_at_k(RANKED, REL) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "b", "z"], REL) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], REL) == 0.0


def test_cutoff_ignores_anything_below_k():
    ranked = ["x"] * 10 + ["a"]
    assert recall_at_k(ranked, {"a"}) == 0.0 and reciprocal_rank(ranked, {"a"}) == 0.0
    assert ndcg_at_k(ranked, {"a"}) == 0.0
    assert recall_at_k(ranked, {"a"}, k=11) == 1.0


def test_ideal_ranking_is_capped_by_k():
    # 12 relevant ids, k=10: a perfect top 10 must score 1.0, not less
    rel = {str(i) for i in range(12)}
    assert ndcg_at_k([str(i) for i in range(10)], rel) == pytest.approx(1.0)


def test_no_relevant_ids_is_an_error():
    for fn in (recall_at_k, hit_at_k, reciprocal_rank, ndcg_at_k):
        with pytest.raises(ValueError):
            fn(["a"], set())


def test_mean_scores():
    per = [score_query(["a"], {"a"}), score_query(["x"], {"a"})]
    m = mean_scores(per)
    assert m["Recall@10"] == 0.5 and m["MRR@10"] == 0.5 and m["Hit@10"] == 0.5
    assert mean_scores([])["nDCG@10"] == 0.0


# ---------- query file ----------
def _line(**kw):
    base = {"id": "q1", "text": "neon", "image": None, "relevant": ["1", "2", "3"]}
    return json.dumps({**base, **kw})


def test_parse_valid_and_kinds():
    qs, p = parse([_line(), _line(id="q2", image="/x.jpg"), _line(id="q3", text="", image="/x.jpg"), ""])
    assert not p.errors and not p.warnings
    assert [q.kind for q in qs] == ["text", "mixed", "photo"]


@pytest.mark.parametrize("line, fragment", [
    ("{not json", "not valid JSON"),
    (_line(id=""), "missing id"),
    (_line(text=""), "needs text, image, or both"),
    (_line(relevant=[], status="verified"), "no relevant ids"),
    (_line(relevant=["1", "1", "2"]), "repeated ids"),
])
def test_parse_errors(line, fragment):
    qs, p = parse([line])
    assert qs == [] and any(fragment in e for e in p.errors)


def test_parse_duplicate_id_and_label_count_warning():
    qs, p = parse([_line(), _line()])
    assert len(qs) == 1 and any("duplicate" in e for e in p.errors)
    _, p = parse([_line(relevant=["1", "2"])])
    assert any("expected 3-10" in w for w in p.warnings)


def test_shipped_query_file_is_valid():
    from pathlib import Path
    import evals
    qs, p = parse((Path(evals.__file__).parent / "queries.jsonl").read_text("utf-8").splitlines())
    assert not p.errors and not p.warnings
    assert 25 <= len(qs) <= 30


# ---------- fusion split and ablation ----------
def test_search_equals_ranked_lists_plus_fuse(index, embedder):
    lists = index.ranked_lists(embedder, query="neon cyberpunk night", image=b"warm sepia")
    direct = index.search(embedder, query="neon cyberpunk night", image=b"warm sepia", limit=5)
    fused = index.fuse(lists, limit=5)
    assert [h.record.id for h in direct] == [h.record.id for h in fused]
    assert [h.score for h in direct] == [h.score for h in fused]


def test_zero_weight_switches_a_signal_off(index, embedder):
    lists = index.ranked_lists(embedder, query="magenta glow asphalt")
    assert "text->look" in {n for h in index.fuse(lists) for n in h.ranks}
    off = index.fuse(lists, weights={"text->look": 0.0})
    assert all("text->look" not in h.ranks for h in off)


def test_variants_cover_each_active_signal():
    v = evals_run.variants(set(evals_run.SIGNALS))
    assert evals_run.HYBRID in v and evals_run.DENSE in v
    assert v["− BM25"]["text->keywords"] == 0.0 and v["− BM25"]["text->meaning"] == 1.0
    assert v["only photo↔images"]["photo->look"] == 1.0
    assert sum(1 for w in v["only text↔prompts"].values() if w) == 1
    # only text signals active: no rows for photo signals
    names = " ".join(evals_run.variants({"text->look", "text->meaning", "text->keywords"}))
    assert "photo" not in names


def test_evaluate_and_ablation_on_toy_index(index, embedder):
    queries = {"a": "neon cyberpunk night", "b": "ukiyo-e woodblock"}
    relevant = {"a": {"2"}, "b": {"3"}}
    lists = {q: index.ranked_lists(embedder, t) for q, t in queries.items()}
    per = evals_run.evaluate(index, lists, relevant, evals_run.variants(set().union(*lists.values())))
    assert per[evals_run.HYBRID]["a"]["MRR@10"] == 1.0
    assert per["only BM25"]["b"]["Recall@10"] == 1.0
    # the sample image of prompt 2 shares no words with this query, so the look signal alone finds nothing
    assert per["only text↔images"]["a"]["Recall@10"] == 0.0


def test_paired_bootstrap_is_deterministic_and_brackets_the_mean():
    a, b = [1.0, 0.8, 0.9, 1.0, 0.7], [0.5, 0.6, 0.9, 0.4, 0.5]
    m1 = evals_run.paired_bootstrap(a, b)
    assert m1 == evals_run.paired_bootstrap(a, b)
    mean, lo, hi = m1
    assert mean == pytest.approx(0.3) and lo <= mean <= hi and lo > 0


def test_tune_prefers_defaults_on_ties_and_reports_loo(index, embedder):
    lists = {"a": index.ranked_lists(embedder, "neon cyberpunk night")}
    t = evals_run.tune(index, lists, {"a": {"2"}}, set(lists["a"]))
    assert t["best_in_sample"] == 1.0 and t["leave_one_out"] == 1.0
    assert t["default_in_sample"] == 1.0
    assert t["best_weights"] == {s: evals_run.WEIGHTS[s] for s in lists["a"]}  # closest to defaults


# ---------- the runner, end to end on a toy index ----------
def _write_queries(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")


def test_main_end_to_end_and_report_keeps_hand_written_text(index, tmp_path, capsys):
    index.save(tmp_path / "idx")
    img = tmp_path / "photo.png"
    img.write_bytes(make_png((230, 0, 160)))
    qfile = tmp_path / "q.jsonl"
    _write_queries(qfile, [
        {"id": "t1", "lang": "en", "text": "neon cyberpunk night", "relevant": ["2", "1", "3"]},
        {"id": "t2", "lang": "ru", "text": "ukiyo-e woodblock", "relevant": ["3", "5", "99999"]},
        {"id": "m1", "lang": "en", "text": "portrait", "image": str(img), "relevant": ["1", "2", "3"], "status": "verified"},
    ])
    report = tmp_path / "REPORT.md"
    report.write_text("# Mine\n\nbefore\n\n<!-- generated:begin -->\nold\n<!-- generated:end -->\n\nMy conclusions.\n", "utf-8")

    code = evals_run.main(["--queries", str(qfile), "--index", str(tmp_path / "idx"),
                           "--report", str(report), "--tune", "--json", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert "| Hybrid (all signals, default weights) |" in out and "RRF weight search" in out
    assert "Summary: 3 queries" in out and "1 labels verified" in out
    assert "t2:99999" in out  # a label missing from the index is reported, not counted as a miss
    text = report.read_text("utf-8")
    assert text.startswith("# Mine") and "My conclusions." in text and "old\n" not in text
    assert "Ablation" in text
    assert set(json.loads((tmp_path / "out.json").read_text())) >= {evals_run.HYBRID, evals_run.DENSE}


def test_main_rejects_bad_query_file_and_bad_photo(index, tmp_path, capsys):
    index.save(tmp_path / "idx")
    bad = tmp_path / "bad.jsonl"
    _write_queries(bad, [{"id": "x", "text": "", "relevant": ["1"]}])
    assert evals_run.main(["--queries", str(bad), "--index", str(tmp_path / "idx")]) == 2
    assert "needs text, image, or both" in capsys.readouterr().err

    gone = tmp_path / "gone.jsonl"
    _write_queries(gone, [{"id": "x", "text": "a", "image": str(tmp_path / "nope.jpg"), "relevant": ["1", "2", "3"]}])
    with pytest.raises(ValueError, match="File not found"):  # a broken photo path stops the run
        evals_run.main(["--queries", str(gone), "--index", str(tmp_path / "idx")])


def test_main_skips_query_whose_labels_are_all_missing(index, tmp_path, capsys):
    index.save(tmp_path / "idx")
    qfile = tmp_path / "q.jsonl"
    _write_queries(qfile, [{"id": "a", "text": "neon", "relevant": ["2", "1", "3"]},
                           {"id": "b", "text": "neon", "relevant": ["777", "888", "999"]}])
    assert evals_run.main(["--queries", str(qfile), "--index", str(tmp_path / "idx")]) == 0
    assert "b skipped" in capsys.readouterr().err


# ---------- labelling page ----------
def test_review_pool_has_labelled_first_and_every_signal(index, embedder):
    from evals.review import pool_for
    lists = index.ranked_lists(embedder, "neon cyberpunk night", image=b"wave fuji")
    pool = pool_for(index, lists, ["5", "missing-id"])
    assert pool[0] == ("5", ["labelled"])            # labelled first, unknown ids dropped
    reasons = " ".join(r for _, why in pool for r in why)
    assert "hybrid #1" in reasons and "BM25 #1" in reasons and "photo↔images #1" in reasons
    assert len({rid for rid, _ in pool}) == len(pool)  # each id once


def test_review_page_is_written(index, tmp_path, capsys):
    from evals import review
    index.save(tmp_path / "idx")
    qfile = tmp_path / "q.jsonl"
    _write_queries(qfile, [{"id": "a", "text": "neon </script>", "relevant": ["2", "1", "3"]}])
    out = tmp_path / "r.html"
    assert review.main(["--queries", str(qfile), "--index", str(tmp_path / "idx"),
                        "--images", str(tmp_path), "--out", str(out), "--no-open"]) == 0
    page = out.read_text("utf-8")
    assert "Скачать queries.jsonl" in page and "</script>\"" not in page and "__DATA__" not in page
    assert '"checked": true' in page


def test_draft_without_answers_is_kept_with_a_note():
    qs, p = parse([_line(relevant=[])])
    assert len(qs) == 1 and qs[0].relevant == [] and not p.errors
    assert any("not labelled yet" in w for w in p.warnings)


def test_main_skips_unlabelled_draft(index, tmp_path, capsys):
    index.save(tmp_path / "idx")
    qfile = tmp_path / "q.jsonl"
    _write_queries(qfile, [{"id": "a", "text": "neon", "relevant": ["2", "1", "3"]},
                           {"id": "b", "text": "neon", "relevant": []}])
    assert evals_run.main(["--queries", str(qfile), "--index", str(tmp_path / "idx")]) == 0
    assert "b skipped" in capsys.readouterr().err


# ---------- adding queries ----------
def test_add_appends_draft_rows_with_next_id(tmp_path, capsys):
    from evals import add
    qfile = tmp_path / "q.jsonl"
    _write_queries(qfile, [{"id": "q07", "text": "a", "relevant": ["1", "2", "3"]}])  # no trailing newline
    img = tmp_path / "p.png"
    img.write_bytes(make_png((10, 20, 30)))
    assert add.main(["--queries", str(qfile), "--image", str(img), "--text", "но зимой", "--lang", "ru"]) == 0
    assert add.main(["--queries", str(qfile), "--image", str(img)]) == 0
    rows = [json.loads(line) for line in qfile.read_text("utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q07", "q08", "q09"]
    assert rows[1] == {"id": "q08", "text": "но зимой", "image": str(img), "relevant": [],
                       "lang": "ru", "status": "draft", "note": ""}
    assert "mixed" in capsys.readouterr().out


def test_add_refuses_a_photo_that_cannot_be_read(tmp_path):
    from evals import add
    qfile = tmp_path / "q.jsonl"
    with pytest.raises(ValueError, match="File not found"):
        add.main(["--queries", str(qfile), "--image", str(tmp_path / "nope.jpg")])
    assert not qfile.exists()


def test_review_page_shows_photo_query_and_strips_helper_field(index, tmp_path):
    from evals import review
    index.save(tmp_path / "idx")
    img = tmp_path / "p.png"
    img.write_bytes(make_png((230, 0, 160)))
    qfile = tmp_path / "q.jsonl"
    _write_queries(qfile, [{"id": "p1", "text": "", "image": str(img), "relevant": []}])
    out = tmp_path / "r.html"
    assert review.main(["--queries", str(qfile), "--index", str(tmp_path / "idx"),
                        "--images", str(tmp_path), "--out", str(out), "--no-open"]) == 0
    page = out.read_text("utf-8")
    assert img.resolve().as_uri() in page and "delete row.imageSrc" in page
