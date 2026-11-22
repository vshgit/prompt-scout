"""Evaluate search quality on the labelled query set.

    python -m evals.run                       # metrics table + ablation, to the terminal
    python -m evals.run --report evals/REPORT.md   # also (re)write the generated part of the report
    python -m evals.run --tune                # also search RRF weights (with leave-one-out check)

Needs the real index (`prompt-scout index`) and GEMINI_API_KEY for query embeddings.
Each query is embedded once; every variant below re-fuses the same ranked lists, so the
ablation costs no extra API calls.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from datetime import date
from pathlib import Path

from prompt_scout.index import WEIGHTS, Index

from .metrics import METRICS, mean_scores, score_query
from .queries import Query, load

SIGNALS = ["photo->look", "photo->meaning", "text->look", "text->meaning", "text->keywords"]
NAMES = {
    "photo->look": "photo↔images", "photo->meaning": "photo↔prompts",
    "text->look": "text↔images", "text->meaning": "text↔prompts", "text->keywords": "BM25",
}
HYBRID = "Hybrid (all signals, default weights)"
DENSE = "Dense only (no BM25)"
K = 10
BEGIN, END = "<!-- generated:begin -->", "<!-- generated:end -->"

Lists = dict[str, list[int]]


def variants(active: set[str]) -> dict[str, dict[str, float]]:
    """Weight sets to compare: full hybrid, each signal removed, dense only, each signal alone."""
    signals = [s for s in SIGNALS if s in active]
    out = {HYBRID: dict(WEIGHTS)}
    if len(signals) > 1:
        for s in signals:
            out[f"− {NAMES[s]}"] = {**WEIGHTS, s: 0.0}
    if "text->keywords" in active:
        out[DENSE] = {**WEIGHTS, "text->keywords": 0.0}
    if len(signals) > 1:
        for s in signals:
            out[f"only {NAMES[s]}"] = {k: (WEIGHTS[k] if k == s else 0.0) for k in WEIGHTS}
    return out


def rank_ids(index: Index, lists: Lists, weights: dict[str, float]) -> list[str]:
    return [h.record.id for h in index.fuse(lists, limit=K, weights=weights)]


def evaluate(index: Index, lists_by_query: dict[str, Lists], relevant: dict[str, set[str]],
             weight_sets: dict[str, dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    """variant -> query id -> metric -> value."""
    return {name: {qid: score_query(rank_ids(index, lists, w), relevant[qid], K)
                   for qid, lists in lists_by_query.items()}
            for name, w in weight_sets.items()}


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over queries with a 95% bootstrap interval (resampling queries)."""
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


# ---------- weight search ----------
def tune(index: Index, lists_by_query: dict[str, Lists], relevant: dict[str, set[str]], active: set[str],
         grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0), metric: str = "nDCG@10") -> dict:
    """Grid search over per-signal weights, plus a leave-one-query-out estimate.

    With a few dozen queries the best in-sample weights are optimistic; the leave-one-out
    score (weights chosen without the query being scored) is the honest number.
    """
    signals = [s for s in SIGNALS if s in active]
    qids = list(lists_by_query)
    combos = [dict(zip(signals, c)) for c in itertools.product(grid, repeat=len(signals))]
    combos = [c for c in combos if any(c.values())]

    def dist(c):  # ties go to the combination closest to the current defaults
        return sum(abs(c[s] - WEIGHTS[s]) for s in signals)

    table = []  # per combo: metric value for each query
    for c in combos:
        w = {**{k: 0.0 for k in WEIGHTS}, **c}
        table.append([score_query(rank_ids(index, lists_by_query[q], w), relevant[q], K)[metric] for q in qids])

    n = len(qids)
    totals = [sum(row) for row in table]
    best_i = max(range(len(combos)), key=lambda i: (totals[i], -dist(combos[i])))
    loo = 0.0
    for held in range(n):
        pick = max(range(len(combos)), key=lambda i: (totals[i] - table[i][held], -dist(combos[i])))
        loo += table[pick][held]
    default_i = next((i for i, c in enumerate(combos) if all(c[s] == WEIGHTS[s] for s in signals)), None)
    return {
        "metric": metric, "grid": list(grid), "combos": len(combos),
        "best_weights": combos[best_i], "best_in_sample": totals[best_i] / n,
        "default_in_sample": None if default_i is None else totals[default_i] / n,
        "leave_one_out": loo / n,
    }


# ---------- formatting ----------
def _f(x: float) -> str:
    return f"{x:.3f}"


def table(rows: dict[str, dict[str, float]], baseline: str | None = None) -> str:
    cols = list(METRICS)
    head = "| Variant | " + " | ".join(cols) + (" | Δ nDCG@10 |" if baseline else " |")
    sep = "|---|" + "---:|" * len(cols) + ("---:|" if baseline else "")
    lines = [head, sep]
    for name, m in rows.items():
        delta = f" {m['nDCG@10'] - rows[baseline]['nDCG@10']:+.3f} |" if baseline else ""
        lines.append(f"| {name} | " + " | ".join(_f(m[c]) for c in cols) + " |" + delta)
    return "\n".join(lines)


def build_report(index: Index, queries: list[Query], per: dict, dropped: list[str]) -> tuple[str, str]:
    """Returns (markdown, one-line summary for the CV)."""
    qids = [q.id for q in queries]
    by_id = {q.id: q for q in queries}
    overall = {name: mean_scores([per[name][q] for q in qids]) for name in per}
    out = [f"Index: `{index.embedder_name}`, {len(index.records)} prompts "
           f"({int(index.has_image.sum())} with sample images). Queries: {len(queries)} "
           f"({sum(q.kind == 'text' for q in queries)} text, {sum(q.kind == 'photo' for q in queries)} photo, "
           f"{sum(q.kind == 'mixed' for q in queries)} mixed), "
           f"{sum(q.status == 'verified' for q in queries)} labels verified by hand, "
           f"{sum(q.status != 'verified' for q in queries)} draft. Run on {date.today().isoformat()}.", ""]
    if dropped:
        out += [f"Labelled ids missing from this index (ignored): {', '.join(dropped)}.", ""]
    out += ["### Ablation (all queries)", "", table(overall, HYBRID), ""]
    photo_q = [q for q in qids if by_id[q].image]
    if photo_q and len(photo_q) < len(qids):
        # photo signals exist only for queries with a photo: averaged over all queries their rows look tiny
        rows = {n: mean_scores([per[n][q] for q in photo_q]) for n in per
                if n == HYBRID or "photo" in n or n.startswith("− ")}
        out += [f"### Ablation, photo and mixed queries only (n = {len(photo_q)})", "",
                "Text queries have no photo lists, so the photo rows of the table above average in zeros; "
                "this table compares on the queries where photo signals exist.", "",
                table(rows, HYBRID), ""]

    for title, key in (("language", lambda q: q.lang), ("query type", lambda q: q.kind)):
        groups = sorted({key(by_id[q]) for q in qids})
        if len(groups) < 2:
            continue
        out += [f"### Hybrid vs dense only, by {title}", "",
                "| Group | n | Hybrid Recall@10 | Hybrid nDCG@10 | Dense Recall@10 | Dense nDCG@10 |", "|---|---:|---:|---:|---:|---:|"]
        for g in groups:
            ids = [q for q in qids if key(by_id[q]) == g]
            h, d = mean_scores([per[HYBRID][q] for q in ids]), mean_scores([per[DENSE][q] for q in ids]) if DENSE in per else None
            out.append(f"| {g} | {len(ids)} | {_f(h['Recall@10'])} | {_f(h['nDCG@10'])} | "
                       + (f"{_f(d['Recall@10'])} | {_f(d['nDCG@10'])} |" if d else "– | – |"))
        out.append("")

    h = overall[HYBRID]
    summary = f"{len(queries)} queries, Recall@10 = {h['Recall@10']:.2f}, MRR@10 = {h['MRR@10']:.2f}, nDCG@10 = {h['nDCG@10']:.2f}"
    if DENSE in per:
        mean_d, lo, hi = paired_bootstrap([per[HYBRID][q]["nDCG@10"] for q in qids], [per[DENSE][q]["nDCG@10"] for q in qids])
        rec_d, rlo, rhi = paired_bootstrap([per[HYBRID][q]["Recall@10"] for q in qids], [per[DENSE][q]["Recall@10"] for q in qids])
        out += [f"Hybrid minus dense only: Recall@10 {rec_d:+.3f} (95% CI {rlo:+.3f} … {rhi:+.3f}), "
                f"nDCG@10 {mean_d:+.3f} (95% CI {lo:+.3f} … {hi:+.3f}); paired bootstrap over queries.", ""]
        summary += f"; hybrid vs dense-only: Recall@10 {rec_d:+.2f}, nDCG@10 {mean_d:+.2f}"
    return "\n".join(out), summary


def write_report(path: Path, generated: str, tuning: str = "") -> None:
    """Replace only the generated block, so hand-written conclusions survive re-runs."""
    block = f"{BEGIN}\n{generated}\n{tuning}{END}"
    if path.exists() and BEGIN in (text := path.read_text("utf-8")) and END in text:
        head, rest = text.split(BEGIN, 1)
        path.write_text(head + block + rest.split(END, 1)[1], "utf-8")
    else:
        path.write_text(f"# Search quality report\n\n{block}\n", "utf-8")


def format_tuning(t: dict) -> str:
    names = ", ".join(f"{NAMES[s]} {w:g}" for s, w in t["best_weights"].items())
    default = "" if t["default_in_sample"] is None else f" (default weights: {t['default_in_sample']:.3f})"
    return ("\n### RRF weight search\n\n"
            f"Grid {t['grid']} per signal, {t['combos']} combinations, objective {t['metric']}. "
            f"Best in-sample: {names} → {t['best_in_sample']:.3f}{default}. "
            f"Leave-one-query-out estimate: {t['leave_one_out']:.3f}. "
            "Treat the in-sample number as optimistic; change the defaults only if the leave-one-out score beats them.\n\n")


# ---------- entry point ----------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--queries", type=Path, default=Path(__file__).with_name("queries.jsonl"))
    p.add_argument("--index", type=Path, help="index folder (default: the configured one)")
    p.add_argument("--report", type=Path, help="write the generated part of this markdown file")
    p.add_argument("--json", type=Path, help="also dump per-query scores as JSON")
    p.add_argument("--tune", action="store_true", help="search RRF weights")
    args = p.parse_args(argv)

    from prompt_scout.config import index_dir
    from prompt_scout.embed import embedder_for
    from prompt_scout.service import read_image

    queries, problems = load(args.queries)
    for w in problems.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if problems.errors:
        print("Query file has errors:\n  " + "\n  ".join(problems.errors), file=sys.stderr)
        return 2

    index = Index.load(args.index or index_dir())
    embedder = embedder_for(index.embedder_name)

    relevant: dict[str, set[str]] = {}
    dropped: list[str] = []
    usable: list[Query] = []
    for q in queries:
        present = {r for r in q.relevant if r in index.by_id}
        dropped += [f"{q.id}:{r}" for r in q.relevant if r not in index.by_id]
        if present:
            relevant[q.id] = present
            usable.append(q)
        else:
            print(f"warning: {q.id} skipped, none of its relevant ids are in the index", file=sys.stderr)
    if not usable:
        print("No usable queries.", file=sys.stderr)
        return 2

    lists_by_query: dict[str, Lists] = {}
    for q in usable:
        image = read_image(q.image) if q.image else None  # a bad path stops the run: no silent skipping
        lists_by_query[q.id] = index.ranked_lists(embedder, q.text, image)

    active = {s for lists in lists_by_query.values() for s in lists}
    per = evaluate(index, lists_by_query, relevant, variants(active))
    generated, summary = build_report(index, usable, per, dropped)
    tuning = ""
    if args.tune:
        tuning = format_tuning(tune(index, lists_by_query, relevant, active))

    print(generated + tuning)
    print(f"Summary: {summary}")
    if args.report:
        write_report(args.report, generated, tuning)
        print(f"Report written to {args.report}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(per, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
