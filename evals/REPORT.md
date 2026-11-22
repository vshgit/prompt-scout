# Search quality report

<!-- generated:begin -->
Index: `gemini-embedding-2@768`, 2158 prompts (2157 with sample images). Queries: 32 (26 text, 4 photo, 2 mixed), 32 labels verified by hand, 0 draft. Run on 2026-11-22.

### Ablation (all queries)

| Variant | Recall@10 | MRR@10 | nDCG@10 | Hit@10 | Δ nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Hybrid (all signals, default weights) | 0.780 | 0.854 | 0.732 | 1.000 | +0.000 |
| − photo↔images | 0.731 | 0.826 | 0.693 | 1.000 | -0.039 |
| − photo↔prompts | 0.731 | 0.815 | 0.685 | 1.000 | -0.047 |
| − text↔images | 0.673 | 0.807 | 0.651 | 0.969 | -0.081 |
| − text↔prompts | 0.607 | 0.722 | 0.578 | 0.938 | -0.154 |
| − BM25 | 0.735 | 0.849 | 0.702 | 1.000 | -0.030 |
| Dense only (no BM25) | 0.735 | 0.849 | 0.702 | 1.000 | -0.030 |
| only photo↔images | 0.078 | 0.086 | 0.062 | 0.125 | -0.670 |
| only photo↔prompts | 0.068 | 0.097 | 0.064 | 0.125 | -0.668 |
| only text↔images | 0.417 | 0.527 | 0.398 | 0.844 | -0.334 |
| only text↔prompts | 0.591 | 0.729 | 0.584 | 0.875 | -0.148 |
| only BM25 | 0.298 | 0.406 | 0.303 | 0.500 | -0.429 |

### Hybrid vs dense only, by language

| Group | n | Hybrid Recall@10 | Hybrid nDCG@10 | Dense Recall@10 | Dense nDCG@10 |
|---|---:|---:|---:|---:|---:|
| en | 18 | 0.781 | 0.737 | 0.715 | 0.675 |
| ru | 14 | 0.779 | 0.726 | 0.762 | 0.736 |

### Hybrid vs dense only, by query type

| Group | n | Hybrid Recall@10 | Hybrid nDCG@10 | Dense Recall@10 | Dense nDCG@10 |
|---|---:|---:|---:|---:|---:|
| mixed | 2 | 1.000 | 0.931 | 1.000 | 0.931 |
| photo | 4 | 0.938 | 0.822 | 0.938 | 0.822 |
| text | 26 | 0.739 | 0.703 | 0.684 | 0.665 |

Hybrid minus dense only: Recall@10 +0.045 (95% CI +0.008 … +0.090), nDCG@10 +0.030 (95% CI -0.000 … +0.066); paired bootstrap over queries.


### RRF weight search

Grid [0.0, 0.5, 1.0, 2.0] per signal, 1023 combinations, objective nDCG@10. Best in-sample: photo↔images 1, photo↔prompts 0.5, text↔images 1, text↔prompts 1, BM25 1 → 0.732 (default weights: 0.732). Leave-one-query-out estimate: 0.694. Treat the in-sample number as optimistic; change the defaults only if the leave-one-out score beats them.

<!-- generated:end -->

## Conclusions

1. **The hybrid beats every reduced version.** Switching off any of the five signals lowers nDCG@10. On all 32 queries: text↔prompts is the backbone (−0.154), then text↔images (−0.081), photo↔prompts (−0.047), photo↔images (−0.039), BM25 (−0.030).
2. **Photo queries need both photo signals.** Photo signals exist only for the 6 photo and mixed queries, so the "only photo…" rows above average in zeros from text queries. Recomputed on those 6 queries (the runner now prints this table): hybrid nDCG@10 ≈ 0.86; without photo↔images ≈ 0.65, without photo↔prompts ≈ 0.61; either photo signal alone ≈ 0.33–0.34 and finds nothing correct for 2 of the 6 queries. Photo↔prompts, weighted 0.5 as the "weaker" cross-modal signal, matters as much as photo↔images. Six queries are few: treat this as direction, not size.
3. **Hybrid vs dense only: small, consistent.** +0.045 Recall@10 (95% CI +0.008 … +0.090); for nDCG@10 the interval touches zero (+0.030, CI −0.000 … +0.066). The gain comes from BM25 on English text queries (+0.062 nDCG@10); on Russian queries BM25 slightly hurts (−0.010): Russian words never occur in the English prompts. Photo and mixed queries are unaffected. Switching BM25 off for non-Latin queries is a candidate change; 14 Russian queries are too few to decide.
4. **Label review flipped a conclusion.** On the first, text-only labels, removing text↔images *raised* nDCG@10 by +0.117 and text↔prompts alone looked best. After the owner reviewed pooled candidates with their sample images (hybrid Recall@10 on the 26 text queries went from 0.525 to 0.739), text↔images turned out to be useful. Labels built from one modality reward the signal that works in that modality.
5. **Default RRF weights stay.** On a grid of 1,023 weight combinations the best in-sample one *is* the current default (0.732), and choosing weights without the scored query gives 0.694 under leave-one-out: there is nothing to gain from tuning on this set.

First run, draft labels, 26 text queries, for reference: hybrid Recall@10 0.525, MRR@10 0.586, nDCG@10 0.488, Hit@10 0.808.

## Method

- **Index:** the `profile-avatar` category (about 2,200 prompts), Gemini Embedding 2 at 768 dimensions, sample images included.
- **Queries:** `evals/queries.jsonl`, 32 queries: 26 text, 4 photo, 2 mixed (photo + wish), all labels reviewed by hand (3–10 per query). Most intents appear twice, in English and in Russian, with the same correct ids, so the pair shows how much a language switch costs. Photo and mixed queries (`image` field) use shareable photos from outside the library (Unsplash, Wikimedia Commons), referenced by URL: the library's own sample images would trivially find themselves, and private photos cannot go in a public file.
- **Labels:** first draft: 5 correct ids per query, chosen by reading titles, descriptions and prompt texts, without search results or sample images (`status: "draft"`). Review: `python -m evals.review` shows each query's pool (labelled ids + hybrid top 10 + top 5 of every signal; top 10 since Nov 2026, with sample images); the owner keeps what truly fits and adds missing answers, 3–10 per query (`status: "verified"`). Pooling over several systems keeps the labels from favouring one signal.
- **Metrics**, all cut off at 10 and averaged over queries with binary relevance: Recall@10 (share of the labelled ids found), MRR@10 (mean of 1 / rank of the first correct id), nDCG@10 (correct ids near the top count more), Hit@10 (share of queries with at least one correct id).
- **Ablation:** each query is embedded once and its five ranked lists are re-fused with one signal switched off at a time, with BM25 off (dense only), and with each signal alone. Same lists, so no extra API calls.
- **Hybrid vs dense only** comes with a paired bootstrap 95% interval over queries.

## Limitations

- **Pooled labels.** The reviewed answers come from what the signals found (labelled ids + hybrid top 10 + top 5 of each signal). A fitting prompt that no signal surfaced is unknown, so recall is relative to the pool and absolute numbers are somewhat optimistic; comparisons between variants are fairer than the absolute numbers, with the caveat below.
- **The pool favours the hybrid.** The review page showed the hybrid top 10 but only the top 5 of each single signal, so a correct answer at rank 6–10 of one signal could stay unlabelled. "Only X" rows are therefore lower bounds; "− X" rows, which still fuse the remaining signals, are less affected. The review page now pools the top 10 of every signal for new queries; the existing 32 were labelled with the old pool.
- **Few photo queries.** 4 photo + 2 mixed (3–4 labels each, Unsplash photos by URL; if a URL stops working the runner stops with an error rather than skipping it): enough to see that both photo signals matter, not to measure by how much.
- **Small set.** 26 queries, 12 of them Russian versions of English ones (they may share answers): intervals are wide, and small differences between variants are noise.
- **Index vs labels.** Labels come from the library as of 2026-11-22. Ids absent from your index are ignored and listed in the report.
