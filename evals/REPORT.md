# Search quality report

<!-- generated:begin -->
Index: `gemini-embedding-2@768`, 2158 prompts (2157 with sample images). Queries: 26 (26 text, 0 photo, 0 mixed), 26 labels verified by hand, 0 draft. Run on 2026-11-22.

### Ablation (all queries)

| Variant | Recall@10 | MRR@10 | nDCG@10 | Hit@10 | Δ nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Hybrid (all signals, default weights) | 0.739 | 0.821 | 0.703 | 1.000 | +0.000 |
| − text↔images | 0.655 | 0.801 | 0.645 | 1.000 | -0.058 |
| − text↔prompts | 0.574 | 0.696 | 0.555 | 0.962 | -0.148 |
| − BM25 | 0.684 | 0.814 | 0.665 | 1.000 | -0.037 |
| Dense only (no BM25) | 0.684 | 0.814 | 0.665 | 1.000 | -0.037 |
| only text↔images | 0.446 | 0.571 | 0.426 | 0.962 | -0.277 |
| only text↔prompts | 0.673 | 0.821 | 0.668 | 1.000 | -0.035 |
| only BM25 | 0.367 | 0.500 | 0.373 | 0.615 | -0.330 |

### Hybrid vs dense only, by language

| Group | n | Hybrid Recall@10 | Hybrid nDCG@10 | Dense Recall@10 | Dense nDCG@10 |
|---|---:|---:|---:|---:|---:|
| en | 14 | 0.737 | 0.712 | 0.651 | 0.633 |
| ru | 12 | 0.742 | 0.692 | 0.722 | 0.703 |

Hybrid minus dense only: Recall@10 +0.055 (95% CI +0.007 … +0.106), nDCG@10 +0.037 (95% CI +0.002 … +0.078); paired bootstrap over queries.


### RRF weight search

Grid [0.0, 0.5, 1.0, 2.0] per signal, 63 combinations, objective nDCG@10. Best in-sample: text↔images 0.5, text↔prompts 1, BM25 0.5 → 0.709 (default weights: 0.703). Leave-one-query-out estimate: 0.676. Treat the in-sample number as optimistic; change the defaults only if the leave-one-out score beats them.

<!-- generated:end -->

## Conclusions

1. **The hybrid beats every reduced version.** Switching off any text signal lowers nDCG@10: text↔prompts is the backbone (−0.148), text↔images adds −0.058, BM25 −0.037. The best single signal (text↔prompts) reaches 0.668 against 0.703 for the hybrid.
2. **Hybrid vs dense only: small but real.** +0.055 Recall@10 (95% CI +0.007 … +0.106), +0.037 nDCG@10 (CI +0.002 … +0.078). BM25 helps English queries (+0.079 nDCG@10) and does not help Russian ones (−0.011): Russian words never occur in the English prompts. Switching BM25 off for non-Latin queries is a candidate change, but 12 queries are too few to decide.
3. **Label review flipped a conclusion.** On the first, text-only labels, removing text↔images *raised* nDCG@10 by +0.117 and text↔prompts alone looked best. After the owner reviewed pooled candidates with their sample images (Recall@10 of the hybrid went from 0.525 to 0.739), text↔images turned out to be useful. Labels built from one modality reward the signal that works in that modality.
4. **Default RRF weights stay.** The best grid weights gain +0.006 in-sample, and their leave-one-query-out score (0.676) is below the defaults (0.703): with 26 queries, tuning overfits.

First run, draft labels, for reference: hybrid Recall@10 0.525, MRR@10 0.586, nDCG@10 0.488, Hit@10 0.808.

## Method

- **Index:** the `profile-avatar` category (about 2,200 prompts), Gemini Embedding 2 at 768 dimensions, sample images included.
- **Queries:** `evals/queries.jsonl`, 26 text queries, 187 reviewed labels (3–10 per query). Most intents appear twice, in English and in Russian, with the same correct ids, so the pair shows how much a language switch costs. The runner also accepts photo and mixed queries (`image` field: path or URL); none are in the set yet, so the two photo signals are not measured.
- **Labels:** first draft: 5 correct ids per query, chosen by reading titles, descriptions and prompt texts, without search results or sample images (`status: "draft"`). Review: `python -m evals.review` shows each query's pool (labelled ids + hybrid top 10 + top 5 of every signal, with sample images); the owner keeps what truly fits and adds missing answers, 3–10 per query (`status: "verified"`). Pooling over several systems keeps the labels from favouring one signal.
- **Metrics**, all cut off at 10 and averaged over queries with binary relevance: Recall@10 (share of the labelled ids found), MRR@10 (mean of 1 / rank of the first correct id), nDCG@10 (correct ids near the top count more), Hit@10 (share of queries with at least one correct id).
- **Ablation:** each query is embedded once and its five ranked lists are re-fused with one signal switched off at a time, with BM25 off (dense only), and with each signal alone. Same lists, so no extra API calls.
- **Hybrid vs dense only** comes with a paired bootstrap 95% interval over queries.

## Limitations

- **Pooled labels.** The reviewed answers come from what the signals found (labelled ids + hybrid top 10 + top 5 of each signal). A fitting prompt that no signal surfaced is unknown, so recall is relative to the pool and absolute numbers are somewhat optimistic; comparisons between variants are fair because every variant's top results were in the pool.
- **Photo signals not measured yet.** All 26 queries are text; photo↔images and photo↔prompts need photo and mixed queries.
- **Small set.** 26 queries, 12 of them Russian versions of English ones (they may share answers): intervals are wide, and small differences between variants are noise.
- **Index vs labels.** Labels come from the library as of 2026-11-22. Ids absent from your index are ignored and listed in the report.
