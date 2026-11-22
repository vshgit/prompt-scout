# Search quality report

<!-- generated:begin -->
_No results yet. Run `python -m evals.run --report evals/REPORT.md` against your index to fill this block._
<!-- generated:end -->

## Conclusions

_To be written after the first real run (this section is never overwritten by the script)._

## Method

- **Index:** the `profile-avatar` category (about 2,200 prompts), Gemini Embedding 2 at 768 dimensions, sample images included.
- **Queries:** `evals/queries.jsonl`, 26 text queries. Most intents appear twice, in English and in Russian, with the same correct ids, so the pair shows how much a language switch costs. The runner also accepts photo and mixed queries (`image` field: path or URL); none are in the set yet, so the two photo signals are not measured.
- **Labels:** 5 correct ids per query, chosen by reading titles, descriptions and prompt texts in the library, without looking at any search results. Sample images were not inspected. Every row has `status: "draft"` until its labels are checked by hand and switched to `"verified"`.
- **Metrics**, all cut off at 10 and averaged over queries with binary relevance: Recall@10 (share of the labelled ids found), MRR@10 (mean of 1 / rank of the first correct id), nDCG@10 (correct ids near the top count more), Hit@10 (share of queries with at least one correct id).
- **Ablation:** each query is embedded once and its five ranked lists are re-fused with one signal switched off at a time, with BM25 off (dense only), and with each signal alone. Same lists, so no extra API calls.
- **Hybrid vs dense only** comes with a paired bootstrap 95% interval over queries.

## Limitations

- **Sparse labels.** Many prompts fit a query (dozens for "cyberpunk neon portrait"), only 5 are labelled. A correct but unlabelled result counts as a miss, so Recall@10 and nDCG@10 are lower bounds. Hit@10 is the least sensitive to this.
- **Labels lean towards words.** They were picked by reading text, so signals that work on text (BM25, text↔prompts) are favoured and the image signal (text↔images) is probably undervalued.
- **Small set.** Around 26 queries, half of them paraphrases of the other half: intervals are wide, and small differences between variants are noise.
- **Index vs labels.** Labels come from the library as of 2026-11-22. Ids absent from your index are ignored and listed in the report.
