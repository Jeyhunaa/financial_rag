# Evaluation protocol

**Task**: 150 publicly available FinanceBench questions, each with one or more zero-index `evidence_page_num` labels. For each question, retrieve 5 *distinct PDF pages*.

- **Document Recall@5**: binary fraction of questions where at least one retrieved result belongs to an annotated document.
- **Evidence Page Recall@5**: binary fraction where at least one `(document, zero-index page)` matches an annotated evidence page. **It is hit rate / any-evidence recall**, not percentage of all gold evidence pages recovered.
- **MRR@5**: mean reciprocal rank of first relevant document/page; zero if absent.
- Do not give `FinanceBench doc_name`, gold evidence texts, answers or page labels to retrieval. Do not choose fusion weights against the final evaluation split and later describe it as held-out.
- Verify the same index, model, dataset, k, and page indexing before comparing runs.

## Historical results

See `reports/historical_metrics.json`, `reports/historical_evaluation.csv`, `reports/historical_comparison.png`. These results were uploaded by the project owner and are development-evaluation numbers; no new full-model benchmark was run during the final portfolio packaging.

## Answer-quality audit (still needed)

1. Run `python evaluate.py --answers --limit 10 --answer-mode dense` with a valid Gemini key, subject to quota.
2. For each returned question, separately check numerical correctness, units/period, citation existence, citation factual support, and abstention if evidence insufficient.
3. Record successes *and failures*, plus sample size and API/model version. Token F1 is not a substitute for these labels.
4. Never call Gemini for retrieval-only comparisons.
