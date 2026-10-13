# Design notes

**Why dense by default?** In observed development runs, dense retrieval reached 60/150 evidence-page hits; all hybrid modes were lower. Retaining simple baselines is more credible than choosing an unverified fusion or reranker.

**Indexing**: PDF pages are extracted by PyMuPDF with best-effort table rows; 220-word chunks with 45-word overlap, per-page zero-index metadata; all-MiniLM-L6-v2 normalized embeddings in FAISS IndexFlatIP and BM25 for comparison.

**Serving**: Retrieval picks unique source pages; Gemini receives up to 5 excerpt snippets and is instructed to cite valid snippet IDs. Invalid/missing IDs cause abstention. Sources report both human PDF page numbers and machine page indices.

**Why not more dependencies?** Rerankers, agents, vector DB services, and OCR add significant maintenance and have not yet demonstrated a validated gain on this dataset. They are consciously out of scope for this maintainable junior-engineer prototype.

**Known weak spots**: dense page hit rate, financial table layout recovery, multi-page numeric arithmetic, PDF extraction/financial annotations, factual entailment of citations, provider-dependent model availability.
