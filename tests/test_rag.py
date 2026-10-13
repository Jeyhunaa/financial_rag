from rag import chunk_page, tokens
from evaluate import evidence_hit, evidence_rank, token_f1
from fastapi.testclient import TestClient
from api import app


def test_chunk_overlap():
    chunks = chunk_page('a b c d e f', size=4, overlap=2)
    assert chunks == ['a b c d', 'c d e f']


def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_page('hello', size=2, overlap=2)


def test_tokenization():
    assert tokens('2023 Revenue: $10.5M') == ['2023', 'revenue', '10.5', 'm']


def test_evidence_page_matching_accepts_pdf_page_number():
    chunks = [{'doc': 'ABC_2020_10K', 'page': 2}, {'doc': 'XYZ', 'page': 0}]
    evidence = [{'evidence_doc_name': 'ABC_2020_10K', 'evidence_page_num': 3}]
    assert not evidence_hit(chunks, evidence)
    assert evidence_rank(chunks, evidence) == 0


def test_document_evidence_matching():
    chunks = [{'doc': 'ABC_2020_10K', 'page': 8}]
    evidence = [{'evidence_doc_name': 'ABC_2020_10K', 'evidence_page_num': 2}]
    assert evidence_hit(chunks, evidence, page_level=False)


def test_answer_f1_exact():
    assert token_f1('net income was 10', 'net income was 10') == 1


def test_answer_f1_counts_duplicate_tokens():
    assert token_f1('revenue revenue 10', 'revenue 10') < 1


def test_invalid_mode_without_loading_index():
    response = TestClient(app).post('/ask', json={'question': 'What was revenue?', 'mode': 'invalid'})
    assert response.status_code == 422


def test_financebench_evidence_schema_document_match():
    from evaluate import evidence_rank
    chunks = [{'doc': '3M_2018_10K', 'page': 59, 'text': 'cash flow'}]
    evidence = [{'doc_name': '3M_2018_10K', 'evidence_page_num': 59}]
    assert evidence_rank(chunks, evidence, page_level=False) == 1
    assert evidence_rank(chunks, evidence, page_level=True) == 1


def test_financebench_page_offset_match():
    from evaluate import evidence_rank
    chunks = [{'doc': '3M_2018_10K', 'page': 58, 'text': 'cash flow'}]
    evidence = [{'doc_name': '3M_2018_10K', 'evidence_page_num': 59}]
    assert evidence_rank(chunks, evidence, page_level=True) == 0


def test_document_filter_requires_company_and_year():
    from rag import FinancialRAG
    rag = FinancialRAG()
    rag.doc_names = {'3M_2022_10K', 'COSTCO_2021_10K', '3M_2018_10K'}
    assert rag._matching_documents('Was 3M capital intensive in 2022?') == {'3M_2022_10K'}
    assert rag._matching_documents('What were Costco net sales in 2021?') == {'COSTCO_2021_10K'}
    assert not rag._matching_documents('What were net sales in 2021?')


def test_default_chunk_size_is_smaller_for_precise_retrieval():
    from rag import chunk_page
    parts = chunk_page(' '.join(f'word{i}' for i in range(600)))
    assert len(parts) > 2
    assert max(len(part.split()) for part in parts) <= 280


def test_document_restricted_search_does_not_drop_global_candidates():
    import numpy as np
    import pytest
    faiss = pytest.importorskip('faiss')
    from rag import FinancialRAG

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.]] if len(texts) == 1 else [[1., 0.]] * len(texts), dtype='float32')

    rag = FinancialRAG(model=FakeEncoder())
    rag.chunks = ([{'doc': 'OTHER_2021_10K', 'page': i, 'text': 'revenue'} for i in range(120)] +
                  [{'doc': 'COSTCO_2021_10K', 'page': 0, 'text': 'net sales'}])
    rag.doc_names = {c['doc'] for c in rag.chunks}
    rag.doc_to_ids = {'OTHER_2021_10K': list(range(120)), 'COSTCO_2021_10K': [120]}
    index = faiss.IndexFlatIP(2)
    vectors = np.array([[1., 0.]] * 120 + [[0.9, 0.43589]], dtype='float32')
    faiss.normalize_L2(vectors)
    index.add(vectors)
    rag.index = index
    rag.bm25 = object()
    assert rag.retrieve('What were Costco net sales in 2021?', k=1, mode='dense')[0]['doc'] == 'COSTCO_2021_10K'
