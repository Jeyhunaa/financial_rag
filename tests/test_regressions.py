"""Regressions: page-label correctness, extraction, and answer provenance."""
import fitz
import pytest
from evaluate import evidence_rank
from rag import FinancialRAG, chunk_page, page_text


def test_adjacent_page_cannot_score():
    evidence = [{'evidence_doc_name': 'X_2020_10K', 'evidence_page_num': 4}]
    assert evidence_rank([{'doc': 'X_2020_10K', 'page': 3}], evidence) == 0
    assert evidence_rank([{'doc': 'X_2020_10K', 'page': 5}], evidence) == 0
    assert evidence_rank([{'doc': 'X_2020_10K', 'page': 4}], evidence) == 1


def test_empty_page_and_invalid_chunk_size():
    assert chunk_page('') == []
    with pytest.raises(ValueError):
        chunk_page('x', size=0)


def test_pdf_financial_numbers_are_present():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 70), 'Revenue 2022 1,250.50 USD; Cost 900.25 USD')
    text = page_text(page)
    assert '1,250.50' in text
    assert '900.25' in text
    pdf.close()


def test_source_only_includes_used_citations(monkeypatch):
    engine = FinancialRAG()
    evidence = [{'doc': 'ACME_2020_10K', 'page': 2, 'text': 'Revenue $10 million'},
                {'doc': 'ACME_2020_10K', 'page': 3, 'text': 'Costs $6 million'}]
    monkeypatch.setattr(engine, '_generate', lambda prompt: 'Revenue was $10 million [1].')
    result = engine.answer('Revenue?', chunks=evidence)
    assert result['answer'].endswith('[1].')
    assert len(result['sources']) == 1
    assert result['sources'][0]['pdf_page'] == 3


def test_hallucinated_citation_is_rejected(monkeypatch):
    engine = FinancialRAG()
    monkeypatch.setattr(engine, '_generate', lambda prompt: 'Profit was $999 billion [8].')
    result = engine.answer('Profit?', chunks=[{'doc': 'A', 'page': 0, 'text': 'Revenue 50'}])
    assert result['answer'] == 'Insufficient evidence.'
    assert not result['citation_valid']
    assert result['sources'] == []


def test_uncited_answer_is_rejected(monkeypatch):
    engine = FinancialRAG()
    monkeypatch.setattr(engine, '_generate', lambda prompt: 'Profit was 100.')
    result = engine.answer('Profit?', chunks=[{'doc': 'A', 'page': 0, 'text': 'Profit 100'}])
    assert result['answer'] == 'Insufficient evidence.'


def test_abstention_without_citation_is_accepted(monkeypatch):
    engine = FinancialRAG()
    monkeypatch.setattr(engine, '_generate', lambda prompt: 'Insufficient evidence.')
    result = engine.answer('Profit?', chunks=[{'doc': 'A', 'page': 0, 'text': 'Revenue 200'}])
    assert result['citation_valid']
