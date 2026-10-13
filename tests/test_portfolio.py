"""Regression checks for portfolio release behavior without model downloads."""
from pathlib import Path
from unittest.mock import patch
import json

from rag import FinancialRAG
from evaluate import evidence_rank

def test_default_retriever_is_verified_dense():
    assert FinancialRAG.retrieve.__defaults__ == (5, "dense")
    assert FinancialRAG.answer.__defaults__ == (5, "dense", None)

def test_zero_index_exact_matching():
    chunk = {"doc": "A_2020_10K", "page": 9}
    gold = [{"doc_name": "A_2020_10K", "evidence_page_num": 10}]
    assert evidence_rank([chunk], gold) == 0
    assert evidence_rank([chunk], [{**gold[0], "evidence_page_num": 9}]) == 1

def test_index_manifest_matches_bundled_chunks_if_present():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts" / "manifest.json"
    chunks = root / "artifacts" / "chunks.json"
    if manifest.exists() and chunks.exists():
        assert json.loads(manifest.read_text())["chunks"] == len(json.loads(chunks.read_text()))
