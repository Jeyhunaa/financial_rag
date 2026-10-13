"""FastAPI entrypoint: uvicorn api:app --reload"""
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag import FinancialRAG

app = FastAPI(title='Financial Document Intelligence RAG', version='1.0')


class Question(BaseModel):
    question: str = Field(min_length=3)
    mode: str = 'dense'
    k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def engine():
    return FinancialRAG().load()


@app.get('/health')
def health():
    
    try:
        return {'status': 'ok', 'index_loaded': engine().index is not None}
    except (OSError, FileNotFoundError, ValueError):
        return {'status': 'not_ready', 'index_loaded': False}


@app.post('/ask')
def ask(query: Question):
    if query.mode not in ('dense', 'hybrid', 'hybrid_balanced', 'page_hybrid'):
        raise HTTPException(422, 'mode must be dense, hybrid, hybrid_balanced or page_hybrid')
    try:
        return engine().answer(query.question, query.k, query.mode)
    except Exception as exc:
        raise HTTPException(503, f'RAG unavailable: {exc.__class__.__name__}') from exc
