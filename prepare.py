"""Download only FinanceBench's public question/metadata files and matching PDFs."""
import json
import os
import time
import socket
import re
import urllib.request
from pathlib import Path

from rag import DATA, ART, extract_documents, FinancialRAG

RAW = 'https://raw.githubusercontent.com/patronus-ai/financebench/main/data/'


def download(url, path):
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=40) as response, temp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block: break
                out.write(block)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def prepare():
    start = time.monotonic()
    DATA.mkdir(exist_ok=True)
    print("[prepare] Downloading/checking FinanceBench inputs...", flush=True)
    for name in ('financebench_open_source.jsonl', 'financebench_document_information.jsonl'):
        download(RAW + name, DATA / name)
    questions = [json.loads(x) for x in (DATA / 'financebench_open_source.jsonl').read_text().splitlines()]
    metadata = [json.loads(x) for x in (DATA / 'financebench_document_information.jsonl').read_text().splitlines()]
    required = {q['doc_name'] for q in questions}
    urls = {m['doc_name']: m.get('doc_link') for m in metadata}
    pdfs = DATA / 'pdfs'
    failures = []
    for n, name in enumerate(sorted(required), 1):
        print(f"[prepare] PDF {n}/{len(required)}: {name}", flush=True)
        # Prefer PDFs in the FinanceBench repository, fallback to published source link.
        path = pdfs / (name + '.pdf')
        candidates = ['https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/' + name + '.pdf']
        if urls.get(name):
            candidates.append(urls[name])
        for url in candidates:
            try:
                download(url, path)
                if path.read_bytes()[:4] != b'%PDF':
                    path.unlink(missing_ok=True)
                    continue
                break
            except Exception:
                path.unlink(missing_ok=True)
        if not path.exists():
            failures.append(name)
    if failures:
        print('Missing PDFs:', failures)
        (DATA / 'missing_pdfs.json').write_text(json.dumps(failures, indent=2))
    print(f'Questions: {len(questions)}; PDFs: {len(list(pdfs.glob("*.pdf")))}; missing: {len(failures)}')
    if failures:
        raise RuntimeError(f'Preparation stopped: {len(failures)} source PDFs missing; see data/missing_pdfs.json. No incomplete index will be built.')
    if not list(pdfs.glob('*.pdf')):
        raise RuntimeError('No PDFs downloaded. Check network access or manually place PDFs in data/pdfs')
    print("[prepare] Extracting PDF pages (cached per PDF for resumable runs)...", flush=True)
    chunks = extract_documents()
    print(f"[prepare] Extracted {len(chunks)} chunks; embedding/indexing now. This can take substantial CPU time on Windows.", flush=True)
    rag = FinancialRAG().build(chunks)
    rag.save()
    print(f'[prepare] FINISHED. Indexed {len(chunks)} chunks in {time.monotonic()-start:.1f}s. Index saved to {ART}', flush=True)


if __name__ == '__main__':
    prepare()
