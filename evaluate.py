"""Evaluate dense vs hybrid FinanceBench retrieval and optional Gemini answers."""
import argparse
import csv
import json
import re
import time
from collections import Counter

from rag import FinancialRAG, load_questions, ART


def _evidence_targets(evidence):
    targets = []
    for item in evidence or []:
        doc = item.get('doc_name') or item.get('evidence_doc_name')
        page = item.get('evidence_page_num')
        if doc:
            targets.append((doc, page))
    return targets


def evidence_rank(chunks, evidence, page_level=True):
    """Return 1-based rank of first relevant result, or 0 when no result matches."""
    targets = _evidence_targets(evidence)
    for rank, chunk in enumerate(chunks, 1):
        for doc, page in targets:
            if chunk['doc'] != doc:
                continue
            if not page_level:
                return rank
            if page is None:
                continue
            # FinanceBench evidence_page_num is ZERO-indexed, exactly as PyMuPDF.
            if chunk['page'] == int(page):
                return rank
    return 0


def evidence_hit(chunks, evidence, page_level=True):
    return bool(evidence_rank(chunks, evidence, page_level))


def normalize(text):
    return re.findall(r'[a-z0-9.]+', str(text).lower())


def token_f1(pred, gold):
    pred_tokens, gold_tokens = normalize(pred), normalize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _write_rows(rows):
    if not rows:
        return
    ART.mkdir(exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (ART / 'evaluation.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def evaluate(limit=0, answers=False, k=5, answer_mode='dense', answer_delay=4.0):
    rag = FinancialRAG().load()
    questions = load_questions()
    if limit:
        questions = questions[:limit]
    if not limit and len(questions) != 150:
        raise ValueError(f'Expected all 150 public FinanceBench questions; found {len(questions)}')
    indexed_docs = rag.doc_names
    expected_docs = {q['doc_name'] for q in questions}
    missing = expected_docs - indexed_docs
    if missing:
        raise RuntimeError(f'Missing indexed filings ({len(missing)}): {sorted(missing)}')
    if not questions:
        raise ValueError('No FinanceBench questions found')

    ART.mkdir(exist_ok=True)
    rows = []
    for mode in ('dense', 'hybrid', 'hybrid_balanced', 'page_hybrid'):
        print(f'\nEvaluating {mode} retrieval...', flush=True)
        for n, q in enumerate(questions, 1):
            chunks = rag.retrieve(q['question'], k=k, mode=mode)
            doc_rank = evidence_rank(chunks, q.get('evidence', []), False)
            page_rank = evidence_rank(chunks, q.get('evidence', []), True)
            row = {
                'mode': mode,
                'question_id': q['financebench_id'],
                'document_hit': int(bool(doc_rank)),
                'page_hit': int(bool(page_rank)),
                'document_rr': round(1 / doc_rank, 4) if doc_rank else 0.0,
                'page_rr': round(1 / page_rank, 4) if page_rank else 0.0,
                'question': q['question'],
                'retrieved_pages': '; '.join(f"{c['doc']}:pdf{c['page'] + 1}" for c in chunks),
                'reference_pages': '; '.join(f'{d}:{p}' for d, p in _evidence_targets(q.get('evidence', []))),
            }
            if answers and mode == answer_mode:
                try:
                    generated = rag.answer(q['question'], k=k, mode=mode, chunks=chunks)['answer']
                    row['answer_token_f1'] = round(token_f1(generated, q['answer']), 4)
                    row['generated_answer'] = generated
                    row['llm_status'] = 'success'
                    row['llm_error'] = ''
                except Exception as exc:
                    # Provider failures must never invalidate retrieval evaluation.
                    row['answer_token_f1'] = ''
                    row['generated_answer'] = ''
                    row['llm_status'] = 'failed'
                    row['llm_error'] = f'{exc.__class__.__name__}: {str(exc)[:180]}'
                    print(f'  LLM failed for {q["financebench_id"]}: {exc.__class__.__name__}', flush=True)
                if answer_delay > 0 and n < len(questions):
                    time.sleep(answer_delay)
            elif answers:
                row['answer_token_f1'] = ''
                row['generated_answer'] = ''
                row['llm_status'] = 'not_evaluated'
                row['llm_error'] = ''
            rows.append(row)
            _write_rows(rows)  # checkpoint so long API evaluations survive interruption
            print(f'  [{n}/{len(questions)}] done', flush=True)

    summary = {}
    for mode in ('dense', 'hybrid', 'hybrid_balanced', 'page_hybrid'):
        subset = [r for r in rows if r['mode'] == mode]
        summary[mode] = {
            'document_recall@k': round(sum(r['document_hit'] for r in subset) / len(subset), 4),
            'page_recall@k': round(sum(r['page_hit'] for r in subset) / len(subset), 4),
            'document_mrr@k': round(sum(r['document_rr'] for r in subset) / len(subset), 4),
            'page_mrr@k': round(sum(r['page_rr'] for r in subset) / len(subset), 4),
        }
        if answers and mode == answer_mode:
            valid = [r['answer_token_f1'] for r in subset if r.get('llm_status') == 'success']
            failures = [r for r in subset if r.get('llm_status') == 'failed']
            summary[mode]['answer_token_f1'] = round(sum(valid) / len(valid), 4) if valid else None
            summary[mode]['answered'] = len(valid)
            summary[mode]['llm_failures'] = len(failures)
        elif answers:
            summary[mode]['answer_token_f1'] = None
            summary[mode]['answered'] = 0
            summary[mode]['llm_failures'] = 0

    payload = {'n_questions': len(questions), 'page_matching': 'exact_zero_index', 'page_diversified': True, 'page_dense_weight': __import__('os').getenv('PAGE_DENSE_WEIGHT', '3.0'), 'hybrid_dense_weight': __import__('os').getenv('HYBRID_DENSE_WEIGHT', '3.0'), 'k': k, 'answer_evaluation': answers,
               'answer_mode': answer_mode if answers else None, 'metrics': summary}
    (ART / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metric_names = ['document_recall@k', 'page_recall@k', 'document_mrr@k', 'page_mrr@k']
        x = range(len(metric_names))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([i - .25 for i in x], [summary['dense'][m] for m in metric_names], .25, label='Dense')
        ax.bar([i for i in x], [summary['hybrid'][m] for m in metric_names], .25, label='Hybrid weighted')
        ax.bar([i + .25 for i in x], [summary['hybrid_balanced'][m] for m in metric_names], .25, label='Hybrid balanced (old)')
        ax.bar([i + .50 for i in x], [summary['page_hybrid'][m] for m in metric_names], .25, label='Page hybrid (experimental)')
        ax.set_xticks(list(x), metric_names, rotation=15, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score (0–1)')
        ax.set_title(f'FinanceBench Retrieval Comparison (n={len(questions)}, k={k})')
        ax.grid(axis='y', alpha=0.2)
        ax.legend()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        ax.set_ylim(0, 1.12)
        fig.tight_layout()
        fig.savefig(ART / 'comparison.png', dpi=160)
        plt.close(fig)

        if answers:
            score = summary[answer_mode].get('answer_token_f1')
            if score is not None:
                fig, ax = plt.subplots(figsize=(5.5, 4.2))
                bars = ax.bar([f'{answer_mode.title()} + Gemini'], [score], width=.5)
                ax.set_ylim(0, 1.12)
                ax.set_ylabel('Token F1 (0–1)')
                ax.set_title(f'FinanceBench Answer Quality (successful n={summary[answer_mode]["answered"]})')
                ax.grid(axis='y', alpha=0.2)
                ax.bar_label(bars, fmt='%.3f', padding=3)
                fig.tight_layout()
                fig.savefig(ART / 'answer_quality.png', dpi=160)
                plt.close(fig)
    except ImportError:
        print('matplotlib unavailable; plot skipped')

    print('\nFinal metrics:')
    print(json.dumps(summary, indent=2))
    print(f'\nSaved: {ART / "evaluation.csv"}, {ART / "metrics.json"}, {ART / "comparison.png"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='0 means all public questions')
    parser.add_argument('--answers', action='store_true', help='Also evaluate Gemini-generated answers')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--answer-mode', choices=['dense', 'hybrid', 'hybrid_balanced', 'page_hybrid'], default='dense',
                        help='Retriever used for Gemini answer evaluation (default: dense)')
    parser.add_argument('--answer-delay', type=float, default=4.0,
                        help='Seconds between Gemini answer requests (default: 4)')
    args = parser.parse_args()
    evaluate(args.limit, args.answers, args.k, args.answer_mode, args.answer_delay)
