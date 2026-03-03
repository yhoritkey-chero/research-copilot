"""Evaluación del RAG Pipeline con las 4 estrategias de prompt."""
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline


def evaluate(strategies=None, n_results=5):
    if strategies is None:
        strategies = ['v1', 'v2', 'v3', 'v4']

    with open('eval/questions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data['questions']

    pipeline = RAGPipeline(
        papers_dir='papers',
        catalog_path='papers/paper_catalog.json',
    )

    results = []
    for q in questions:
        print(f'\n[{q["id"]}] {q["question"][:70]}...')
        for strategy in strategies:
            print(f'  -> Estrategia {strategy}...', end=' ', flush=True)
            result = pipeline.query(q['question'], strategy=strategy, n_results=n_results)
            results.append({
                'question_id': q['id'],
                'question':    q['question'],
                'strategy':    strategy,
                'answer':      result['answer'],
                'sources':     result['sources'],
                'n_chunks':    result['n_chunks_used'],
            })
            print(f'OK ({len(result["sources"])} fuentes)')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'eval/results_{timestamp}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n[V] Resultados guardados en {out_path}')
    print(f'  Total evaluaciones: {len(results)} ({len(questions)} preguntas × {len(strategies)} estrategias)')
    return results


if __name__ == '__main__':
    evaluate()
