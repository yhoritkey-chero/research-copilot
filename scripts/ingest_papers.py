"""Script para indexar los 20 papers en ChromaDB."""
import os
import sys

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline

if __name__ == '__main__':
    print('=== Research Copilot — Ingestión de Papers ===\n')
    pipeline = RAGPipeline(
        papers_dir='papers',
        catalog_path='papers/paper_catalog.json',
        chroma_dir='./chroma_db',
    )
    print('Iniciando ingestión (puede tardar ~2-3 min)...\n')
    total = pipeline.ingest()
    print(f'\n[V] ChromaDB listo con {total} chunks.')
