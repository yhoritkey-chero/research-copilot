import os
import sys
import traceback
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.getcwd())

load_dotenv()

from src.rag_pipeline import RAGPipeline

def test():
    try:
        print(f"CWD: {os.getcwd()}")
        p = RAGPipeline(papers_dir='papers', catalog_path='papers/paper_catalog.json')
        print(f"Catalog loaded with {len(p.catalog['papers'])} papers")
        
        # Test just the first paper
        paper = p.catalog['papers'][0]
        pdf_path = os.path.join('papers', paper['filename'])
        print(f"Testing paper 0: {paper['id']} | Path: {pdf_path}")
        print(f"Path exists: {os.path.exists(pdf_path)}")
        
        total = p.ingest()
        print(f"Ingested {total} chunks")
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    test()
