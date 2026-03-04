import os
import json
from urllib.parse import unquote

from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.text_cleaner import clean_extracted_text
from src.chunking.chunker import TokenChunker
from src.embedding.embedder import OpenAIEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.retriever import retrieve, format_context
from src.generation.generator import generate


def get_pdf_path(paper, papers_dir):
    """Extrae solo el nombre del archivo y lo combina con papers_dir."""
    filename = os.path.basename(paper['filename'])
    filename = unquote(filename)  # decodifica %20, %2C, etc.
    return os.path.join(papers_dir, filename)


class RAGPipeline:
    def __init__(
        self,
        papers_dir='papers',
        catalog_path='papers/paper_catalog.json',
        chroma_dir='./chroma_db',
        chunk_size=512,
    ):
        self.papers_dir = papers_dir
        self.catalog_path = catalog_path

        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)

        self.chunker = TokenChunker(chunk_size=chunk_size)
        self.embedder = OpenAIEmbedder()
        self.store = ChromaVectorStore(persist_directory=chroma_dir)
        self.store.create_collection('research_papers')

    def ingest(self):
        """Indexa todos los papers del catálogo en ChromaDB."""
        total_chunks = 0
        failed = []
        for paper in self.catalog['papers']:
            pdf_path = get_pdf_path(paper, self.papers_dir)
            if not os.path.exists(pdf_path):
                print(f'  [X] {paper["id"]} - archivo no encontrado: {os.path.basename(pdf_path)}')
                failed.append(paper['id'])
                continue
            try:
                raw        = extract_text_from_pdf(pdf_path)
                clean_text = clean_extracted_text(raw['text'])
                metadata   = {
                    'paper_id': paper['id'],
                    'title':    paper['title'],
                    'authors':  ', '.join(paper['authors']),
                    'year':     int(paper['year']),
                    'venue':    paper['venue'],
                    'topics':   ', '.join(paper.get('topics', []))
                }
                chunks     = self.chunker.chunk_text(clean_text, metadata=metadata)
                texts      = [c['text'] for c in chunks]
                embeddings = self.embedder.embed_texts(texts)
                self.store.add_documents(
                    ids        = [f'{paper["id"]}_chunk_{c["chunk_id"]}' for c in chunks],
                    documents  = texts,
                    embeddings = embeddings,
                    metadatas  = [c['metadata'] for c in chunks]
                )
                total_chunks += len(chunks)
                print(f'  [V] {paper["id"]} - {len(chunks)} chunks | {paper["title"][:55]}...')
            except Exception as e:
                print(f'  [X] {paper["id"]} - ERROR: {e}')
                failed.append(paper['id'])
        print(f'\nIngestión completa: {total_chunks} chunks totales')
        if failed:
            print(f'Papers con error: {failed}')
        return total_chunks

    def query(self, question, strategy='v1', n_results=5, threshold=0.55):
        """Ejecuta el pipeline RAG con bloqueo por umbral de similitud."""
        all_chunks = retrieve(question, self.embedder, self.store, n_results=n_results)
        
        # Filtrar por distancia: solo nos quedamos con los que realmente se parecen a la pregunta
        chunks = [c for c in all_chunks if c['distance'] < threshold]
        
        # Bloqueo de Seguridad: Si no hay nada suficientemente cerca
        if not chunks:
            return {
                'question':      question,
                'strategy':      strategy,
                'answer':        "❌ **Información no encontrada**: No hay fragmentos en la base de datos que hablen con suficiente precisión sobre este tema. El sistema ha bloqueado la respuesta para evitar alucinaciones.",
                'sources':       [],
                'n_chunks_used': 0
            }

        context = format_context(chunks)
        answer  = generate(question, context, strategy=strategy)
        
        # Asegurar que las fuentes mostradas son solo las que pasaron el filtro
        titles_in_db = {c['metadata']['title'] for c in chunks}
        
        return {
            'question':      question,
            'strategy':      strategy,
            'answer':        answer,
            'sources':       list(titles_in_db),
            'n_chunks_used': len(chunks)
        }



    def get_all_paper_titles(self):
        """Retorna lista de (id, title) para la UI."""
        return [(p['id'], p['title']) for p in self.catalog['papers']]
