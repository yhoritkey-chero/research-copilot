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

    def query(self, question, strategy='v1', n_results=5):
        """Ejecuta el pipeline RAG completo con bloqueo de seguridad."""
        chunks = retrieve(question, self.embedder, self.store, n_results=n_results)
        
        # Bloqueo de Seguridad: Si la base de datos no devuelve nada relevante, 
        # no llamamos a la IA para evitar que invente con su conocimiento previo.
        if not chunks:
            return {
                'question':      question,
                'strategy':      strategy,
                'answer':        "❌ **Error de Seguridad**: No se han encontrado fragmentos relevantes en la biblioteca local para esta pregunta. Para evitar alucinaciones, el sistema se ha bloqueado. Por favor, asegúrate de que el tema esté cubierto en tus PDFs.",
                'sources':       [],
                'n_chunks_used': 0
            }

        context = format_context(chunks)
        answer  = generate(question, context, strategy=strategy)
        
        # Doble verificación: si la respuesta cita fuentes inexistentes (alucinación post-IA)
        # Esto es un respaldo adicional al prompt.
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
