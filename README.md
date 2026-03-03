# Research Copilot

Sistema RAG para consultar 20 papers académicos sobre **regulación de IA y privacidad de datos**.

## Estructura

```
research-copilot/
├── papers/paper_catalog.json   ← catálogo de 20 papers
├── src/
│   ├── ingestion/              ← PDF extractor + text cleaner
│   ├── chunking/               ← TokenChunker (256/512/1024 tokens)
│   ├── embedding/              ← OpenAIEmbedder
│   ├── vectorstore/            ← ChromaVectorStore
│   ├── retrieval/              ← retrieve() + format_context()
│   ├── generation/             ← 4 prompts (v1–v4) + generate()
│   └── rag_pipeline.py         ← RAGPipeline integrando todo
├── prompts/                    ← 4 estrategias de prompt en texto
├── app/                        ← Streamlit multipage
├── eval/                       ← 10 preguntas + evaluate.py
├── scripts/ingest_papers.py    ← indexa los 20 papers
└── tests/                      ← tests unitarios
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY
```

## Uso

### 1. Indexar papers
```bash
cd research-copilot
python scripts/ingest_papers.py
```

### 2. Lanzar la app
```bash
streamlit run app/main.py
```

### 3. Tests
```bash
pytest tests/
```

### 4. Evaluación
```bash
python eval/evaluate.py
```

## Estrategias de Prompt

| Versión | Técnica | Uso recomendado |
|---------|---------|-----------------|
| v1 | Delimitadores | Respuestas generales académicas |
| v2 | JSON output | Integración programática |
| v3 | Few-shot | Consistencia en formato de respuesta |
| v4 | Chain of thought | Preguntas complejas de síntesis |

## Notas

- Los PDFs están en `../papers/` (relativo al proyecto = `Downloads/papers/`)
- Los archivos tienen extensión `.pdf.pdf`
- ChromaDB persiste en `./chroma_db/`
