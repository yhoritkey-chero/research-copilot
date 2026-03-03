"""
Vector store simple basado en numpy (reemplaza ChromaDB por incompatibilidad con Python 3.14).
Misma interfaz que ChromaVectorStore original.
"""
import os
import pickle
import numpy as np


class ChromaVectorStore:
    def __init__(self, persist_directory='./chroma_db'):
        self.persist_dir = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self._collections = {}

    def _collection_file(self, name):
        return os.path.join(self.persist_dir, f'{name}.pkl')

    def create_collection(self, name):
        path = self._collection_file(name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._collections[name] = pickle.load(f)
        else:
            self._collections[name] = {
                'ids': [], 'embeddings': [], 'documents': [], 'metadatas': []
            }
        self._active = name

    def _save(self):
        path = self._collection_file(self._active)
        with open(path, 'wb') as f:
            pickle.dump(self._collections[self._active], f)

    def add_documents(self, ids, documents, embeddings, metadatas):
        col = self._collections[self._active]
        existing = set(col['ids'])
        for i in range(len(ids)):
            if ids[i] not in existing:
                col['ids'].append(ids[i])
                col['embeddings'].append(embeddings[i])
                col['documents'].append(documents[i])
                col['metadatas'].append(metadatas[i])
        self._save()

    def query(self, query_embedding, n_results=5, where=None):
        col = self._collections[self._active]
        if not col['embeddings']:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        ids        = col['ids']
        embeddings = col['embeddings']
        documents  = col['documents']
        metadatas  = col['metadatas']

        # Filtrar por metadatos si se especifica 'where'
        if where:
            mask = []
            for meta in metadatas:
                match = all(meta.get(k) == v for k, v in where.items())
                mask.append(match)
            indices = [i for i, m in enumerate(mask) if m]
        else:
            indices = list(range(len(embeddings)))

        if not indices:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        q    = np.array(query_embedding, dtype=np.float32)
        embs = np.array([embeddings[i] for i in indices], dtype=np.float32)

        # Similitud coseno
        q_norm    = q / (np.linalg.norm(q) + 1e-10)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        sims      = embs_norm @ q_norm

        n       = min(n_results, len(sims))
        top_pos = np.argsort(sims)[::-1][:n]
        top_idx = [indices[p] for p in top_pos]

        return {
            'documents': [[documents[i] for i in top_idx]],
            'metadatas': [[metadatas[i] for i in top_idx]],
            'distances': [[float(1 - sims[p]) for p in top_pos]],
        }

    def count(self):
        return len(self._collections[self._active]['ids'])

    def delete_collection(self):
        name = self._active
        self._collections[name] = {
            'ids': [], 'embeddings': [], 'documents': [], 'metadatas': []
        }
        path = self._collection_file(name)
        if os.path.exists(path):
            os.remove(path)
