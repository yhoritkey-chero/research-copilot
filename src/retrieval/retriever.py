def retrieve(query, embedder, store, n_results=5):
    query_embedding = embedder.embed_query(query)
    results = store.query(query_embedding, n_results=n_results)
    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            'text':     results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    return retrieved


def format_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk['metadata']
        parts.append(f'[{i}] {m["title"]} ({m["authors"]}, {m["year"]})\n{chunk["text"]}')
    return '\n\n---\n\n'.join(parts)
