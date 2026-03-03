import tiktoken


class TokenChunker:
    def __init__(self, chunk_size=512, chunk_overlap=50, model='gpt-4'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.encoding_for_model(model)

    def count_tokens(self, text):
        return len(self.encoder.encode(text))

    def chunk_text(self, text, metadata=None):
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append({
                'chunk_id': chunk_id, 'text': chunk_text,
                'token_count': len(chunk_tokens),
                'start_token': start, 'end_token': end,
                'metadata': metadata or {}
            })
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        return chunks


# 3 configuraciones disponibles
CHUNKER_SMALL  = TokenChunker(chunk_size=256)   # preguntas factuales precisas
CHUNKER_DEFAULT = TokenChunker(chunk_size=512)  # configuración por defecto
CHUNKER_LARGE  = TokenChunker(chunk_size=1024)  # razonamiento complejo
