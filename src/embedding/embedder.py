from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, model='text-embedding-3-small'):
        self.client = OpenAI()
        self.model = model

    def embed_texts(self, texts):
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query):
        return self.embed_texts([query])[0]
