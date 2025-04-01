from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def embed_single(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]