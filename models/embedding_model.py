from sentence_transformers import SentenceTransformer
from config import Config


class EmbeddingModel:
    def __init__(self):
        self.config = Config()
        self.model = SentenceTransformer(
            self.config.EMBEDDING_MODEL, trust_remote_code=True
        )

    def encode(self, texts):
        return self.model.encode(texts)
