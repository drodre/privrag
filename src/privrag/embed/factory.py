from privrag.config import EmbeddingBackend, get_settings
from privrag.embed.base import Embedder
from privrag.embed.local import LocalSentenceTransformerEmbedder
from privrag.embed.openai_embed import OpenAIEmbedder


def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == EmbeddingBackend.LOCAL:
        return LocalSentenceTransformerEmbedder(s.local_embedding_model)
    if s.embedding_backend == EmbeddingBackend.OPENAI:
        if not s.openai_api_key:
            raise ValueError("OPENAI_API_KEY requerido para EMBEDDING_BACKEND=openai")
        return OpenAIEmbedder(s.openai_api_key, s.openai_embedding_model)
    raise ValueError(f"Backend de embeddings no soportado: {s.embedding_backend}")
