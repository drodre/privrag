from sentence_transformers import SentenceTransformer

from privrag.embed.base import Embedder


def _sentence_transformer_dim(model: SentenceTransformer) -> int:
    """Compatible con distintas versiones de sentence-transformers."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    if hasattr(model, "get_sentence_embedding_dimension"):
        return int(model.get_sentence_embedding_dimension())
    vec = model.encode("x", convert_to_numpy=True)
    return int(vec.shape[0])


class LocalSentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_id(self) -> str:
        return f"local:{self._model_name}"

    @property
    def vector_size(self) -> int:
        return _sentence_transformer_dim(self._model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return [v.tolist() for v in vectors]
