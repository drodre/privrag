import httpx

from privrag.embed.base import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._vector_size: int | None = None

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            vecs = self.encode(["dimension probe"])
            self._vector_size = len(vecs[0])
        return self._vector_size

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # La API admite batch; troceamos por si hay muchos textos largos.
        batch = 64
        with httpx.Client(timeout=120.0) as client:
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                r = client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "input": chunk},
                )
                r.raise_for_status()
                data = r.json()["data"]
                data.sort(key=lambda x: x["index"])
                out.extend(item["embedding"] for item in data)
        return out
