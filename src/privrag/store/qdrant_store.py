from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from privrag.config import get_settings


@dataclass
class SearchHit:
    score: float
    text: str
    payload: dict


class QdrantStore:
    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        s = get_settings()
        key = api_key if api_key is not None else s.qdrant_api_key
        kwargs: dict = {"url": url or s.qdrant_url, "check_compatibility": False}
        if key:
            kwargs["api_key"] = key
        self._client = QdrantClient(**kwargs)

    def ensure_collection(self, name: str, vector_size: int) -> None:
        if self._client.collection_exists(name):
            info = self._client.get_collection(name)
            existing = info.config.params.vectors
            existing_size: int | None = None
            if isinstance(existing, qm.VectorParams):
                existing_size = existing.size
            elif isinstance(existing, dict) and existing:
                first = next(iter(existing.values()))
                if isinstance(first, qm.VectorParams):
                    existing_size = first.size
            if existing_size is not None and existing_size != vector_size:
                raise ValueError(
                    f"La colección {name!r} tiene dimensión {existing_size}, "
                    f"pero el embedder actual produce {vector_size}. "
                    "Usa otra colección o re-embeda con el mismo modelo."
                )
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def upsert_chunks(
        self,
        collection: str,
        texts: list[str],
        vectors: list[list[float]],
        common_payload: dict,
        source_path: str,
        embedding_model: str,
    ) -> None:
        if len(texts) != len(vectors):
            raise ValueError("texts y vectors deben tener la misma longitud")
        points: list[qm.PointStruct] = []
        for i, (text, vec) in enumerate(zip(texts, vectors, strict=True)):
            payload = {
                **common_payload,
                "text": text,
                "source_path": source_path,
                "chunk_index": i,
                "embedding_model": embedding_model,
            }
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                )
            )
        self._client.upload_points(collection_name=collection, points=points)

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        filter_topic: str | None = None,
        source_path_prefix: str | None = None,
    ) -> list[SearchHit]:
        """Búsqueda por similitud. `filter_topic` aplica filtro en Qdrant; `source_path_prefix`
        filtra por prefijo de `source_path` en cliente (tras ampliar el límite de recuperación).
        """
        must: list[qm.FieldCondition] = []
        if filter_topic is not None:
            must.append(
                qm.FieldCondition(key="topic", match=qm.MatchValue(value=filter_topic)),
            )
        query_filter: qm.Filter | None = qm.Filter(must=must) if must else None

        fetch_limit = limit
        if source_path_prefix:
            fetch_limit = min(max(limit * 25, 50), 500)

        res = self._client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
            search_params=qm.SearchParams(hnsw_ef=128),
        )
        hits: list[SearchHit] = []
        for p in res.points or []:
            score = float(p.score) if p.score is not None else 0.0
            pl = p.payload or {}
            text = str(pl.get("text", ""))
            hits.append(SearchHit(score=score, text=text, payload=dict(pl)))

        if source_path_prefix:
            norm = str(Path(source_path_prefix).expanduser().resolve())
            hits = [h for h in hits if str(h.payload.get("source_path", "")).startswith(norm)]

        return hits[:limit]
