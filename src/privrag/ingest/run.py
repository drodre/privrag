"""Ingestión indexada (archivo o carpeta) reutilizable desde CLI y web."""

from pathlib import Path

from privrag.config import get_settings
from privrag.embed import get_embedder
from privrag.ingest import chunk_text, iter_documents, read_file_text
from privrag.store import QdrantStore


def ingest_path(path: Path, collection: str, game: str | None = None) -> list[tuple[str, int]]:
    """Indexa un archivo o carpeta. Devuelve lista de (ruta absoluta, número de chunks) por archivo."""
    s = get_settings()
    embedder = get_embedder()
    store = QdrantStore()
    store.ensure_collection(collection, embedder.vector_size)

    files = iter_documents(path)
    if not files:
        raise ValueError("No se encontraron documentos admitidos en la ruta indicada.")

    results: list[tuple[str, int]] = []
    for fp in files:
        raw = read_file_text(fp)
        chunks = chunk_text(raw, s.chunk_size, s.chunk_overlap)
        if not chunks:
            results.append((str(fp.resolve()), 0))
            continue
        vectors = embedder.encode(chunks)
        common: dict = {}
        if game:
            common["game"] = game
        store.upsert_chunks(
            collection=collection,
            texts=chunks,
            vectors=vectors,
            common_payload=common,
            source_path=str(fp.resolve()),
            embedding_model=embedder.model_id,
        )
        results.append((str(fp.resolve()), len(chunks)))
    return results
