"""Ingestión indexada (archivo o carpeta) reutilizable desde CLI y web."""

from dataclasses import dataclass
from pathlib import Path

from privrag.config import get_settings
from privrag.embed import get_embedder
from privrag.ingest import chunk_text, iter_documents, read_file_text
from privrag.ingest.ocr import is_ocr_document, make_searchable_pdf
from privrag.store import QdrantStore


@dataclass
class IngestResult:
    source_path: str
    chunks: int
    indexed_path: str | None = None
    ocr_output_path: str | None = None
    ocr_status: str | None = None
    ocr_message: str | None = None

    def __iter__(self):
        yield self.source_path
        yield self.chunks


def ingest_path(
    path: Path,
    collection: str,
    topic: str | None = None,
    *,
    ocr_pdf: bool | None = None,
    ocr_language: str | None = None,
    ocr_output_dir: Path | None = None,
    ocr_timeout: int | None = None,
) -> list[IngestResult]:
    """Indexa un archivo o carpeta. Devuelve lista de (ruta absoluta, número de chunks) por archivo."""
    s = get_settings()
    use_ocr = s.ocr_enabled if ocr_pdf is None else ocr_pdf
    language = (ocr_language or s.ocr_language).strip() or s.ocr_language
    output_dir = ocr_output_dir or Path(s.ocr_output_dir)
    timeout = ocr_timeout or s.ocr_timeout
    embedder = get_embedder()
    store = QdrantStore()
    store.ensure_collection(collection, embedder.vector_size)

    files = iter_documents(path, include_images=use_ocr)
    if not files:
        raise ValueError("No se encontraron documentos admitidos en la ruta indicada.")

    results: list[IngestResult] = []
    for fp in files:
        source_path = str(fp.resolve())
        indexed_fp = fp
        ocr_output_path: str | None = None
        ocr_status: str | None = None
        ocr_message: str | None = None
        if use_ocr and is_ocr_document(fp):
            ocr = make_searchable_pdf(
                fp,
                output_dir,
                language=language,
                timeout=timeout,
            )
            indexed_fp = ocr.output_path
            ocr_output_path = str(ocr.output_path)
            ocr_status = ocr.status
            ocr_message = ocr.message

        raw = read_file_text(indexed_fp)
        chunks = chunk_text(raw, s.chunk_size, s.chunk_overlap)
        if not chunks:
            results.append(
                IngestResult(
                    source_path=source_path,
                    chunks=0,
                    indexed_path=str(indexed_fp.resolve()),
                    ocr_output_path=ocr_output_path,
                    ocr_status=ocr_status,
                    ocr_message=ocr_message,
                )
            )
            continue
        vectors = embedder.encode(chunks)
        common: dict = {}
        if topic:
            common["topic"] = topic
        if ocr_output_path:
            common["ocr_source_path"] = source_path
            common["ocr_output_path"] = ocr_output_path
            common["ocr_status"] = ocr_status
        store.upsert_chunks(
            collection=collection,
            texts=chunks,
            vectors=vectors,
            common_payload=common,
            source_path=str(indexed_fp.resolve()),
            embedding_model=embedder.model_id,
        )
        results.append(
            IngestResult(
                source_path=source_path,
                chunks=len(chunks),
                indexed_path=str(indexed_fp.resolve()),
                ocr_output_path=ocr_output_path,
                ocr_status=ocr_status,
                ocr_message=ocr_message,
            )
        )
    return results
