from pathlib import Path

import typer

from privrag.debug_lmstudio import format_probe_json, format_probe_text, run_lmstudio_probe
from privrag.ingest import ingest_path
from privrag.rag import answer, retrieve

app = typer.Typer(no_args_is_help=True, help="RAG para documentos de RPG (Qdrant + embeddings + LLM opcional)")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Archivo o carpeta con .md, .txt, .pdf o imágenes"),
    collection: str = typer.Option("docs", "--collection", "-c", help="Nombre de la colección Qdrant"),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Metadato opcional (ej. dnd5e)"),
    ocr: bool = typer.Option(False, "--ocr", help="Genera PDFs buscables con OCR antes de ingestar"),
    ocr_language: str | None = typer.Option(None, "--ocr-language", help="Idiomas Tesseract, ej. spa+eng"),
    ocr_output_dir: Path | None = typer.Option(None, "--ocr-output-dir", help="Directorio para PDFs OCR"),
    ocr_timeout: int | None = typer.Option(None, "--ocr-timeout", help="Timeout OCR por documento"),
) -> None:
    """Indexa documentos en Qdrant (chunk → embed → upsert)."""
    try:
        results = ingest_path(
            path,
            collection,
            topic,
            ocr_pdf=ocr,
            ocr_language=ocr_language,
            ocr_output_dir=ocr_output_dir,
            ocr_timeout=ocr_timeout,
        )
    except (RuntimeError, TimeoutError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    for item in results:
        typer.echo(f"Ingestando {item.source_path} …")
        if item.ocr_output_path:
            typer.echo(f"  OCR → {item.ocr_output_path}")
        if item.chunks == 0:
            typer.echo("  (vacío, omitido)")
        else:
            typer.echo(f"  → {item.chunks} chunks")
    typer.echo("Listo.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Pregunta en lenguaje natural"),
    collection: str = typer.Option("docs", "--collection", "-c"),
    limit: int = typer.Option(5, "--limit", "-k", help="Fragmentos a recuperar"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Solo mostrar pasajes recuperados"),
    topic: str | None = typer.Option(
        None,
        "--topic",
        "-t",
        help="Solo chunks con este metadato (debe coincidir con ingest --topic)",
    ),
    source_prefix: Path | None = typer.Option(
        None,
        "--source-prefix",
        help="Solo fuentes cuya ruta absoluta empieza por esta (archivo o carpeta)",
    ),
    no_citations: bool = typer.Option(
        False,
        "--no-citations",
        help="Sin citas en la respuesta del modelo; contexto más compacto (menos tokens)",
    ),
) -> None:
    """Recupera contexto y opcionalmente genera respuesta con el LLM configurado."""
    prefix = str(source_prefix.resolve()) if source_prefix else None
    if no_llm:
        hits = retrieve(
            question,
            collection,
            limit=limit,
            filter_topic=topic,
            source_path_prefix=prefix,
        )
        for h in hits:
            typer.echo(f"score={h.score:.4f}  {h.payload.get('source_path', '')}")
            typer.echo(h.text[:2000])
            if len(h.text) > 2000:
                typer.echo("…")
            typer.echo()
        return

    hits, reply = answer(
        question,
        collection,
        limit=limit,
        use_llm=True,
        filter_topic=topic,
        source_path_prefix=prefix,
        include_citations=not no_citations,
    )
    if reply is None:
        for h in hits:
            typer.echo(f"score={h.score:.4f}  {h.payload.get('source_path', '')}")
            typer.echo(h.text[:2000])
            typer.echo()
        return
    typer.echo(reply)
    typer.echo()
    if not no_citations:
        typer.echo("--- Fuentes ---")
        for h in hits:
            typer.echo(f"  {h.score:.4f}  {h.payload.get('source_path', '')}")


@app.command("lmstudio-probe")
def lmstudio_probe(
    json_out: bool = typer.Option(False, "--json", help="Salida en JSON"),
    model: str | None = typer.Option(None, "--model", "-m", help="Modelo para POSTs de prueba"),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Sobrescribe LM_STUDIO_BASE_URL (p. ej. http://127.0.0.1:1234/v1)",
    ),
) -> None:
    """Prueba rutas habituales contra LM Studio (GET /v1/models, POST chat, etc.)."""
    data = run_lmstudio_probe(base_url=base_url, model=model)
    typer.echo(format_probe_json(data) if json_out else format_probe_text(data))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
