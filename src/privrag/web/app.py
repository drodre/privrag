"""Interfaz web mínima para ingestión y consultas."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from privrag.config import LLMBackend, get_settings
from privrag.ingest import ingest_path
from privrag.rag import answer, retrieve

app = FastAPI(title="privrag", version="0.1.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_llm_backend(raw: str | None) -> LLMBackend | None:
    """Convierte el valor del JSON (string) al enum; evita 422 rígidos del esquema antiguo."""
    if raw is None or not str(raw).strip():
        return None
    try:
        return LLMBackend(str(raw).strip())
    except ValueError:
        allowed = ", ".join(f"'{e.value}'" for e in sorted(LLMBackend, key=lambda x: x.value))
        raise HTTPException(
            status_code=422,
            detail=f"llm_backend inválido. Use uno de: {allowed}",
        ) from None


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_static_dir() / "index.html")


@app.get("/status")
def status_page():
    """Página de estado de la aplicación."""
    from qdrant_client import QdrantClient
    import json

    s = get_settings()
    qdrant_status = "unknown"
    collections = []
    qdrant_version = None

    try:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        client.get_collections()
        qdrant_status = "ok"
        # Obtener lista de colecciones
        collections = [c.name for c in client.get_collections().collections]
        # Obtener versión de Qdrant
        try:
            info = client.get_cluster_info()
            qdrant_version = getattr(info, "status", "unknown")
        except Exception:
            pass
    except Exception as e:
        qdrant_status = str(e)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Status - privrag</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 50px auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .status-ok {{ color: green; font-weight: bold; }}
        .status-error {{ color: red; font-weight: bold; }}
        .card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .label {{ color: #666; font-size: 14px; }}
        .value {{ font-size: 18px; margin-top: 5px; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ background: #fff; padding: 10px; margin: 5px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>🔍 Estado de privrag</h1>
    
    <div class="card">
        <div class="label">Qdrant</div>
        <div class="value">
            <span class="{"status-ok" if qdrant_status == "ok" else "status-error"}">
                {"🟢 Conectado" if qdrant_status == "ok" else f"🔴 Error: {qdrant_status}"}
            </span>
        </div>
        <div class="label">URL</div>
        <div class="value">{s.qdrant_url}</div>
    </div>
    
    <div class="card">
        <div class="label">Colecciones ({len(collections)})</div>
        <div class="value">
            <ul id="collections-list">
                {"".join(f'''<li><span>{c}</span><button onclick="deleteCollection('{c}')" style="margin-left:10px;padding:4px 8px;background:#f85149;color:#fff;border:none;border-radius:4px;cursor:pointer;">Eliminar</button></li>''' for c in collections) or "<li>Sin colecciones</li>"}
            </ul>
        </div>
    </div>

    <script>
    async function deleteCollection(name) {{
        if (!confirm("¿Eliminar colección " + name + "?")) return;
        try {{
            const res = await fetch("/api/collections/" + name, {{ method: "DELETE" }});
            const data = await res.json();
            if (res.ok) {{
                alert("Colección eliminada");
                location.reload();
            }} else {{
                alert("Error: " + (data.detail || res.statusText));
            }}
        }} catch (e) {{
            alert("Error: " + e);
        }}
    }}
    </script>
    
    <div class="card">
        <div class="label">Configuración</div>
        <ul>
            <li><strong>Embedding:</strong> {s.embedding_backend.value}</li>
            <li><strong>LLM:</strong> {s.llm_backend.value}</li>
            <li><strong>Chunk size:</strong> {s.chunk_size}</li>
        </ul>
    </div>
</body>
</html>"""
    return HTMLResponse(html, media_type="text/html")


@app.get("/api/debug/lmstudio")
def debug_lmstudio() -> JSONResponse:
    """Diagnóstico LM Studio: prueba endpoints (solo uso local)."""
    from privrag.debug_lmstudio import run_lmstudio_probe

    return JSONResponse(run_lmstudio_probe())


@app.get("/api/health")
def health_check() -> JSONResponse:
    """Health check: verifica estado de Qdrant y servicios."""
    from qdrant_client import QdrantClient

    s = get_settings()
    qdrant_status = "unknown"
    collections = []

    try:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        client.get_collections()
        qdrant_status = "ok"
        # Obtener lista de colecciones
        collections = [c.name for c in client.get_collections().collections]
    except Exception as e:
        qdrant_status = f"error: {type(e).__name__}"

    return JSONResponse(
        {
            "status": "ok" if qdrant_status == "ok" else "degraded",
            "qdrant": qdrant_status,
            "qdrant_url": s.qdrant_url,
            "collections": collections,
        }
    )


@app.get("/api/collections")
def list_collections() -> JSONResponse:
    """Lista todas las colecciones con información."""
    from qdrant_client import QdrantClient

    s = get_settings()
    collections = []

    try:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        result = client.get_collections()
        collections = [
            {
                "name": c.name,
                "description": (getattr(c, "description", None) or ""),
            }
            for c in result.collections
        ]
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse({"collections": collections})


@app.get("/api/config")
def api_config() -> JSONResponse:
    """Valores por defecto del `.env` (sin claves secretas) para rellenar la UI."""
    s = get_settings()

    # Parsear modelos disponibles
    def parse_models(raw: str) -> list[str]:
        if not raw:
            return []
        return [m.strip() for m in raw.split(",") if m.strip()]

    return JSONResponse(
        {
            "llm_backend": s.llm_backend.value,
            "ollama_model": s.ollama_model,
            "openai_model": s.openai_model,
            "openrouter_model": s.openrouter_model,
            "lm_studio_base_url": s.lm_studio_base_url,
            "lm_studio_model": s.lm_studio_model,
            "llm_max_tokens": s.llm_max_tokens,
            "llm_citations": s.llm_citations,
            # Modelos disponibles para la UI
            "available_ollama_models": parse_models(s.available_ollama_models),
            "available_openai_models": parse_models(s.available_openai_models),
            "available_openrouter_models": parse_models(s.available_openrouter_models),
            "available_lm_studio_models": parse_models(s.available_lm_studio_models),
        }
    )


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = "docs"
    limit: int = Field(5, ge=1, le=50)
    game: str | None = None
    source_prefix: str | None = None
    no_llm: bool = False
    llm_backend: str | None = None
    llm_model: str | None = None
    max_tokens: int | None = Field(None, ge=1, le=256000)
    include_citations: bool = True


@app.post("/api/ingest")
async def api_ingest(
    files: list[UploadFile] = File(...),
    collection: str = Form("docs"),
    game: str | None = Form(None),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Sube al menos un archivo.")

    game = (game or "").strip() or None

    tmp = Path(tempfile.mkdtemp(prefix="privrag_ingest_"))
    try:
        written = 0
        for uf in files:
            raw_name = Path(uf.filename or "").name
            if not raw_name:
                continue
            dest = tmp / raw_name
            data = await uf.read()
            dest.write_bytes(data)
            written += 1
        if written == 0:
            raise HTTPException(status_code=400, detail="No se pudo guardar ningún archivo válido.")

        try:
            results = ingest_path(tmp, collection, game)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total = sum(n for _, n in results)
    return JSONResponse(
        {
            "ok": True,
            "collection": collection,
            "files": [{"path": p, "chunks": n} for p, n in results],
            "total_chunks": total,
        }
    )


@app.post("/api/query")
def api_query(body: QueryBody) -> JSONResponse:
    prefix = None
    if body.source_prefix and body.source_prefix.strip():
        prefix = str(Path(body.source_prefix.strip()).expanduser().resolve())

    game = (body.game or "").strip() or None

    if body.no_llm:
        hits = retrieve(
            body.question,
            body.collection,
            limit=body.limit,
            filter_game=game,
            source_path_prefix=prefix,
        )
        return JSONResponse(
            {
                "answer": None,
                "include_citations": True,
                "hits": [
                    {
                        "score": h.score,
                        "source_path": h.payload.get("source_path", ""),
                        "text": h.text,
                    }
                    for h in hits
                ],
            }
        )

    try:
        hits, reply = answer(
            body.question,
            body.collection,
            limit=body.limit,
            use_llm=True,
            filter_game=game,
            source_path_prefix=prefix,
            llm_backend=_parse_llm_backend(body.llm_backend),
            llm_model=(body.llm_model or "").strip() or None,
            max_tokens=body.max_tokens,
            include_citations=body.include_citations,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(
        {
            "answer": reply,
            "include_citations": body.include_citations,
            "hits": [
                {
                    "score": h.score,
                    "source_path": h.payload.get("source_path", ""),
                    "text": h.text,
                }
                for h in hits
            ],
        }
    )


@app.delete("/api/collections/{collection}")
def delete_collection(collection: str) -> JSONResponse:
    """Elimina una colección de Qdrant."""
    from qdrant_client import QdrantClient

    if not collection or not collection.strip():
        raise HTTPException(status_code=400, detail="Nombre de colección requerido")

    s = get_settings()
    try:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        client.delete_collection(collection)
        return JSONResponse({"ok": True, "collection": collection, "deleted": True})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
