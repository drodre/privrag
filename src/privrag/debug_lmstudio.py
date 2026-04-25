"""Pruebas de conectividad contra LM Studio (endpoints OpenAI vs API nativa)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from privrag.llm.lm_studio import _http_origin


def run_lmstudio_probe(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Devuelve un dict con `origin`, `config` y `probes` (lista de intentos).

    Cada probe: method, url, status (o null si error de red), body_snippet o error.
    """
    from privrag.config import get_settings

    s = get_settings()
    raw_base = (base_url or s.lm_studio_base_url).strip()
    key = api_key if api_key is not None else s.lm_studio_api_key
    m = (model or s.lm_studio_model or "").strip() or "test-model"

    try:
        origin = _http_origin(raw_base)
    except ValueError as e:
        return {
            "error": str(e),
            "lm_studio_base_url": raw_base,
            "probes": [],
        }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    probes: list[dict[str, Any]] = []

    def record(
        method: str,
        url: str,
        *,
        json_body: dict | None = None,
        use_auth: bool = True,
    ) -> None:
        h = headers if use_auth else {"Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=20.0) as client:
                if method == "GET":
                    r = client.get(url, headers=h)
                else:
                    r = client.post(url, headers=h, json=json_body or {})
            text = (r.text or "")[:500].replace("\n", " ")
            probe: dict[str, Any] = {
                "method": method,
                "url": url,
                "status": r.status_code,
                "content_type": r.headers.get("content-type", ""),
                "body_snippet": text,
            }
            if r.headers.get("content-type", "").startswith("application/json"):
                try:
                    probe["json_keys"] = list((r.json() or {}).keys())[:20]
                except ValueError:
                    pass
            probes.append(probe)
        except httpx.HTTPError as e:
            probes.append(
                {
                    "method": method,
                    "url": url,
                    "status": None,
                    "error": str(e),
                }
            )

    # OpenAI-compat: listar modelos (ligero)
    record("GET", f"{origin}/v1/models")
    # Raíz del servidor (a veces devuelve HTML o JSON)
    record("GET", f"{origin}/", use_auth=False)

    post_openai = {
        "model": m,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 8,
        "stream": False,
    }
    record("POST", f"{origin}/v1/chat/completions", json_body=post_openai)

    post_native = {
        "model": m,
        "input": "Say OK.",
        "stream": False,
    }
    record("POST", f"{origin}/api/v1/chat", json_body=post_native)

    # Ruta equivocada habitual (sin /v1)
    record("POST", f"{origin}/chat/completions", json_body=post_openai)

    # Modelos nativos LM Studio
    record("GET", f"{origin}/api/v1/models")

    return {
        "lm_studio_base_url": raw_base,
        "origin": origin,
        "model_used_for_post": m,
        "probes": probes,
    }


def format_probe_text(data: dict[str, Any]) -> str:
    lines: list[str] = []
    if "error" in data:
        lines.append(f"Error: {data['error']}")
        return "\n".join(lines)
    lines.append(f"LM_STUDIO_BASE_URL (config): {data.get('lm_studio_base_url')}")
    lines.append(f"Origen (scheme + host:puerto): {data.get('origin')}")
    lines.append(f"Modelo usado en POSTs de prueba: {data.get('model_used_for_post')}")
    lines.append("")
    lines.append("— Pruebas (HTTP) —")
    for i, p in enumerate(data.get("probes") or [], 1):
        lines.append(f"\n[{i}] {p.get('method')} {p.get('url')}")
        if p.get("error"):
            lines.append(f"    red/conexión: {p['error']}")
            continue
        lines.append(f"    status: {p.get('status')}")
        if p.get("json_keys"):
            lines.append(f"    claves JSON: {p['json_keys']}")
        if p.get("body_snippet"):
            lines.append(f"    cuerpo: {p['body_snippet'][:300]}…" if len(p.get("body_snippet", "")) > 300 else f"    cuerpo: {p['body_snippet']}")

    probes = data.get("probes") or []
    statuses = [p.get("status") for p in probes if p.get("error") is None and p.get("status") is not None]
    if statuses and all(s == 404 for s in statuses):
        lines.append("")
        lines.append("AVISO — Todas las pruebas devolvieron 404 con HTML «Cannot GET/POST»:")
        lines.append("  Ese puerto casi seguro NO es el API de LM Studio (suele ser otro programa, p. ej. Express).")
        lines.append("  En LM Studio: pestaña «Developer» → inicia «Start Server» y copia la URL que indica (host + puerto).")
        lines.append("  Actualiza LM_STUDIO_BASE_URL con ese origen + «/v1» (p. ej. http://127.0.0.1:1234/v1 si ahí escucha).")
        lines.append("  Comprueba con: ss -tlnp | grep -E '1234|41343'  (o el puerto que muestre LM Studio).")

    lines.append("")
    lines.append("Interpretación rápida:")
    lines.append("  • GET /v1/models = 200 → API compatible OpenAI activa.")
    lines.append("  • POST /v1/chat/completions = 200 → chat OpenAI OK (modelo debe existir).")
    lines.append("  • POST /api/v1/chat = 200 → API nativa OK.")
    lines.append("  • 404 en /v1/... → otro puerto, servidor no iniciado o versión sin esa ruta.")
    lines.append("  • JSON con error en cuerpo y status 200 → lee el mensaje en body_snippet.")
    return "\n".join(lines)


def format_probe_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
