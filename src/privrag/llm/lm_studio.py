"""LM Studio: OpenAI-compatible `/v1/chat/completions` con fallback a API nativa `/api/v1/chat`."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from privrag.llm.base import LLM
from privrag.llm.tokens import resolve_max_tokens


def _parse_openai_compat(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def _http_origin(base_url: str) -> str:
    p = urlparse(base_url.strip())
    if not p.scheme or not p.netloc:
        raise ValueError(f"LM_STUDIO_BASE_URL inválida: {base_url!r}")
    return f"{p.scheme}://{p.netloc}"


def _parse_native_v1_chat(data: dict) -> str:
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    parts: list[str] = []
    for item in data.get("output") or []:
        if item.get("type") == "message" and item.get("content"):
            parts.append(str(item["content"]))
    return "\n".join(parts).strip()


class LMStudioLLM(LLM):
    """POST a `/v1/chat/completions`; si 404, POST a `/api/v1/chat` (LM Studio 0.4+)."""

    def __init__(self, base_url: str, model: str, api_key: str, *, timeout: int) -> None:
        self._base = base_url.rstrip("/")
        self._origin = _http_origin(self._base)
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        mt = resolve_max_tokens(max_tokens)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        openai_body: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if mt is not None:
            openai_body["max_tokens"] = mt

        url_openai = f"{self._origin}/v1/chat/completions"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(url_openai, headers=headers, json=openai_body)
                if r.status_code == 200:
                    return _parse_openai_compat(r.json())

                if r.status_code != 404:
                    r.raise_for_status()

                native_body: dict = {
                    "model": self._model,
                    "system_prompt": system,
                    "input": user,
                    "stream": False,
                }
                if mt is not None:
                    native_body["max_output_tokens"] = mt

                url_native = f"{self._origin}/api/v1/chat"
                r2 = client.post(url_native, headers=headers, json=native_body)
                if r2.status_code != 200:
                    hint = (
                        f"LM Studio respondió 404 en {url_openai} y error en {url_native} "
                        f"({r2.status_code}): {(r2.text or '')[:400]}\n"
                        "Comprueba en LM Studio: pestaña Developer → servidor en marcha, "
                        "puerto correcto y modelo cargado. El puerto lo muestra la propia app."
                    )
                    raise RuntimeError(hint)
                return _parse_native_v1_chat(r2.json())
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Timeout del LLM LM Studio tras {self._timeout}s. "
                "Sube LLM_TIMEOUT o el campo 'Timeout LLM' en la consulta."
            ) from e
