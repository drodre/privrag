import httpx

from privrag.llm.base import LLM
from privrag.llm.tokens import resolve_max_tokens


def _parse_openai_compat(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def _parse_native_chat(data: dict) -> str:
    return str((data.get("message") or {}).get("content") or "")


def _normalize_ollama_base(url: str) -> str:
    """Evita URLs tipo http://host:11434/v1 que duplican /v1/v1/...."""
    u = url.strip().rstrip("/")
    for suf in ("/v1", "/api"):
        if u.endswith(suf):
            return u[: -len(suf)]
    return u


def _resp_hint(resp: httpx.Response) -> str:
    t = (resp.text or "").strip().replace("\n", " ")[:500]
    return t if t else "(cuerpo vacío)"


class OllamaLLM(LLM):
    def __init__(self, base_url: str, model: str, *, timeout: int) -> None:
        self._base = _normalize_ollama_base(base_url)
        self._model = model
        self._timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        mt = resolve_max_tokens(max_tokens)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        base_chat: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        payload_v1 = dict(base_chat)
        payload_native = dict(base_chat)
        if mt is not None:
            payload_v1["max_tokens"] = mt
            payload_native["options"] = {"num_predict": mt}
        headers = {
            "Content-Type": "application/json",
            # Varios despliegues / proxies lo esperan; Ollama lo ignora en local
            "Authorization": "Bearer ollama",
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                # 1) API compatible con OpenAI
                r = client.post(
                    f"{self._base}/v1/chat/completions",
                    json=payload_v1,
                    headers=headers,
                )
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError as e:
                        raise RuntimeError(
                            f"Ollama devolvió 200 pero no es JSON válido: {_resp_hint(r)}"
                        ) from e
                    err = data.get("error")
                    if err:
                        raise RuntimeError(
                            f"Ollama (error en JSON): {err!r}. Base URL usada: {self._base!r}, modelo: {self._model!r}"
                        )
                    return _parse_openai_compat(data)

                # 2) API nativa /api/chat
                r2 = client.post(
                    f"{self._base}/api/chat",
                    json=payload_native,
                    headers=headers,
                )
                if r2.status_code == 200:
                    data = r2.json()
                    err = data.get("error")
                    if err:
                        raise RuntimeError(
                            f"Ollama (error en JSON): {err!r}. Base URL: {self._base!r}, modelo: {self._model!r}"
                        )
                    return _parse_native_chat(data)

                # 3) Ollama antiguo: /api/generate
                prompt = f"{system}\n\n{user}"
                gen_body: dict = {
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                }
                if mt is not None:
                    gen_body["options"] = {"num_predict": mt}
                r3 = client.post(
                    f"{self._base}/api/generate",
                    json=gen_body,
                    headers=headers,
                )
                if r3.status_code == 200:
                    data = r3.json() or {}
                    if data.get("error"):
                        raise RuntimeError(
                            f"Ollama (error en JSON): {data['error']!r}. Base URL: {self._base!r}, modelo: {self._model!r}"
                        )
                    return str(data.get("response") or "")
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Timeout del LLM Ollama tras {self._timeout}s. "
                "Sube LLM_TIMEOUT o el campo 'Timeout LLM' en la consulta."
            ) from e

        raise RuntimeError(
            "Ollama rechazó las tres rutas. Detalle:\n"
            f"  POST {self._base}/v1/chat/completions → {r.status_code}: {_resp_hint(r)}\n"
            f"  POST {self._base}/api/chat → {r2.status_code}: {_resp_hint(r2)}\n"
            f"  POST {self._base}/api/generate → {r3.status_code}: {_resp_hint(r3)}\n"
            f"Modelo configurado: {self._model!r}. Base URL normalizada: {self._base!r}\n"
            "Comprueba OLLAMA_BASE_URL (sin /v1 al final), OLLAMA_MODEL (p. ej. qwen3:latest; "
            "`ollama list`) y que el contexto no exceda el modelo (prueba bajar LLM_MAX_CONTEXT_CHARS)."
        )
