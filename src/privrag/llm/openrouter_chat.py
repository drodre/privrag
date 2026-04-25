import httpx

from privrag.llm.base import LLM
from privrag.llm.tokens import resolve_max_tokens


class OpenRouterChatLLM(LLM):
    """Cliente OpenRouter (API tipo OpenAI: POST .../chat/completions)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        extra_headers: dict[str, str] | None = None,
        *,
        timeout: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base = base_url.rstrip("/")
        self._extra = extra_headers or {}
        self._timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        mt = resolve_max_tokens(max_tokens)
        body: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if mt is not None:
            body["max_tokens"] = mt
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(
                    f"{self._base}/chat/completions",
                    headers=headers,
                    json=body,
                )
                r.raise_for_status()
                data = r.json()
        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Timeout del LLM OpenRouter tras {self._timeout}s. "
                "Sube LLM_TIMEOUT o el campo 'Timeout LLM' en la consulta."
            ) from e
        return str(data["choices"][0]["message"]["content"])
