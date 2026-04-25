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
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base = base_url.rstrip("/")
        self._extra = extra_headers or {}

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
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{self._base}/chat/completions",
                headers=headers,
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        return str(data["choices"][0]["message"]["content"])
