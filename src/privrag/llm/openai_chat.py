import httpx

from privrag.llm.base import LLM
from privrag.llm.tokens import resolve_max_tokens


class OpenAIChatLLM(LLM):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

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
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]
