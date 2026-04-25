from privrag.llm.base import LLM


class NoopLLM(LLM):
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        return ""
