from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        ...
