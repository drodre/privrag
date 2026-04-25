from abc import ABC, abstractmethod


class Embedder(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identificador estable para metadata y migraciones."""
        ...

    @property
    @abstractmethod
    def vector_size(self) -> int:
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...
