from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingBackend(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class LLMBackend(StrEnum):
    NONE = "none"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LM_STUDIO = "lmstudio"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    embedding_backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.LOCAL,
        alias="EMBEDDING_BACKEND",
    )
    local_embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    llm_backend: LLMBackend = Field(default=LLMBackend.OLLAMA, alias="LLM_BACKEND")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    # Límite aproximado de caracteres de contexto enviados al LLM (evita 400/413 en modelos con ventana pequeña)
    llm_max_context_chars: int = Field(default=24000, alias="LLM_MAX_CONTEXT_CHARS")
    # Máximo de tokens de salida del LLM (None = dejar el valor por defecto de cada API/modelo)
    llm_max_tokens: int | None = Field(default=None, alias="LLM_MAX_TOKENS", ge=1)
    # Si es False: prompt sin pedir citas y contexto sin rutas (menos tokens de entrada/salida)
    llm_citations: bool = Field(default=True, alias="LLM_CITATIONS")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # OpenRouter (API compatible OpenAI: https://openrouter.ai/docs)
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="OPENROUTER_MODEL",
    )
    openrouter_http_referer: str | None = Field(default=None, alias="OPENROUTER_HTTP_REFERER")
    openrouter_app_title: str | None = Field(default=None, alias="OPENROUTER_APP_TITLE")

    # LM Studio: API compatible OpenAI (ruta /v1; el puerto lo muestra la app al iniciar el servidor)
    lm_studio_base_url: str = Field(default="http://127.0.0.1:41343/v1", alias="LM_STUDIO_BASE_URL")
    lm_studio_api_key: str = Field(default="lm-studio", alias="LM_STUDIO_API_KEY")
    lm_studio_model: str = Field(default="", alias="LM_STUDIO_MODEL")

    # Modelos disponibles para la UI (separados por coma)
    available_ollama_models: str = Field(default="llama3.2", alias="AVAILABLE_OLLAMA_MODELS")
    available_openai_models: str = Field(
        default="gpt-4o-mini,gpt-4o,gpt-4o-mini", alias="AVAILABLE_OPENAI_MODELS"
    )
    available_openrouter_models: str = Field(
        default="openai/gpt-4o-mini,nvidia/nemotron-3-super-120b-a12b:free",
        alias="AVAILABLE_OPENROUTER_MODELS",
    )
    available_lm_studio_models: str = Field(default="", alias="AVAILABLE_LM_STUDIO_MODELS")

    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    @field_validator("lm_studio_base_url", mode="after")
    @classmethod
    def normalize_lm_studio_base_url(cls, v: str) -> str:
        """LM Studio expone POST .../v1/chat/completions; sin /v1 se acaba en /chat/completions y falla."""
        s = str(v).strip().rstrip("/")
        if not s.endswith("/v1"):
            s = f"{s}/v1"
        return s


@lru_cache
def get_settings() -> Settings:
    return Settings()
