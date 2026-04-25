from privrag.config import get_settings


def resolve_max_tokens(override: int | None) -> int | None:
    """Prioridad: argumento explícito > `LLM_MAX_TOKENS` en `.env` > None (default de la API)."""
    if override is not None:
        return override
    return get_settings().llm_max_tokens
