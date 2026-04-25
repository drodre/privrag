from privrag.config import LLMBackend, get_settings
from privrag.llm.base import LLM
from privrag.llm.none_llm import NoopLLM
from privrag.llm.ollama import OllamaLLM
from privrag.llm.openai_chat import OpenAIChatLLM
from privrag.llm.lm_studio import LMStudioLLM
from privrag.llm.openrouter_chat import OpenRouterChatLLM


def get_llm(
    *,
    backend: LLMBackend | None = None,
    model: str | None = None,
) -> LLM:
    """Si `backend` es None, usa el del `.env`. `model` (si no está vacío) sustituye al modelo
    del backend activo."""
    s = get_settings()
    b = backend if backend is not None else s.llm_backend
    m = (model or "").strip() or None

    if b == LLMBackend.NONE:
        return NoopLLM()
    if b == LLMBackend.OLLAMA:
        name = m or s.ollama_model
        return OllamaLLM(s.ollama_base_url, name)
    if b == LLMBackend.OPENAI:
        if not s.openai_api_key:
            raise ValueError("OPENAI_API_KEY requerido para LLM_BACKEND=openai")
        name = m or s.openai_model
        return OpenAIChatLLM(s.openai_api_key, name)
    if b == LLMBackend.OPENROUTER:
        if not s.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY requerido para LLM_BACKEND=openrouter")
        extra: dict[str, str] = {}
        if s.openrouter_http_referer:
            extra["HTTP-Referer"] = s.openrouter_http_referer
        if s.openrouter_app_title:
            extra["X-Title"] = s.openrouter_app_title
        name = m or s.openrouter_model
        return OpenRouterChatLLM(
            s.openrouter_api_key,
            name,
            s.openrouter_base_url,
            extra or None,
        )
    if b == LLMBackend.LM_STUDIO:
        name = m or s.lm_studio_model
        if not name.strip():
            raise ValueError(
                "LM Studio: indica el modelo en la interfaz o define LM_STUDIO_MODEL en .env "
                "(debe coincidir con el modelo cargado en LM Studio)."
            )
        return LMStudioLLM(s.lm_studio_base_url, name.strip(), s.lm_studio_api_key)
    raise ValueError(f"Backend LLM no soportado: {b}")
