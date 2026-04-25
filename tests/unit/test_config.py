"""Tests para privrag.config."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from privrag.config import (
    EmbeddingBackend,
    LLMBackend,
    Settings,
    get_settings,
)


def _clean_env(extra=None):
    """Limpia el entorno para tests."""
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    if extra:
        safe_env.update(extra)
    return patch.dict(os.environ, safe_env, clear=True)


class TestSettingsDefaults:
    """Tests de valores por defecto de Settings."""

    def test_defaults_qdrant_url(self):
        with _clean_env():
            s = Settings()
            assert s.qdrant_url == "http://localhost:6333"

    def test_defaults_qdrant_timeout(self):
        with _clean_env():
            s = Settings()
            assert s.qdrant_timeout == 30

    def test_defaults_embedding_backend(self):
        with _clean_env():
            s = Settings()
            assert s.embedding_backend == EmbeddingBackend.LOCAL

    def test_defaults_local_embedding_model(self):
        with _clean_env():
            s = Settings()
            assert s.local_embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_defaults_llm_backend(self):
        with _clean_env():
            s = Settings()
            assert s.llm_backend in [e for e in LLMBackend]

    def test_defaults_llm_timeout(self):
        with _clean_env():
            s = Settings()
            assert s.llm_timeout == 300

    def test_defaults_chunk_size(self):
        with _clean_env():
            s = Settings()
            assert s.chunk_size == 512

    def test_defaults_chunk_overlap(self):
        with _clean_env():
            s = Settings()
            assert s.chunk_overlap == 64


class TestSettingsEnvVars:
    """Tests de lectura de variables de entorno."""

    def test_qdrant_url_from_env(self):
        with _clean_env({"QDRANT_URL": "http://custom:6333"}):
            s = Settings()
            assert s.qdrant_url == "http://custom:6333"

    def test_qdrant_api_key_from_env(self):
        with _clean_env({"QDRANT_API_KEY": "secret123"}):
            s = Settings()
            assert s.qdrant_api_key == "secret123"

    def test_qdrant_timeout_from_env(self):
        with _clean_env({"QDRANT_TIMEOUT": "60"}):
            s = Settings()
            assert s.qdrant_timeout == 60

    def test_embedding_backend_openai(self):
        with _clean_env({"EMBEDDING_BACKEND": "openai"}):
            s = Settings()
            assert s.embedding_backend == EmbeddingBackend.OPENAI

    def test_embedding_backend_local(self):
        with _clean_env({"EMBEDDING_BACKEND": "local"}):
            s = Settings()
            assert s.embedding_backend == EmbeddingBackend.LOCAL

    def test_llm_backend_openai(self):
        with _clean_env({"LLM_BACKEND": "openai", "OPENAI_API_KEY": "sk-test"}):
            s = Settings()
            assert s.llm_backend == LLMBackend.OPENAI

    def test_llm_backend_none(self):
        with _clean_env({"LLM_BACKEND": "none"}):
            s = Settings()
            assert s.llm_backend == LLMBackend.NONE

    def test_llm_timeout_from_env(self):
        with _clean_env({"LLM_TIMEOUT": "600"}):
            s = Settings()
            assert s.llm_timeout == 600

    def test_llm_backend_lm_studio(self):
        with _clean_env({"LLM_BACKEND": "lmstudio", "LM_STUDIO_MODEL": "modelo"}):
            s = Settings()
            assert s.llm_backend == LLMBackend.LM_STUDIO


class TestLmStudioUrlNormalization:
    """Tests de normalización de LM Studio URL."""

    def test_adds_v1_suffix(self):
        with _clean_env({"LM_STUDIO_BASE_URL": "http://localhost:41343"}):
            s = Settings()
            assert s.lm_studio_base_url == "http://localhost:41343/v1"

    def test_keeps_existing_v1(self):
        with _clean_env({"LM_STUDIO_BASE_URL": "http://localhost:41343/v1"}):
            s = Settings()
            assert s.lm_studio_base_url == "http://localhost:41343/v1"

    def test_removes_trailing_slash(self):
        with _clean_env({"LM_STUDIO_BASE_URL": "http://localhost:41343/v1/"}):
            s = Settings()
            assert s.lm_studio_base_url == "http://localhost:41343/v1"


class TestInvalidEnumValues:
    """Tests de valores inválidos para enums."""

    def test_invalid_embedding_backend(self):
        with _clean_env({"EMBEDDING_BACKEND": "invalid"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_llm_backend(self):
        with _clean_env({"LLM_BACKEND": "invalid"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_llm_max_tokens_negative(self):
        with _clean_env({"LLM_MAX_TOKENS": "-1"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_llm_max_tokens_zero(self):
        with _clean_env({"LLM_MAX_TOKENS": "0"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_qdrant_timeout_zero(self):
        with _clean_env({"QDRANT_TIMEOUT": "0"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_llm_timeout_zero(self):
        with _clean_env({"LLM_TIMEOUT": "0"}):
            with pytest.raises(ValidationError):
                Settings()


class TestGetSettingsCaching:
    """Tests del cache de get_settings."""

    def test_returns_cached_instance(self):
        get_settings.cache_clear()
        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2
        finally:
            get_settings.cache_clear()
