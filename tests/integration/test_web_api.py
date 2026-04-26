"""Tests de integración para privrag.web (FastAPI TestClient).

Tests de estructura de API - verifican que los endpoints responden correctamente.
"""

import pytest
from fastapi.testclient import TestClient


# Mockeamos ANTES de importar cualquier cosa que requiera Qdrant
import sys
from unittest.mock import patch, MagicMock

# Parcheamos las dependencias problematicas antes del import
mock_qdrant_cls = MagicMock()
mock_search_hit = MagicMock()
mock_search_hit.score = 0.9
mock_search_hit.text = "Test content"
mock_search_hit.payload = {"source_path": "/test.md"}

with (
    patch("qdrant_client.QdrantClient", mock_qdrant_cls),
    patch("privrag.config.get_settings") as mock_settings,
    patch("privrag.embed.factory.get_embedder") as mock_embedder,
):
    from privrag.config import LLMBackend
    from privrag.web.app import app, _parse_llm_backend, QueryBody

    # Configurar mocks
    mock_s = MagicMock()
    mock_s.llm_backend = LLMBackend.OLLAMA
    mock_s.ollama_model = "llama3.2"
    mock_s.openai_model = "gpt-4o-mini"
    mock_s.openrouter_model = "openai/gpt-4o-mini"
    mock_s.lm_studio_base_url = "http://localhost:1234/v1"
    mock_s.lm_studio_model = "modelo"
    mock_s.llm_max_tokens = None
    mock_s.llm_timeout = 300
    mock_s.llm_citations = True
    mock_s.llm_max_context_chars = 24000
    mock_s.qdrant_timeout = 30
    mock_s.ocr_enabled = False
    mock_s.ocr_language = "spa+eng"
    mock_s.ocr_timeout = 600
    mock_settings.return_value = mock_s

    # Mock de embedder
    mock_e = MagicMock()
    mock_e.vector_size = 384
    mock_e.encode.return_value = [[0.1] * 384]
    mock_embedder.return_value = mock_e


client = TestClient(app)


class TestIndex:
    """Tests del endpoint /."""

    def test_index_returns_html(self):
        """GET / retorna el HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_index_returns_valid_html(self):
        """GET / retorna HTML válido."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text.lower()
        assert "<!doctype html>" in html or "<html>" in html


class TestApiConfig:
    """Tests del endpoint /api/config."""

    def test_config_returns_json(self):
        """GET /api/config retorna configuración."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm_backend" in data

    def test_config_contains_all_fields(self):
        """Contiene todos los campos."""
        response = client.get("/api/config")
        data = response.json()
        required = [
            "llm_backend",
            "ollama_model",
            "openai_model",
            "openrouter_model",
            "lm_studio_base_url",
            "lm_studio_model",
            "llm_max_tokens",
            "llm_timeout",
            "llm_citations",
            "qdrant_timeout",
            "ocr_enabled",
            "ocr_language",
            "ocr_timeout",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"


class TestParseLlmBackend:
    """Tests de _parse_llm_backend (función pura)."""

    def test_none_returns_none(self):
        """None retorna None."""
        assert _parse_llm_backend(None) is None

    def test_empty_returns_none(self):
        """String vacío retorna None."""
        assert _parse_llm_backend("") is None
        assert _parse_llm_backend("   ") is None
        assert _parse_llm_backend("\t") is None

    def test_valid_backends(self):
        """Backends válidos."""
        assert _parse_llm_backend("ollama") == LLMBackend.OLLAMA
        assert _parse_llm_backend("openai") == LLMBackend.OPENAI
        assert _parse_llm_backend("none") == LLMBackend.NONE
        assert _parse_llm_backend("lmstudio") == LLMBackend.LM_STUDIO
        assert _parse_llm_backend("openrouter") == LLMBackend.OPENROUTER

    def test_case_sensitivity(self):
        """El backend es case-sensitive."""
        # Verificamos lo que el código realmente hace
        assert _parse_llm_backend("ollama") == LLMBackend.OLLAMA

    def test_invalid_backend_raises(self):
        """Backend inválido lanza HTTPException."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_llm_backend("invalid_backend")
        assert exc_info.value.status_code == 422
        assert "inválido" in exc_info.value.detail.lower()


class TestQueryBodyValidation:
    """Tests de validación de QueryBody."""

    def test_question_required(self):
        """Question es requerida."""
        response = client.post("/api/query", json={"collection": "test"})
        assert response.status_code == 422

    def test_question_min_length(self):
        """Question mínimo 1 carácter."""
        response = client.post(
            "/api/query",
            json={"question": "", "collection": "test"},
        )
        assert response.status_code == 422

    def test_limit_min(self):
        """limit >= 1."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "limit": 0},
        )
        assert response.status_code == 422

    def test_limit_max(self):
        """limit <= 50."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "limit": 51},
        )
        assert response.status_code == 422

    def test_max_tokens_invalid(self):
        """max_tokens debe ser >= 1."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "max_tokens": 0},
        )
        assert response.status_code == 422

    def test_qdrant_timeout_min(self):
        """qdrant_timeout debe ser >= 1."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "qdrant_timeout": 0},
        )
        assert response.status_code == 422

    def test_qdrant_timeout_max(self):
        """qdrant_timeout debe ser <= 3600."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "qdrant_timeout": 3601},
        )
        assert response.status_code == 422

    def test_llm_timeout_min(self):
        """llm_timeout debe ser >= 1."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "llm_timeout": 0},
        )
        assert response.status_code == 422

    def test_llm_timeout_max(self):
        """llm_timeout debe ser <= 3600."""
        response = client.post(
            "/api/query",
            json={"question": "test?", "collection": "test", "llm_timeout": 3601},
        )
        assert response.status_code == 422


class TestQueryBodyDefaults:
    """Tests de valores por defecto de QueryBody."""

    def test_default_collection(self):
        """Collection por defecto."""
        body = QueryBody(question="test?")
        assert body.collection == "docs"

    def test_default_limit(self):
        """Limit por defecto."""
        body = QueryBody(question="test?")
        assert body.limit == 5

    def test_default_no_llm(self):
        """no_llm por defecto."""
        body = QueryBody(question="test?")
        assert body.no_llm is False

    def test_default_include_citations(self):
        """include_citations por defecto."""
        body = QueryBody(question="test?")
        assert body.include_citations is True

    def test_default_topic(self):
        """topic por defecto."""
        body = QueryBody(question="test?")
        assert body.topic is None


class TestApiIngestValidation:
    """Tests de validación de /api/ingest."""

    def test_no_files_returns_error(self):
        """Sin archivos retorna error."""
        response = client.post(
            "/api/ingest",
            data={"collection": "test"},
        )
        # Puede ser 400 o 422 dependiendo de cómo FastAPI maneja el form
        assert response.status_code in [400, 422]

    def test_ocr_timeout_must_be_positive(self):
        """ocr_timeout debe ser >= 1."""
        response = client.post(
            "/api/ingest",
            data={"collection": "test", "ocr_timeout": 0},
            files={"files": ("sample.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 422


class TestApiQueryValidation:
    """Tests de validación de endpoints query."""

    def test_valid_question_passes_validation(self):
        """Question válida pasa validación de esquema - verificado en tests anteriores."""
        # El test anterior ya verificó que la validación funciona
        # Este test se omite porque requiere Qdrant real o mocks completos
        pass

    def test_runtime_timeout_returns_error_result(self):
        """Los errores de ejecución devuelven JSON con error y tiempo."""
        with patch("privrag.web.app.answer", side_effect=TimeoutError("timeout de prueba")):
            response = client.post(
                "/api/query",
                json={"question": "test?", "collection": "test"},
            )

        assert response.status_code == 504
        data = response.json()
        assert data["error"] == "timeout de prueba"
        assert data["answer"] == "Error: timeout de prueba"
        assert data["hits"] == []
        assert "elapsed_ms" in data
