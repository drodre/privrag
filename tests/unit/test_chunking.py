"""Tests para privrag.ingest.chunking."""

import pytest

from privrag.ingest.chunking import chunk_text


class TestChunkText:
    """Tests de chunk_text (función pura)."""

    def test_empty_string_returns_empty_list(self):
        """Texto vacío debe retornar lista vacío."""
        result = chunk_text("", chunk_size=100, overlap=10)
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Solo espacios/tabs debe retornar lista vacía."""
        result = chunk_text("   \n\t  ", chunk_size=100, overlap=10)
        assert result == []

    def test_single_chunk_when_text_fits(self):
        """Texto más pequeño que chunk_size produce un chunk."""
        text = "Hello world"
        result = chunk_text(text, chunk_size=512, overlap=64)
        assert result == ["Hello world"]

    def test_exact_chunk_size(self):
        """Texto igual a chunk_size produce un chunk exacto."""
        text = "a" * 256
        result = chunk_text(text, chunk_size=256, overlap=0)
        assert len(result) == 1
        assert result[0] == text

    def test_multiple_chunks_no_overlap(self):
        """Texto largo genera múltiples chunks sin overlap."""
        text = "a" * 600  # 256 + 256 + 88
        result = chunk_text(text, chunk_size=256, overlap=0)
        assert len(result) == 3
        assert result[0] == "a" * 256
        assert result[1] == "a" * 256
        assert result[2] == "a" * 88

    def test_multiple_chunks_with_overlap(self):
        """Chunks con overlap comparten contenido."""
        text = "abcdefghij"  # 10 chars
        result = chunk_text(text, chunk_size=6, overlap=2)
        # chunk1: abcdefghij (0-6) -> "abcdef"
        # chunk2: cdefghij (4-10) -> "efghij" (step = 6-2 = 4)
        # La implementación real produce "abcdef" y "efghij", no "cdefgh"
        assert len(result) >= 2
        assert "abcdef" in result
        assert "efghij" in result

    def test_overlap_larger_than_step_still_works(self):
        """Overlap = chunk_size - 1 produce step = 1 (shift mínimo)."""
        text = "abcde"
        result = chunk_text(text, chunk_size=3, overlap=2)
        # chunk1: abc
        # chunk2: bc (índice 2)
        # chunk3: cde (índice 3)
        assert len(result) == 3

    def test_strips_whitespace_from_chunks(self):
        """Chunks reducen espacios en blanco."""
        text = "  hello  \n\n  world  "
        result = chunk_text(text, chunk_size=50, overlap=0)
        assert len(result) == 1
        # El texto se strip() dentro de cada chunk
        # Verificamos que no tenga espacios al inicio/final
        assert result[0] == result[0].strip()
        assert "hello" in result[0]
        assert "world" in result[0]

    def test_invalid_chunk_size_zero(self):
        """chunk_size=0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=0, overlap=0)

    def test_invalid_chunk_size_negative(self):
        """chunk_size negativo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=-1, overlap=0)

    def test_invalid_overlap_negative(self):
        """overlap negativo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=100, overlap=-1)

    def test_invalid_overlap_equal_to_chunk_size(self):
        """overlap >= chunk_size debe lanzar ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=100, overlap=100)

    def test_realistic_rpg_text(self):
        """Test con texto realista de RPG."""
        text = """
        # stats

        Strength: 18 (+4)
        Dexterity: 14 (+2)
        Constitution: 16 (+3)

        # abilities

        Athletics: +8
        Stealth: +5
        """

        result = chunk_text(text.strip(), chunk_size=50, overlap=10)
        assert len(result) >= 1
        assert all(isinstance(c, str) for c in result)
