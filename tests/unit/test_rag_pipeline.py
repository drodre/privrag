"""Tests para privrag.rag.pipeline (funciones puras)."""

import pytest

from privrag.rag.pipeline import _system_prompt_for_citations, format_context


class TestFormatContext:
    """Tests de format_context (función pura)."""

    def test_empty_hits(self):
        """Lista vacía de hits."""
        result = format_context([], max_chars=1000, include_citations=True)
        assert result == ""

    def test_single_hit_with_citations(self):
        """Un hit con citation."""
        hits = [_make_hit(text="Spell deals 1d6 damage.", source="/spells/fireball.md")]
        result = format_context(hits, max_chars=5000, include_citations=True)
        assert "[1] (/spells/fireball.md)" in result
        assert "Spell deals 1d6 damage." in result

    def test_single_hit_without_citations(self):
        """Un hit sin citation."""
        hits = [_make_hit(text="Spell deals 1d6 damage.", source="/spells/fireball.md")]
        result = format_context(hits, max_chars=5000, include_citations=False)
        assert "/spells/fireball.md" not in result
        assert "Spell deals 1d6 damage." in result

    def test_multiple_hits(self):
        """Múltiples hits."""
        hits = [
            _make_hit(text="Spell A", source="/a.md"),
            _make_hit(text="Spell B", source="/b.md"),
            _make_hit(text="Spell C", source="/c.md"),
        ]
        result = format_context(hits, max_chars=5000, include_citations=True)
        assert "[1] (/a.md)" in result
        assert "[2] (/b.md)" in result
        assert "[3] (/c.md)" in result
        assert "Spell A" in result
        assert "Spell B" in result

    def test_truncates_by_max_chars(self):
        """Trunca cuando excede max_chars."""
        # Crear hits largos
        long_text = "x" * 1000
        hits = [
            _make_hit(text=long_text, source="/long.md"),
            _make_hit(text=long_text, source="/long2.md"),
        ]
        result = format_context(hits, max_chars=1500, include_citations=True)
        # Debe contener al menos algo del primer hit
        assert "/long.md" in result
        # No debe contener el segundo (excede max_chars)
        assert result.count("/long") == 1

    def test_default_max_chars(self):
        """默认值 de max_chars."""
        hits = [_make_hit(text="short", source="/s.md")]
        result = format_context(hits)
        # Por defecto max_chars=12000
        assert "short" in result


class TestSystemPrompt:
    """Tests de _system_prompt_for_citations."""

    def test_with_citations(self):
        """Prompt con citas."""
        result = _system_prompt_for_citations(include_citations=True)
        assert "Reference sources" in result
        assert "[n]" in result

    def test_without_citations(self):
        """Prompt sin citas."""
        result = _system_prompt_for_citations(include_citations=False)
        assert "Do not cite sources" in result
        assert "cite" not in result.lower() or "do not cite" in result.lower()

    def test_both_include_rpg_context(self):
        """Ambos prompts mencionan RPG."""
        result_with = _system_prompt_for_citations(include_citations=True)
        result_without = _system_prompt_for_citations(include_citations=False)
        assert "tabletop RPG" in result_with
        assert "tabletop RPG" in result_without

    def test_both_mention_language(self):
        """Ambos prompts mencionan idioma."""
        result_with = _system_prompt_for_citations(include_citations=True)
        result_without = _system_prompt_for_citations(include_citations=False)
        assert "same language" in result_with
        assert "same language" in result_without


# === Helpers ===

from dataclasses import dataclass


@dataclass
class FakeHit:
    """Hit falso para tests."""

    text: str
    payload: dict

    @property
    def score(self) -> float:
        return 0.9


def _make_hit(text: str, source: str = "/test.md") -> FakeHit:
    """Crea un hit falso para testing."""
    return FakeHit(
        text=text,
        payload={"source_path": source},
    )
