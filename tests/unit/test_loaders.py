"""Tests para privrag.ingest.loaders."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from privrag.ingest.loaders import iter_documents, read_file_text


THIS_DIR = Path(__file__).parent


class TestReadFileText:
    """Tests de read_file_text."""

    def test_read_markdown(self):
        """Lee archivos .md correctamente."""
        path = THIS_DIR.parent / "fixtures" / "sample.md"
        result = read_file_text(path)
        assert "Sample RPG Document" in result
        assert "Stats" in result

    def test_read_txt(self):
        """Lee archivos .txt correctamente."""
        path = THIS_DIR.parent / "fixtures" / "sample.txt"
        result = read_file_text(path)
        assert "Plain text sample" in result

    def test_unsupported_extension(self):
        """Extensión no soportada lanza ValueError."""
        path = Path("/tmp/test.xyz")
        path.write_text("test")
        with pytest.raises(ValueError, match="Formato no soportado"):
            read_file_text(path)

    def test_read_rst(self):
        """Lee archivos .rst correctamente."""
        path = Path("/tmp/test.rst")
        path.write_text("Titulo\n=====\n\nContenido", encoding="utf-8")
        result = read_file_text(path)
        assert "Titulo" in result

    def test_read_markdown_extension(self):
        """Soporta .markdown."""
        path = Path("/tmp/test.markdown")
        path.write_text("# Titulo\n\nContenido", encoding="utf-8")
        result = read_file_text(path)
        assert "Titulo" in result


class TestIterDocuments:
    """Tests de iter_documents."""

    def test_single_file(self):
        """input=archivo retorna lista con ese archivo."""
        path = THIS_DIR.parent / "fixtures" / "sample.md"
        result = iter_documents(path)
        assert len(result) == 1
        assert result[0] == path

    def test_directory_returns_supported_files(self):
        """Directorio retorna solo archivos soportados."""
        fixtures = THIS_DIR.parent / "fixtures"
        result = iter_documents(fixtures)
        # sample.md y sample.txt
        assert len(result) == 2
        for p in result:
            assert p.suffix.lower() in {".md", ".txt"}

    def test_directory_excludes_unsupported(self):
        """Directorio excluye archivos no soportados."""
        fixtures = Path(tempfile.mkdtemp(prefix="privrag_test_"))
        try:
            (fixtures / "doc.md").write_text("# Doc")
            (fixtures / "doc.txt").write_text("Doc")
            (fixtures / "doc.xyz").write_text("Doc")
            (fixtures / "doc.pdf").write_text("Doc")

            result = iter_documents(fixtures)
            suffixes = [p.suffix.lower() for p in result]
            assert ".md" in suffixes
            assert ".txt" in suffixes
            assert ".xyz" not in suffixes
            # .pdf necesita pypdf y se permite, pero nuestro temp no lo tiene
            # assert ".pdf" in suffixes
        finally:
            shutil.rmtree(fixtures, ignore_errors=True)

    def test_empty_directory(self):
        """Directorio vacío retorna lista vacía."""
        fixtures = Path(tempfile.mkdtemp(prefix="privrag_test_empty_"))
        try:
            result = iter_documents(fixtures)
            assert result == []
        finally:
            shutil.rmtree(fixtures, ignore_errors=True)

    def test_sorts_by_name(self):
        """Archivos se retornan ordenados."""
        fixtures = Path(tempfile.mkdtemp(prefix="privrag_test_"))
        try:
            (fixtures / "z_file.md").write_text("# Z")
            (fixtures / "a_file.md").write_text("# A")
            (fixtures / "m_file.md").write_text("# M")

            result = iter_documents(fixtures)
            names = [p.name for p in result]
            assert names == sorted(names)
        finally:
            shutil.rmtree(fixtures, ignore_errors=True)

    def test_recursive_search(self):
        """Busca recursively en subdirectorios."""
        fixtures = Path(tempfile.mkdtemp(prefix="privrag_test_"))
        try:
            subdir = fixtures / "subdir"
            subdir.mkdir()
            (fixtures / "root.md").write_text("# Root")
            (subdir / "nested.md").write_text("# Nested")

            result = iter_documents(fixtures)
            names = {p.name for p in result}
            assert "root.md" in names
            assert "nested.md" in names
        finally:
            shutil.rmtree(fixtures, ignore_errors=True)

    def test_case_insensitive_extension(self):
        """Extensiones son case-insensitive."""
        fixtures = Path(tempfile.mkdtemp(prefix="privrag_test_"))
        try:
            (fixtures / "doc.MD").write_text("# Doc")
            (fixtures / "doc.Txt").write_text("Doc")

            result = iter_documents(fixtures)
            assert len(result) == 2
        finally:
            shutil.rmtree(fixtures, ignore_errors=True)
