"""Tests para privrag.ingest.ocr."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from privrag.ingest.ocr import is_image_document, make_searchable_pdf


class TestOcrHelpers:
    def test_is_image_document(self):
        assert is_image_document(Path("scan.png")) is True
        assert is_image_document(Path("scan.PDF")) is False


class TestMakeSearchablePdf:
    def test_pdf_runs_ocrmypdf(self, tmp_path):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF-1.4")
        output_dir = tmp_path / "out"

        completed = MagicMock()
        completed.stdout = ""
        completed.stderr = ""
        with (
            patch("privrag.ingest.ocr.shutil.which", return_value="/usr/bin/tool"),
            patch("privrag.ingest.ocr.subprocess.run", return_value=completed) as run,
        ):
            result = make_searchable_pdf(
                source,
                output_dir,
                language="spa+eng",
                timeout=10,
            )

        assert result.status == "ok"
        assert result.output_path == output_dir / "scan.ocr.pdf"
        run.assert_called_once()
        command = run.call_args.args[0]
        assert command[:3] == ["ocrmypdf", "--language", "spa+eng"]
        assert str(source.resolve()) in command

    def test_image_converts_to_pdf_before_ocr(self, tmp_path):
        source = tmp_path / "scan.png"
        source.write_bytes(b"fake")
        output_dir = tmp_path / "out"

        completed = MagicMock()
        completed.stdout = ""
        completed.stderr = ""
        with (
            patch("privrag.ingest.ocr.shutil.which", return_value="/usr/bin/tool"),
            patch("privrag.ingest.ocr.subprocess.run", return_value=completed) as run,
        ):
            make_searchable_pdf(source, output_dir, language="eng", timeout=10)

        commands = [call.args[0][0] for call in run.call_args_list]
        assert commands == ["img2pdf", "ocrmypdf"]

    def test_missing_command_has_clear_error(self, tmp_path):
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"%PDF-1.4")

        with patch("privrag.ingest.ocr.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="No se encontró"):
                make_searchable_pdf(source, tmp_path / "out", language="spa", timeout=10)
