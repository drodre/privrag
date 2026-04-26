from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class OcrResult:
    source_path: Path
    output_path: Path
    status: str
    message: str = ""


def is_image_document(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_ocr_document(path: Path) -> bool:
    return path.suffix.lower() == ".pdf" or is_image_document(path)


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            f"No se encontró el comando {command!r}. Instala las dependencias OCR y revisa README.md."
        )


def _run(command: list[str], *, timeout: int, operation: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Timeout durante {operation} tras {timeout}s.") from e
    except subprocess.CalledProcessError as e:
        detail = "\n".join(x for x in [e.stdout.strip(), e.stderr.strip()] if x)
        raise RuntimeError(f"Falló {operation}: {detail or e}") from e


def _unique_output_path(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{source.stem}.ocr.pdf"
    candidate = output_dir / base
    if not candidate.exists():
        return candidate
    return output_dir / f"{source.stem}.{uuid.uuid4().hex[:8]}.ocr.pdf"


def make_searchable_pdf(
    source: Path,
    output_dir: Path,
    *,
    language: str,
    timeout: int,
) -> OcrResult:
    """Convierte un PDF o imagen en un PDF buscable con OCR local."""
    source = source.resolve()
    output = _unique_output_path(source, output_dir.resolve())

    _require_command("ocrmypdf")

    with tempfile.TemporaryDirectory(prefix="privrag_ocr_") as tmp_raw:
        tmp = Path(tmp_raw)
        input_pdf = source
        if is_image_document(source):
            _require_command("img2pdf")
            input_pdf = tmp / f"{source.stem}.pdf"
            _run(
                ["img2pdf", str(source), "-o", str(input_pdf)],
                timeout=timeout,
                operation=f"convertir imagen a PDF ({source.name})",
            )

        _run(
            [
                "ocrmypdf",
                "--language",
                language,
                "--skip-text",
                str(input_pdf),
                str(output),
            ],
            timeout=timeout,
            operation=f"aplicar OCR a {source.name}",
        )

    return OcrResult(
        source_path=source,
        output_path=output,
        status="ok",
        message="PDF buscable generado",
    )
