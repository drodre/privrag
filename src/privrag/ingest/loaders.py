from pathlib import Path

from pypdf import PdfReader


def read_file_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".md", ".markdown", ".txt", ".rst"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suf == ".pdf":
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
        return "\n\n".join(parts)
    raise ValueError(f"Formato no soportado: {path}")


def iter_documents(root: Path) -> list[Path]:
    root = root.resolve()
    if root.is_file():
        return [root]
    allowed = {".md", ".markdown", ".txt", ".rst", ".pdf"}
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed:
            out.append(p)
    return out
