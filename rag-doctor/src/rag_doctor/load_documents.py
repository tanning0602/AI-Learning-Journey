from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Document:
    """A source document loaded from disk."""

    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def read_text_file(path: Path) -> str:
    """Read UTF-8 text, accepting files with a UTF-8 BOM."""

    return path.read_text(encoding="utf-8-sig")


def iter_supported_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return

    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path


def load_documents(path: str | Path) -> list[Document]:
    """Load .txt and .md documents from a file or directory."""

    root = Path(path)
    documents: list[Document] = []
    for file_path in iter_supported_files(root):
        text = read_text_file(file_path).strip()
        if not text:
            continue
        documents.append(
            Document(
                source=str(file_path),
                text=text,
                metadata={
                    "name": file_path.name,
                    "extension": file_path.suffix.lower(),
                },
            )
        )

    if not documents:
        raise ValueError(f"No supported documents found under: {root}")

    return documents
