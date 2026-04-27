from __future__ import annotations

from dataclasses import dataclass, field

from rag_doctor.load_documents import Document


@dataclass(frozen=True)
class Chunk:
    """A searchable piece of a source document."""

    id: str
    source: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, str] = field(default_factory=dict)


def _paragraph_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    return [block for block in blocks if block]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks while keeping paragraphs when possible."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    blocks = _paragraph_blocks(normalized)
    chunks: list[tuple[str, int, int]] = []
    cursor = 0
    current_parts: list[str] = []
    current_start = 0
    current_len = 0

    def flush() -> None:
        nonlocal current_parts, current_start, current_len
        if not current_parts:
            return
        chunk = "\n\n".join(current_parts).strip()
        start = current_start
        end = start + len(chunk)
        chunks.append((chunk, start, end))

        if overlap > 0 and len(chunk) > overlap:
            tail = chunk[-overlap:]
            current_parts = [tail]
            current_start = max(0, end - overlap)
            current_len = len(tail)
        else:
            current_parts = []
            current_start = end
            current_len = 0

    for block in blocks:
        block_start = normalized.find(block, cursor)
        if block_start == -1:
            block_start = cursor
        cursor = block_start + len(block)

        if len(block) > chunk_size:
            flush()
            step = chunk_size - overlap
            for start in range(0, len(block), step):
                piece = block[start : start + chunk_size].strip()
                if not piece:
                    continue
                absolute_start = block_start + start
                chunks.append((piece, absolute_start, absolute_start + len(piece)))
                if start + chunk_size >= len(block):
                    break
            current_parts = []
            current_start = block_start + len(block)
            current_len = 0
            continue

        addition = len(block) if not current_parts else len(block) + 2
        if current_parts and current_len + addition > chunk_size:
            flush()

        if not current_parts:
            current_start = block_start
            current_len = 0
        current_parts.append(block)
        current_len += addition

    flush()
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_index, document in enumerate(documents):
        for chunk_index, (text, start, end) in enumerate(
            chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    id=f"doc{doc_index:03d}-chunk{chunk_index:03d}",
                    source=document.source,
                    text=text,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata,
                )
            )
    return chunks
