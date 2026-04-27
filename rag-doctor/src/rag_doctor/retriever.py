from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_doctor.chunker import Chunk
from rag_doctor.embedder import HashingEmbedder, cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryIndex:
    """Small vector index for local learning and demos."""

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedder: HashingEmbedder | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder or HashingEmbedder()

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        embedder: HashingEmbedder | None = None,
    ) -> "InMemoryIndex":
        selected_embedder = embedder or HashingEmbedder()
        vectors = selected_embedder.embed_many([chunk.text for chunk in chunks])
        return cls(chunks=chunks, vectors=vectors, embedder=selected_embedder)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.embedder.embed(query)
        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [item for item in scored[:top_k] if item.score >= min_score]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": {
                "type": "hashing",
                "dimensions": self.embedder.dimensions,
            },
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectors": self.vectors,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        embedder = HashingEmbedder(dimensions=int(payload["embedder"]["dimensions"]))
        chunks = [Chunk(**chunk_payload) for chunk_payload in payload["chunks"]]
        vectors = payload["vectors"]
        return cls(chunks=chunks, vectors=vectors, embedder=embedder)
