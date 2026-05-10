"""Small, inspectable RAG diagnostics toolkit."""

from .chunker import Chunk, chunk_documents, chunk_text
from .embedder import HashingEmbedder, cosine_similarity, tokenize
from .load_documents import Document, load_documents
from .qa import RagAnswer, answer_question
from .retriever import InMemoryIndex, SearchResult

__all__ = [
    "Chunk",
    "Document",
    "HashingEmbedder",
    "InMemoryIndex",
    "RagAnswer",
    "SearchResult",
    "answer_question",
    "chunk_documents",
    "chunk_text",
    "cosine_similarity",
    "load_documents",
    "tokenize",
]

__version__ = "0.1.0"
