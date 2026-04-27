"""Small, inspectable RAG diagnostics toolkit."""

from rag_doctor.chunker import Chunk, chunk_documents, chunk_text
from rag_doctor.embedder import HashingEmbedder, cosine_similarity, tokenize
from rag_doctor.load_documents import Document, load_documents
from rag_doctor.qa import RagAnswer, answer_question
from rag_doctor.retriever import InMemoryIndex, SearchResult

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
