import unittest

from rag_doctor.chunker import chunk_documents
from rag_doctor.load_documents import Document
from rag_doctor.retriever import InMemoryIndex


class RetrieverTest(unittest.TestCase):
    def test_retriever_returns_relevant_chunk(self):
        documents = [
            Document(source="a.txt", text="RAG systems need citations and retrieval evaluation."),
            Document(source="b.txt", text="Pizza dough needs flour and water."),
        ]
        chunks = chunk_documents(documents, chunk_size=120, overlap=20)
        index = InMemoryIndex.from_chunks(chunks)

        results = index.search("How do I evaluate RAG retrieval?", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source, "a.txt")
        self.assertGreater(results[0].score, 0)


if __name__ == "__main__":
    unittest.main()
