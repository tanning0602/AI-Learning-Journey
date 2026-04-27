import unittest

from rag_doctor.chunker import chunk_documents
from rag_doctor.load_documents import Document
from rag_doctor.qa import answer_question
from rag_doctor.retriever import InMemoryIndex


class QaTest(unittest.TestCase):
    def test_answer_question_includes_citations(self):
        documents = [
            Document(
                source="notes.md",
                text="RAG Doctor checks retrieval quality, citation grounding, and hallucination risk.",
            )
        ]
        index = InMemoryIndex.from_chunks(chunk_documents(documents, chunk_size=120, overlap=20))

        answer = answer_question(index, "What does RAG Doctor check?", top_k=1)

        self.assertIn("retrieval", answer.answer.lower())
        self.assertTrue(answer.citations)
        self.assertIn(answer.risk_label, {"good", "weak_context", "possible_hallucination"})


if __name__ == "__main__":
    unittest.main()
