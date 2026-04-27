import unittest

from rag_doctor.chunker import chunk_text


class ChunkerTest(unittest.TestCase):
    def test_chunk_text_keeps_small_text_together(self):
        chunks = chunk_text("First paragraph.\n\nSecond paragraph.", chunk_size=100, overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertIn("First paragraph", chunks[0][0])
        self.assertIn("Second paragraph", chunks[0][0])

    def test_chunk_text_rejects_bad_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("hello", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
