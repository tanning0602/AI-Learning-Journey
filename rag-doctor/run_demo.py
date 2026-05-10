from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.rag_doctor.chunker import chunk_documents
from src.rag_doctor.evaluate import evaluate_questions, load_question_cases, write_html_report
from src.rag_doctor.load_documents import load_documents
from src.rag_doctor.qa import answer_question
from src.rag_doctor.retriever import InMemoryIndex


def main() -> None:
    docs_path = ROOT / "examples" / "sample_docs"
    questions_path = ROOT / "examples" / "sample_questions.json"
    index_path = ROOT / ".rag-doctor" / "index.json"
    report_path = ROOT / "reports" / "report.html"

    print("Step 1: loading documents")
    documents = load_documents(docs_path)
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks.")

    print("\nStep 2: building local index")
    index = InMemoryIndex.from_chunks(chunks)
    index.save(index_path)
    print(f"Index saved to: {index_path}")

    print("\nStep 3: asking a sample question")
    question = "RAG Doctor 可以帮助定位什么问题？"
    answer = answer_question(index=index, question=question)
    print(f"Question: {answer.question}")
    print(f"Risk: {answer.risk_label}")
    print(f"Answer: {answer.answer}")
    print("Citations:")
    for citation in answer.citations:
        print(f"- {citation}")

    print("\nStep 4: generating evaluation report")
    question_cases = load_question_cases(questions_path)
    results = evaluate_questions(index=index, question_cases=question_cases)
    write_html_report(results, report_path)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
