from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_doctor.chunker import chunk_documents
from rag_doctor.evaluate import evaluate_questions, load_question_cases, write_html_report
from rag_doctor.load_documents import load_documents
from rag_doctor.qa import answer_question
from rag_doctor.retriever import InMemoryIndex


DEFAULT_INDEX_PATH = ".rag-doctor/index.json"


def build_index(args: argparse.Namespace) -> None:
    documents = load_documents(args.path)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    index = InMemoryIndex.from_chunks(chunks)
    index.save(args.index)
    print(json.dumps(
        {
            "documents": len(documents),
            "chunks": len(chunks),
            "index": str(args.index),
        },
        ensure_ascii=False,
        indent=2,
    ))


def ask(args: argparse.Namespace) -> None:
    index = InMemoryIndex.load(args.index)
    rag_answer = answer_question(index=index, question=args.question, top_k=args.top_k)
    print(f"Question: {rag_answer.question}")
    print(f"Risk: {rag_answer.risk_label}")
    print("")
    print(rag_answer.answer)
    print("")
    print("Citations:")
    for citation in rag_answer.citations:
        print(f"- {citation}")
    print("")
    print("Notes:")
    for note in rag_answer.notes:
        print(f"- {note}")


def evaluate(args: argparse.Namespace) -> None:
    index = InMemoryIndex.load(args.index)
    question_cases = load_question_cases(args.questions)
    results = evaluate_questions(index=index, question_cases=question_cases, top_k=args.top_k)
    write_html_report(results, args.output)
    summary = {
        "questions": len(results),
        "average_top_score": round(sum(item.top_score for item in results) / len(results), 3)
        if results
        else 0.0,
        "average_term_coverage": round(
            sum(item.term_coverage for item in results) / len(results),
            3,
        )
        if results
        else 0.0,
        "report": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-doctor",
        description="Diagnose retrieval quality and citation grounding in a small RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build a local index from .md and .txt documents.")
    index_parser.add_argument("path", help="File or directory containing documents.")
    index_parser.add_argument("--index", default=DEFAULT_INDEX_PATH, help="Path to write the JSON index.")
    index_parser.add_argument("--chunk-size", type=int, default=700)
    index_parser.add_argument("--overlap", type=int, default=120)
    index_parser.set_defaults(func=build_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against an existing index.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.set_defaults(func=ask)

    eval_parser = subparsers.add_parser("eval", help="Evaluate an index with a JSON question set.")
    eval_parser.add_argument("questions")
    eval_parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    eval_parser.add_argument("--top-k", type=int, default=4)
    eval_parser.add_argument("--output", default="reports/report.html")
    eval_parser.set_defaults(func=evaluate)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
