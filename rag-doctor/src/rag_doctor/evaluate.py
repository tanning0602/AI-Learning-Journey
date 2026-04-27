from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from rag_doctor.embedder import tokenize
from rag_doctor.qa import answer_question
from rag_doctor.retriever import InMemoryIndex


@dataclass(frozen=True)
class QuestionCase:
    question: str
    expected_terms: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    answer: str
    risk_label: str
    top_score: float
    term_coverage: float
    matched_terms: list[str]
    citations: list[str]


def load_question_cases(path: str | Path) -> list[QuestionCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload["questions"] if isinstance(payload, dict) else payload
    return [
        QuestionCase(
            question=item["question"],
            expected_terms=list(item.get("expected_terms", [])),
        )
        for item in cases
    ]


def evaluate_questions(
    index: InMemoryIndex,
    question_cases: list[QuestionCase],
    top_k: int = 4,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for question_case in question_cases:
        rag_answer = answer_question(index=index, question=question_case.question, top_k=top_k)
        answer_tokens = set(tokenize(rag_answer.answer))
        expected_terms = [term.lower() for term in question_case.expected_terms]
        matched_terms = [
            term
            for term in expected_terms
            if set(tokenize(term)).issubset(answer_tokens) or term in rag_answer.answer.lower()
        ]
        term_coverage = len(matched_terms) / len(expected_terms) if expected_terms else 0.0
        top_score = rag_answer.results[0].score if rag_answer.results else 0.0
        results.append(
            EvaluationResult(
                question=question_case.question,
                answer=rag_answer.answer,
                risk_label=rag_answer.risk_label,
                top_score=top_score,
                term_coverage=term_coverage,
                matched_terms=matched_terms,
                citations=rag_answer.citations,
            )
        )
    return results


def write_html_report(results: list[EvaluationResult], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in results:
        citation_html = "<br>".join(html.escape(citation) for citation in item.citations)
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.question)}</td>"
            f"<td>{html.escape(item.answer)}</td>"
            f"<td><strong>{html.escape(item.risk_label)}</strong></td>"
            f"<td>{item.top_score:.3f}</td>"
            f"<td>{item.term_coverage:.0%}</td>"
            f"<td>{html.escape(', '.join(item.matched_terms))}</td>"
            f"<td>{citation_html}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Doctor Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>RAG Doctor Report</h1>
  <table>
    <thead>
      <tr>
        <th>Question</th>
        <th>Answer</th>
        <th>Risk</th>
        <th>Top Score</th>
        <th>Term Coverage</th>
        <th>Matched Terms</th>
        <th>Citations</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
