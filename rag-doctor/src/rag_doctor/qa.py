from __future__ import annotations

import re
from dataclasses import dataclass

from rag_doctor.embedder import tokenize
from rag_doctor.retriever import InMemoryIndex, SearchResult


SENTENCE_PATTERN = re.compile(r"[^。！？!?。\n]+[。！？!?。]?")


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: list[str]
    results: list[SearchResult]
    risk_label: str
    notes: list[str]


def split_sentences(text: str) -> list[str]:
    sentences = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(text)]
    return [sentence for sentence in sentences if sentence]


def _sentence_score(sentence: str, query_tokens: set[str]) -> int:
    sentence_tokens = set(tokenize(sentence))
    return len(sentence_tokens.intersection(query_tokens))


def _select_evidence_sentences(question: str, results: list[SearchResult], max_sentences: int = 3) -> list[str]:
    query_tokens = set(tokenize(question))
    candidates: list[tuple[int, float, str]] = []
    for result in results:
        for sentence in split_sentences(result.chunk.text):
            candidates.append((_sentence_score(sentence, query_tokens), result.score, sentence))

    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for token_score, _, sentence in candidates:
        if token_score <= 0:
            continue
        normalized = sentence.lower()
        if normalized in seen:
            continue
        selected.append(sentence)
        seen.add(normalized)
        if len(selected) >= max_sentences:
            break
    return selected


def _risk_label(results: list[SearchResult]) -> tuple[str, list[str]]:
    if not results:
        return "no_context", ["No context was retrieved."]

    top_score = results[0].score
    notes: list[str] = []
    if top_score < 0.12:
        notes.append("Top retrieval score is very low.")
        return "possible_hallucination", notes
    if top_score < 0.22:
        notes.append("Top retrieval score is weak; review the citation before trusting the answer.")
        return "weak_context", notes
    return "good", ["Answer is grounded in retrieved context."]


def answer_question(index: InMemoryIndex, question: str, top_k: int = 4) -> RagAnswer:
    results = index.search(question, top_k=top_k)
    risk_label, notes = _risk_label(results)

    if not results:
        return RagAnswer(
            question=question,
            answer="I could not find relevant context in the indexed documents.",
            citations=[],
            results=[],
            risk_label=risk_label,
            notes=notes,
        )

    evidence = _select_evidence_sentences(question, results)
    if evidence:
        answer = " ".join(evidence)
    else:
        answer = results[0].chunk.text[:500].strip()
        notes.append("No high-overlap sentence found; returning the top retrieved chunk preview.")

    citations = [
        f"{result.chunk.source}#{result.chunk.id} score={result.score:.3f}"
        for result in results
    ]
    return RagAnswer(
        question=question,
        answer=answer,
        citations=citations,
        results=results,
        risk_label=risk_label,
        notes=notes,
    )
