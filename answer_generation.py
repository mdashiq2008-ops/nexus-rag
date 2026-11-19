from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence, TypedDict

import step1
from context_builder import (
    ContextBundle,
    ContextBundleDict,
    ContextPassage,
    ContextSource,
    ContextSourceDict,
    TokenCounter,
    build_context,
)


ABSTENTION_TEXT = (
    "I couldn't find enough supporting evidence in the indexed documents "
    "to answer that question."
)


class AnswerGenerator(Protocol):
    def __call__(self, prompt: str, /) -> str: ...


class AnswerCitationDict(ContextSourceDict):
    citation_id: str
    passage_id: str


class AnswerStatementDict(TypedDict):
    text: str
    citation_ids: list[str]


class GroundedAnswerDict(TypedDict):
    answer: str
    abstained: bool
    statements: list[AnswerStatementDict]
    citations: list[AnswerCitationDict]
    context: ContextBundleDict


class AnswerGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class AnswerCitation:
    citation_id: str
    passage_id: str
    source: ContextSource


@dataclass(frozen=True)
class AnswerStatement:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    abstained: bool
    statements: tuple[AnswerStatement, ...]
    citations: tuple[AnswerCitation, ...]
    context: ContextBundle

    def to_dict(self) -> GroundedAnswerDict:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "statements": [
                {
                    "text": statement.text,
                    "citation_ids": list(statement.citation_ids),
                }
                for statement in self.statements
            ],
            "citations": [
                {
                    "citation_id": citation.citation_id,
                    "passage_id": citation.passage_id,
                    "filename": citation.source.filename,
                    "page_number": citation.source.page_number,
                    "char_start": citation.source.char_start,
                    "char_end": citation.source.char_end,
                    "ocr_used": citation.source.ocr_used,
                    "chunk_ids": list(citation.source.chunk_ids),
                }
                for citation in self.citations
            ],
            "context": self.context.to_dict(),
        }


def _build_prompt(question: str, passages: Sequence[ContextPassage]) -> str:
    evidence = "\n\n".join(
        f"[P{position}]\n{passage.text}"
        for position, passage in enumerate(passages, start=1)
    )
    return (
        "Answer using only the supplied passages. This is a strict extractive "
        "task: each statement must be copied as an exact, contiguous excerpt "
        "from its cited passage. Do not paraphrase, add outside facts, or "
        "write citations, filenames, or page numbers in statement text. If "
        "the passages do not directly answer the question, return status "
        '"insufficient_evidence" with an empty statements array.\n'
        'Return only JSON with this shape: {"status":"answered" or '
        '"insufficient_evidence","statements":[{"text":"exact excerpt",'
        '"passage_id":"P1"}]}.\n\n'
        f"Question:\n{question}\n\n"
        f"Passages:\n{evidence}"
    )


def _normalized_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, character in enumerate(text):
        if character.isspace():
            if not characters or characters[-1] == " ":
                if characters:
                    ends[-1] = index + 1
                continue
            characters.append(" ")
            starts.append(index)
            ends.append(index + 1)
        else:
            characters.append(character)
            starts.append(index)
            ends.append(index + 1)

    if characters and characters[-1] == " ":
        characters.pop()
        starts.pop()
        ends.pop()
    return "".join(characters), starts, ends


def _find_verbatim_excerpt(passage_text: str, requested_text: str) -> str | None:
    normalized_passage, starts, ends = _normalized_with_offsets(passage_text)
    normalized_request = " ".join(requested_text.split())
    if not normalized_request:
        return None
    offset = normalized_passage.find(normalized_request)
    if offset < 0:
        return None
    final_character = offset + len(normalized_request) - 1
    return passage_text[starts[offset]:ends[final_character]]


def _abstain(context: ContextBundle) -> GroundedAnswer:
    return GroundedAnswer(
        answer=ABSTENTION_TEXT,
        abstained=True,
        statements=(),
        citations=(),
        context=context,
    )


def answer_question(
    question: str,
    ranked_results: Sequence[step1.SearchResult],
    *,
    generator: AnswerGenerator,
    max_context_tokens: int,
    token_counter: TokenCounter,
) -> GroundedAnswer:
    if not question.strip():
        raise ValueError("Question must not be empty")

    context = build_context(
        ranked_results,
        max_tokens=max_context_tokens,
        token_counter=token_counter,
    )
    if not context.passages:
        return _abstain(context)

    raw_response = generator(_build_prompt(question, context.passages))
    if not isinstance(raw_response, str):
        raise AnswerGenerationError("Generator must return a JSON string")
    try:
        response = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise AnswerGenerationError(
            "Generator returned invalid JSON"
        ) from exc
    if not isinstance(response, dict):
        raise AnswerGenerationError("Generator response must be a JSON object")

    status = response.get("status")
    statements = response.get("statements", [])
    if status == "insufficient_evidence":
        if statements != []:
            raise AnswerGenerationError(
                "Abstaining response must contain no statements"
            )
        return _abstain(context)
    if status != "answered":
        raise AnswerGenerationError(
            "Generator status must be answered or insufficient_evidence"
        )
    if not isinstance(statements, list) or not statements:
        raise AnswerGenerationError(
            "Answered response must contain at least one statement"
        )

    passages = {
        f"P{position}": passage
        for position, passage in enumerate(context.passages, start=1)
    }
    citations_by_source: dict[ContextSource, AnswerCitation] = {}
    grounded_statements: list[AnswerStatement] = []
    seen_statements: set[tuple[str, str]] = set()
    for item in statements:
        if not isinstance(item, dict):
            raise AnswerGenerationError("Each statement must be a JSON object")
        requested_text = item.get("text")
        passage_id = item.get("passage_id")
        if (
            not isinstance(requested_text, str)
            or not requested_text.strip()
            or not isinstance(passage_id, str)
        ):
            raise AnswerGenerationError(
                "Each statement needs nonempty text and a passage_id"
            )
        passage = passages.get(passage_id)
        if passage is None:
            raise AnswerGenerationError(
                f"Generator cited unknown passage {passage_id!r}"
            )
        excerpt = _find_verbatim_excerpt(passage.text, requested_text)
        if excerpt is None:
            raise AnswerGenerationError(
                f"Statement is not an exact excerpt from {passage_id}"
            )

        statement_key = (passage_id, excerpt)
        if statement_key in seen_statements:
            continue
        seen_statements.add(statement_key)

        citation_ids: list[str] = []
        for source in passage.sources:
            citation = citations_by_source.get(source)
            if citation is None:
                citation_id = f"C{len(citations_by_source) + 1}"
                citation = AnswerCitation(citation_id, passage_id, source)
                citations_by_source[source] = citation
            if citation.citation_id not in citation_ids:
                citation_ids.append(citation.citation_id)
        if not citation_ids:
            raise AnswerGenerationError(
                f"Passage {passage_id} has no source metadata"
            )
        grounded_statements.append(
            AnswerStatement(excerpt, tuple(citation_ids))
        )

    if not grounded_statements:
        raise AnswerGenerationError(
            "Answered response did not contain any unique statements"
        )

    rendered_answer = " ".join(
        f"{statement.text} "
        + "".join(f"[{citation_id}]" for citation_id in statement.citation_ids)
        for statement in grounded_statements
    )
    return GroundedAnswer(
        answer=rendered_answer,
        abstained=False,
        statements=tuple(grounded_statements),
        citations=tuple(citations_by_source.values()),
        context=context,
    )
