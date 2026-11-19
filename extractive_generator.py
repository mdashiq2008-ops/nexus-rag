from __future__ import annotations

import json
import re


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "says",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "who",
    "why",
    "with",
}


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOP_WORDS
    }


class ExtractiveAnswerGenerator:
    """Choose a query-relevant sentence verbatim without loading an LLM."""

    def __init__(self, question: str) -> None:
        self._question = question
        self._question_terms = _terms(question)

    def _direct_answer_lines(
        self,
        passage_text: str,
        passage_markers: list[re.Match[str]],
    ) -> tuple[str, str] | None:
        question = self._question.casefold()
        label = None
        if "opening balance" in question:
            label = "opening balance"
        elif "total debits" in question:
            label = "total debits"
        elif "total credits" in question:
            label = "total credits"
        elif "closing balance" in question:
            label = "closing balance"
        elif "salary" in question:
            label = "salary"
        if label is None:
            return None

        for index, passage_marker in enumerate(passage_markers):
            start = passage_marker.end()
            end = (
                passage_markers[index + 1].start()
                if index + 1 < len(passage_markers)
                else len(passage_text)
            )
            for line in passage_text[start:end].splitlines():
                normalized = " ".join(line.split()).casefold()
                if label not in normalized:
                    continue
                value = re.search(
                    r"(?:\u20b9\s*)?[0-9][0-9,]*\.[0-9]{2}",
                    line,
                )
                if value is not None:
                    return passage_marker.group(1), " ".join(line.split())
        return None

    def __call__(self, prompt: str, /) -> str:
        marker = "\n\nPassages:\n"
        if marker not in prompt or not self._question_terms:
            return self._abstention()
        passage_text = prompt.split(marker, maxsplit=1)[1]
        passage_markers = list(
            re.finditer(r"(?m)^\[P(\d+)\]\s*$", passage_text)
        )
        direct_answers = self._direct_answer_lines(passage_text, passage_markers)
        if direct_answers:
            passage_id, sentence = direct_answers
            return json.dumps(
                {
                    "status": "answered",
                    "statements": [
                        {"text": sentence, "passage_id": f"P{passage_id}"}
                    ],
                }
            )
        minimum_matches = 2 if len(self._question_terms) >= 4 else 1
        ranked_sentences: list[tuple[int, float, int, int, str, str]] = []

        for index, passage_marker in enumerate(passage_markers):
            start = passage_marker.end()
            end = (
                passage_markers[index + 1].start()
                if index + 1 < len(passage_markers)
                else len(passage_text)
            )
            text = " ".join(passage_text[start:end].split())
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_terms = _terms(sentence)
                matches = len(self._question_terms & sentence_terms)
                if matches < minimum_matches:
                    continue
                density = matches / max(len(sentence_terms), 1)
                ranked_sentences.append(
                    (
                        matches,
                        density,
                        -len(sentence_terms),
                        -index,
                        passage_marker.group(1),
                        sentence,
                    )
                )

        if not ranked_sentences:
            return self._abstention()
        _, _, _, _, passage_id, sentence = max(ranked_sentences)
        return json.dumps(
            {
                "status": "answered",
                "statements": [
                    {"text": sentence, "passage_id": f"P{passage_id}"}
                ],
            }
        )

    @staticmethod
    def _abstention() -> str:
        return json.dumps(
            {"status": "insufficient_evidence", "statements": []}
        )
