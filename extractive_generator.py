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
        self._question_terms = _terms(question)

    def __call__(self, prompt: str, /) -> str:
        marker = "\n\nPassages:\n"
        if marker not in prompt or not self._question_terms:
            return self._abstention()
        passage_text = prompt.split(marker, maxsplit=1)[1]
        passage_markers = list(
            re.finditer(r"(?m)^\[P(\d+)\]\s*$", passage_text)
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
