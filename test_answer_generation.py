import json
import unittest
from unittest.mock import Mock

import step1
from answer_generation import (
    ABSTENTION_TEXT,
    AnswerGenerationError,
    answer_question,
)


def _word_count(text: str) -> int:
    return len(text.split())


class GroundedAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "RAG retrieves passages to ground an answer."
        self.result = step1.SearchResult(
            "rag.pdf",
            4,
            20,
            20 + len(self.text),
            False,
            self.text,
            0.9,
            42,
        )

    def _answer(self, response: dict[str, object], results=None):
        generator = Mock(return_value=json.dumps(response))
        answer = answer_question(
            "What does RAG do?",
            [self.result] if results is None else results,
            generator=generator,
            max_context_tokens=100,
            token_counter=_word_count,
        )
        return answer, generator

    def test_renders_only_verbatim_evidence_with_metadata_citations(self) -> None:
        answer, generator = self._answer(
            {
                "status": "answered",
                "statements": [
                    {
                        "text": "RAG retrieves passages to ground an answer.",
                        "passage_id": "P1",
                    }
                ],
            }
        )

        self.assertEqual(
            answer.answer,
            "RAG retrieves passages to ground an answer. [C1]",
        )
        self.assertFalse(answer.abstained)
        self.assertEqual(answer.citations[0].source.filename, "rag.pdf")
        self.assertEqual(answer.citations[0].source.page_number, 4)
        self.assertEqual(answer.citations[0].source.chunk_ids, (42,))
        self.assertEqual(
            answer.to_dict()["citations"][0]["chunk_ids"],
            [42],
        )
        prompt = generator.call_args.args[0]
        self.assertIn("[P1]", prompt)
        self.assertNotIn("rag.pdf", prompt)

    def test_empty_retrieval_abstains_without_calling_generator(self) -> None:
        answer, generator = self._answer(
            {"status": "answered", "statements": []},
            results=[],
        )

        self.assertTrue(answer.abstained)
        self.assertEqual(answer.answer, ABSTENTION_TEXT)
        self.assertEqual(answer.citations, ())
        generator.assert_not_called()

    def test_model_can_explicitly_abstain(self) -> None:
        answer, _ = self._answer(
            {"status": "insufficient_evidence", "statements": []}
        )

        self.assertTrue(answer.abstained)
        self.assertEqual(answer.answer, ABSTENTION_TEXT)
        self.assertEqual(answer.citations, ())

    def test_rejects_paraphrases_and_unknown_passage_ids(self) -> None:
        with self.assertRaisesRegex(AnswerGenerationError, "exact excerpt"):
            self._answer(
                {
                    "status": "answered",
                    "statements": [
                        {
                            "text": "RAG uses documents to support an answer.",
                            "passage_id": "P1",
                        }
                    ],
                }
            )

        with self.assertRaisesRegex(AnswerGenerationError, "unknown passage"):
            self._answer(
                {
                    "status": "answered",
                    "statements": [
                        {"text": self.text, "passage_id": "P9"}
                    ],
                }
            )

    def test_invalid_generator_response_fails_explicitly(self) -> None:
        generator = Mock(return_value="not json")
        with self.assertRaisesRegex(AnswerGenerationError, "invalid JSON"):
            answer_question(
                "Question",
                [self.result],
                generator=generator,
                max_context_tokens=100,
                token_counter=_word_count,
            )


if __name__ == "__main__":
    unittest.main()
