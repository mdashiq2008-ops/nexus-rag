from __future__ import annotations

import json
import unittest

from extractive_generator import ExtractiveAnswerGenerator


class ExtractiveAnswerGeneratorTests(unittest.TestCase):
    def test_selects_exact_currency_line_for_statement_question(self) -> None:
        generator = ExtractiveAnswerGenerator("What is the opening balance?")
        prompt = (
            "Question:\nWhat is the opening balance?\n\nPassages:\n"
            "[P1]\nAccount holder Sam\n"
            "Address 7 Example Avenue  Opening balance   ₹48,230.55\n"
            "Every row follows the closing balance."
        )

        response = json.loads(generator(prompt))

        self.assertEqual(
            response["statements"],
            [
                {
                    "text": "Address 7 Example Avenue Opening balance ₹48,230.55",
                    "passage_id": "P1",
                }
            ],
        )

    def test_selects_a_verbatim_relevant_sentence(self) -> None:
        generator = ExtractiveAnswerGenerator(
            "What are the four functions in the framework?"
        )
        prompt = (
            "Question:\nWhat are the four functions in the framework?\n\n"
            "Passages:\n"
            "[P1]\nA paragraph about unrelated material.\n\n"
            "[P2]\nThe Core is composed of four functions: GOVERN, MAP, "
            "MEASURE, and MANAGE."
        )

        response = json.loads(generator(prompt))

        self.assertEqual(response["status"], "answered")
        self.assertEqual(
            response["statements"],
            [
                {
                    "text": (
                        "The Core is composed of four functions: GOVERN, "
                        "MAP, MEASURE, and MANAGE."
                    ),
                    "passage_id": "P2",
                }
            ],
        )

    def test_abstains_when_no_passage_matches_the_question(self) -> None:
        generator = ExtractiveAnswerGenerator("How does a volcano form?")

        response = json.loads(
            generator(
                "Question:\nHow does a volcano form?\n\nPassages:\n"
                "[P1]\nVariables store information in a Python program."
            )
        )

        self.assertEqual(
            response,
            {"status": "insufficient_evidence", "statements": []},
        )


if __name__ == "__main__":
    unittest.main()
