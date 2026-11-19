import unittest

import step1
from context_builder import ContextSource, build_context


def _word_count(text: str) -> int:
    return len(text.split())


class ContextBuilderTests(unittest.TestCase):
    def test_merges_overlapping_chunks_and_preserves_source_metadata(self) -> None:
        results = [
            step1.SearchResult(
                "guide.pdf", 3, 0, 11, False, "hello world", 0.9, 1
            ),
            step1.SearchResult(
                "guide.pdf", 3, 6, 17, True, "world again", 0.8, 2
            ),
        ]

        context = build_context(
            results,
            max_tokens=100,
            token_counter=_word_count,
        )

        self.assertEqual(len(context.passages), 1)
        passage = context.passages[0]
        self.assertEqual(passage.text, "hello world again")
        self.assertEqual(
            passage.sources,
            (
                ContextSource("guide.pdf", 3, 0, 17, True, (1, 2)),
            ),
        )
        self.assertIn("guide.pdf, page 3", context.text)

    def test_deduplicates_content_but_keeps_all_citations(self) -> None:
        results = [
            step1.SearchResult(
                "first.pdf", 1, 0, 10, False, "Same text", 0.9, 1
            ),
            step1.SearchResult(
                "second.pdf", 4, 30, 40, False, "Same  text", 0.8, 2
            ),
        ]

        context = build_context(
            results,
            max_tokens=100,
            token_counter=_word_count,
        )

        self.assertEqual(len(context.passages), 1)
        self.assertEqual(
            [source.filename for source in context.passages[0].sources],
            ["first.pdf", "second.pdf"],
        )
        self.assertEqual(
            [source.chunk_ids for source in context.passages[0].sources],
            [(1,), (2,)],
        )

    def test_truncates_to_fit_the_exact_serialized_context_budget(self) -> None:
        text = "one two three four five"
        result = step1.SearchResult(
            "notes.pdf", 2, 0, len(text), False, text, 0.7, 9
        )

        context = build_context(
            [result],
            max_tokens=10,
            token_counter=_word_count,
        )

        self.assertLessEqual(context.token_count, 10)
        self.assertTrue(context.passages[0].truncated)
        self.assertLess(len(context.passages[0].text), len(text))
        self.assertEqual(
            context.token_count,
            _word_count(context.text),
        )
        self.assertEqual(
            context.to_dict()["passages"][0]["sources"][0]["filename"],
            "notes.pdf",
        )
        self.assertEqual(
            context.to_dict()["passages"][0]["sources"][0]["chunk_ids"],
            [9],
        )

    def test_empty_results_and_invalid_limits(self) -> None:
        empty = build_context(
            [],
            max_tokens=10,
            token_counter=_word_count,
        )
        self.assertEqual(empty.text, "")
        self.assertEqual(empty.token_count, 0)

        with self.assertRaisesRegex(ValueError, "max_tokens"):
            build_context(
                [],
                max_tokens=0,
                token_counter=_word_count,
            )
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            build_context(
                [],
                max_tokens=10,
                token_counter=lambda text: -len(text) - 1,
            )
if __name__ == "__main__":
    unittest.main()
