from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import step1
from answer_generation import ABSTENTION_TEXT
from rag_pipeline import RagPipelineConfig, run_rag_pipeline


def _word_count(text: str) -> int:
    return len(text.split())


class RagPipelineTests(unittest.TestCase):
    def test_runs_retrieval_then_releases_reranker_before_generation(self) -> None:
        events: list[str] = []
        result = step1.SearchResult(
            "rag.pdf",
            4,
            0,
            45,
            False,
            "RAG retrieves passages to ground an answer.",
            0.9,
            42,
        )

        class FakeReranker:
            def close(self) -> None:
                events.append("reranker closed")

        class FakeGenerator:
            def __call__(self, prompt: str, /) -> str:
                events.append("answer generated")
                return json.dumps(
                    {
                        "status": "answered",
                        "statements": [
                            {
                                "text": result.text,
                                "passage_id": "P1",
                            }
                        ],
                    }
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "chunks.sqlite3"
            config = RagPipelineConfig(
                database=database,
                top_k=2,
                candidate_k=7,
                max_context_tokens=100,
            )
            reranker = FakeReranker()

            def search(*args, **kwargs):
                events.append("hybrid search")
                self.assertEqual(kwargs["top_k"], 2)
                self.assertEqual(kwargs["candidate_k"], 7)
                self.assertIs(kwargs["reranker"], reranker)
                return [result]

            with patch(
                "rag_pipeline.CrossEncoderReranker",
                return_value=reranker,
            ) as create_reranker, patch(
                "rag_pipeline.step1.search_chunks_hybrid",
                side_effect=search,
            ):
                answer = run_rag_pipeline(
                    "What does RAG do?",
                    config=config,
                    generator=FakeGenerator(),
                    token_counter=_word_count,
                )

        create_reranker.assert_called_once_with(threads=2)
        self.assertEqual(
            events,
            ["hybrid search", "reranker closed", "answer generated"],
        )
        self.assertEqual(
            answer.answer,
            "RAG retrieves passages to ground an answer. [C1]",
        )
        self.assertEqual(answer.citations[0].source.filename, "rag.pdf")
        self.assertEqual(answer.citations[0].source.page_number, 4)
        self.assertEqual(answer.citations[0].source.chunk_ids, (42,))

    def test_empty_retrieval_abstains_without_starting_generator(self) -> None:
        generator = Mock()
        with tempfile.TemporaryDirectory() as temporary_directory:
            answer = run_rag_pipeline(
                "Unanswerable question",
                config=RagPipelineConfig(
                    database=Path(temporary_directory) / "chunks.sqlite3",
                    rerank=False,
                ),
                generator=generator,
                token_counter=_word_count,
            )

        self.assertTrue(answer.abstained)
        self.assertEqual(answer.answer, ABSTENTION_TEXT)
        generator.assert_not_called()

    def test_requires_candidate_pool_at_least_as_large_as_final_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_k"):
            RagPipelineConfig(top_k=6, candidate_k=5)


if __name__ == "__main__":
    unittest.main()
