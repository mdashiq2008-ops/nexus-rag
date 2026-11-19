from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from answer_generation import AnswerCitation, AnswerStatement, GroundedAnswer
from context_builder import ContextBundle, ContextPassage, ContextSource
from api import create_app


class FakeEmbedder:
    model_name = "test-embedder"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "chunks.sqlite3"
        self.uploads_directory = self.root / "managed-uploads"
        self.client = TestClient(
            create_app(
                database_path=self.database_path,
                uploads_directory=self.uploads_directory,
            )
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary_directory.cleanup()

    def make_pdf_bytes(self) -> bytes:
        path = self.root / "source.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text(
                (72, 72),
                "The NIST AI RMF Core has four functions: GOVERN, MAP, "
                "MEASURE, and MANAGE.",
            )
            document.save(path)
        return path.read_bytes()

    def make_answer(self, *, abstained: bool = False) -> GroundedAnswer:
        source = ContextSource(
            filename="guide.pdf",
            page_number=2,
            char_start=10,
            char_end=40,
            ocr_used=False,
            chunk_ids=(7,),
        )
        passage = ContextPassage(
            rank=1,
            text="A verified answer from the source.",
            score=0.03,
            sources=(source,),
        )
        context = ContextBundle(
            passages=() if abstained else (passage,),
            text="" if abstained else passage.text,
            token_count=0 if abstained else 6,
            max_tokens=2500,
        )
        if abstained:
            return GroundedAnswer(
                answer="I couldn't find enough supporting evidence.",
                abstained=True,
                statements=(),
                citations=(),
                context=context,
            )
        citation = AnswerCitation("C1", "P1", source)
        return GroundedAnswer(
            answer="A verified answer from the source. [C1]",
            abstained=False,
            statements=(AnswerStatement(passage.text, ("C1",)),),
            citations=(citation,),
            context=context,
        )

    def test_health_and_document_list_are_available(self) -> None:
        health = self.client.get("/api/health")
        documents = self.client.get("/api/documents")
        status = self.client.get("/api/status")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["answer_mode"], "extractive")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["document_count"], 0)
        self.assertEqual(status.json()["chunk_count"], 0)
        self.assertIn("llm", status.json())
        self.assertEqual(documents.status_code, 200)
        self.assertEqual(documents.json(), [])

    def test_upload_duplicate_pdf_page_serving_and_delete(self) -> None:
        pdf_bytes = self.make_pdf_bytes()
        with patch("api.step1.LocalEmbedder", return_value=FakeEmbedder()):
            upload = self.client.post(
                "/api/documents",
                files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
            )

        self.assertEqual(upload.status_code, 201, upload.text)
        self.assertEqual(upload.json()["status"], "processed")
        document = upload.json()["document"]
        self.assertEqual(document["page_count"], 1)
        self.assertTrue(document["source_available"])

        duplicate = self.client.post(
            "/api/documents",
            files={"file": ("renamed.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(duplicate.status_code, 201, duplicate.text)
        self.assertEqual(duplicate.json()["status"], "duplicate")

        pdf_response = self.client.get(document["pdf_url"], params={"page": 1})
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.headers["content-type"], "application/pdf")
        self.assertEqual(pdf_response.content, pdf_bytes)
        self.assertEqual(
            self.client.get(document["pdf_url"], params={"page": 2}).status_code,
            404,
        )

        delete = self.client.delete(
            f"/api/documents/{document['sha256']}",
            params={"delete_uploaded_file": "true"},
        )
        self.assertEqual(delete.status_code, 200)
        self.assertTrue(delete.json()["uploaded_file_deleted"])
        self.assertEqual(self.client.get("/api/documents").json(), [])

    def test_chat_returns_traceable_extractive_answer_and_search_mode(self) -> None:
        expected = self.make_answer()
        with patch("api.run_rag_pipeline", return_value=expected) as run_pipeline:
            response = self.client.post(
                "/api/chat",
                json={
                    "question": "What does the source say?",
                    "search_mode": "hybrid_rerank",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertFalse(body["abstained"])
        self.assertEqual(body["answer_mode"], "extractive")
        self.assertEqual(body["search_mode"], "hybrid_rerank")
        self.assertEqual(body["citations"][0]["page_number"], 2)
        self.assertIsNone(body["citations"][0]["pdf_url"])
        self.assertEqual(
            body["source_passages"][0]["text"],
            expected.context.passages[0].text,
        )
        self.assertTrue(run_pipeline.call_args.kwargs["config"].rerank)
        self.assertEqual(
            run_pipeline.call_args.kwargs["generator"].__class__.__name__,
            "ExtractiveAnswerGenerator",
        )

    def test_chat_preserves_insufficient_evidence_status(self) -> None:
        with patch("api.run_rag_pipeline", return_value=self.make_answer(abstained=True)):
            response = self.client.post(
                "/api/chat",
                json={"question": "What is not in the documents?"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["abstained"])
        self.assertEqual(response.json()["citations"], [])
        self.assertEqual(response.json()["source_passages"], [])


if __name__ == "__main__":
    unittest.main()
