from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pymupdf

import step1
from answer_generation import AnswerCitation
from context_builder import ContextSource
from document_store import (
    citation_pdf_paths,
    delete_indexed_document,
    ingest_uploaded_pdf,
    list_indexed_documents,
    pdf_viewer_html,
)


class FakeEmbedder:
    model_name = "test-embedder"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]


class StreamlitAppHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "chunks.sqlite3"
        self.uploads_directory = self.root / "managed-uploads"
        self.connection = sqlite3.connect(self.database_path)
        step1.initialize_database(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        self.temporary_directory.cleanup()

    def make_pdf_bytes(self) -> bytes:
        path = self.root / "source.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "A local PDF passage for testing.")
            document.save(path)
        return path.read_bytes()

    def test_upload_is_persisted_and_duplicate_content_is_not_copied(self) -> None:
        pdf_bytes = self.make_pdf_bytes()
        result = ingest_uploaded_pdf(
            r"C:\fake\document.pdf",
            pdf_bytes,
            self.connection,
            uploads_directory=self.uploads_directory,
            embedder=FakeEmbedder(),
        )

        self.assertEqual(result.status, "processed")
        documents = list_indexed_documents(self.connection)
        self.assertEqual(len(documents), 1)
        self.assertTrue(documents[0].filename.endswith("_document.pdf"))
        self.assertTrue(documents[0].source_path.is_file())
        self.assertEqual(
            documents[0].source_path.parent.resolve(),
            self.uploads_directory.resolve(),
        )

        duplicate = ingest_uploaded_pdf(
            "renamed.pdf",
            pdf_bytes,
            self.connection,
            uploads_directory=self.uploads_directory,
            embedder=FakeEmbedder(),
        )
        self.assertEqual(duplicate.status, "duplicate")
        self.assertEqual(len(list_indexed_documents(self.connection)), 1)
        self.assertEqual(len(list(self.uploads_directory.iterdir())), 1)

    def test_delete_removes_index_cascade_and_only_managed_file_on_request(self) -> None:
        pdf_bytes = self.make_pdf_bytes()
        result = ingest_uploaded_pdf(
            "document.pdf",
            pdf_bytes,
            self.connection,
            uploads_directory=self.uploads_directory,
            embedder=FakeEmbedder(),
        )
        source_path = list_indexed_documents(self.connection)[0].source_path

        filename, file_deleted = delete_indexed_document(
            self.connection,
            result.sha256,
            uploads_directory=self.uploads_directory,
            delete_uploaded_file=False,
        )
        self.assertTrue(filename.endswith("_document.pdf"))
        self.assertFalse(file_deleted)
        self.assertTrue(source_path.is_file())
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()[0],
            0,
        )

    def test_delete_removes_app_managed_file_when_requested(self) -> None:
        pdf_bytes = self.make_pdf_bytes()
        result = ingest_uploaded_pdf(
            "document.pdf",
            pdf_bytes,
            self.connection,
            uploads_directory=self.uploads_directory,
            embedder=FakeEmbedder(),
        )
        source_path = list_indexed_documents(self.connection)[0].source_path

        _, file_deleted = delete_indexed_document(
            self.connection,
            result.sha256,
            uploads_directory=self.uploads_directory,
            delete_uploaded_file=True,
        )

        self.assertTrue(file_deleted)
        self.assertFalse(source_path.exists())

    def test_delete_never_removes_original_outside_managed_uploads(self) -> None:
        original_pdf = self.root / "original.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "A source PDF that remains on disk.")
            document.save(original_pdf)
        result = step1.ingest_pdf(
            original_pdf,
            self.connection,
            embedder=FakeEmbedder(),
        )

        _, file_deleted = delete_indexed_document(
            self.connection,
            result.sha256,
            uploads_directory=self.uploads_directory,
            delete_uploaded_file=True,
        )

        self.assertFalse(file_deleted)
        self.assertTrue(original_pdf.is_file())

    def test_citation_lookup_uses_chunk_metadata_and_pdf_page_viewer(self) -> None:
        pdf_bytes = self.make_pdf_bytes()
        result = ingest_uploaded_pdf(
            "document.pdf",
            pdf_bytes,
            self.connection,
            uploads_directory=self.uploads_directory,
            embedder=FakeEmbedder(),
        )
        chunk_id = self.connection.execute(
            "SELECT id FROM chunks WHERE document_sha256 = ?",
            (result.sha256,),
        ).fetchone()[0]
        citation = AnswerCitation(
            "C1",
            "P1",
            ContextSource(
                f"{result.sha256}_document.pdf",
                1,
                0,
                10,
                False,
                (chunk_id,),
            ),
        )

        paths = citation_pdf_paths(self.connection, [citation])
        self.assertEqual(
            paths["C1"],
            list_indexed_documents(self.connection)[0].source_path,
        )
        viewer = pdf_viewer_html(pdf_bytes, 3)
        self.assertIn("data:application/pdf;base64,", viewer)
        self.assertIn("#page=3", viewer)
        with self.assertRaisesRegex(ValueError, "page_number"):
            pdf_viewer_html(pdf_bytes, 0)


if __name__ == "__main__":
    unittest.main()
