from contextlib import redirect_stdout
from io import StringIO
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf

import step1


class FakeEmbedder:
    model_name = "test-semantic-model"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[tuple[float, ...]]:
        vectors = []
        for text in texts:
            tokens = set(re.findall(r"[a-z]+", text.lower()))
            python_related = bool(
                tokens & {"python", "program", "programming", "code", "coding"}
            )
            fruit_related = bool(tokens & {"banana", "fruit", "smoothie"})
            vectors.append(
                (float(python_related), float(fruit_related), 1.0)
            )
        return vectors


class PdfIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "chunks.sqlite3"
        self.connection = sqlite3.connect(self.database)
        self.addCleanup(self.connection.close)
        step1.initialize_database(self.connection)
        self.embedder = FakeEmbedder()

    def make_pdf(self, path: Path, pages: list[str]) -> None:
        with pymupdf.open() as document:
            for text in pages:
                page = document.new_page()
                page.insert_text((72, 72), text)
            document.save(path)

    def test_ingests_multiple_pdfs_and_skips_duplicate_content(self) -> None:
        first_pdf = self.root / "first.pdf"
        second_pdf = self.root / "second.pdf"
        self.make_pdf(first_pdf, ["Heading\nFirst paragraph.", "Second page."])
        self.make_pdf(second_pdf, ["Another document."])

        first = step1.ingest_pdf(
            first_pdf,
            self.connection,
            chunk_size=20,
            chunk_overlap=3,
            embedder=self.embedder,
        )
        second = step1.ingest_pdf(
            second_pdf, self.connection, embedder=self.embedder
        )
        duplicate_pdf = self.root / "same-content.pdf"
        duplicate_pdf.write_bytes(first_pdf.read_bytes())
        duplicate = step1.ingest_pdf(
            duplicate_pdf, self.connection, embedder=self.embedder
        )

        self.assertEqual(first.status, "processed")
        self.assertEqual(first.pages, 2)
        self.assertEqual(second.status, "processed")
        self.assertEqual(duplicate.status, "duplicate")
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            2,
        )
        rows = self.connection.execute(
            """
            SELECT d.source_path, c.page_number, c.char_start, c.char_end, c.text
            FROM chunks AS c JOIN documents AS d ON d.sha256 = c.document_sha256
            WHERE d.sha256 = ?
            ORDER BY c.chunk_index
            """,
            (first.sha256,),
        ).fetchall()
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0][0], str(first_pdf.resolve()))
        self.assertEqual(rows[0][1], 1)
        self.assertTrue(all(row[3] > row[2] for row in rows))
        self.assertTrue(any("Heading" in row[4] for row in rows))
        self.assertTrue(any(row[1] == 2 for row in rows))

    def test_cli_accepts_multiple_pdfs(self) -> None:
        first_pdf = self.root / "first-cli.pdf"
        second_pdf = self.root / "second-cli.pdf"
        self.make_pdf(first_pdf, ["First document."])
        self.make_pdf(second_pdf, ["Second document."])

        with patch.object(step1, "LocalEmbedder", return_value=self.embedder):
            with redirect_stdout(StringIO()):
                exit_code = step1.main(
                    [
                        str(first_pdf),
                        str(second_pdf),
                        "--database",
                        str(self.database),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            2,
        )

    def test_scanned_page_requires_tesseract_and_is_not_marked_uploaded(self) -> None:
        scanned_pdf = self.root / "scan.pdf"
        pixmap = pymupdf.Pixmap(
            pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False
        )
        pixmap.clear_with(255)
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_image(page.rect, pixmap=pixmap)
            document.save(scanned_pdf)

        with patch("step1._find_tesseract", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Tesseract"):
                step1.ingest_pdf(scanned_pdf, self.connection)

        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            0,
        )

    def test_tables_are_stored_as_markdown(self) -> None:
        pdf_path = self.root / "table.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            for y in (100, 130, 160):
                page.draw_line((50, y), (250, y))
            for x in (50, 150, 250):
                page.draw_line((x, 100), (x, 160))
            page.insert_text((60, 120), "Header")
            page.insert_text((160, 120), "Value")
            page.insert_text((60, 145), "Row")
            page.insert_text((160, 145), "Data")
            document.save(pdf_path)

        result = step1.ingest_pdf(
            pdf_path, self.connection, embedder=self.embedder
        )
        stored_text = "\n".join(
            row[0]
            for row in self.connection.execute(
                "SELECT text FROM chunks WHERE document_sha256 = ?",
                (result.sha256,),
            )
        )

        self.assertIn("| Header |", stored_text)
        self.assertIn("| Row |", stored_text)
        self.assertEqual(stored_text.count("Header"), 1)

    def test_semantic_search_ranks_relevant_chunks_and_backfills_old_rows(self) -> None:
        python_pdf = self.root / "python.pdf"
        fruit_pdf = self.root / "fruit.pdf"
        self.make_pdf(python_pdf, ["Python programming and code"])
        self.make_pdf(fruit_pdf, ["Banana fruit smoothie"])
        step1.ingest_pdf(
            python_pdf, self.connection, embedder=self.embedder
        )
        step1.ingest_pdf(fruit_pdf, self.connection, embedder=self.embedder)

        self.connection.execute("DELETE FROM chunk_embeddings")
        self.connection.commit()
        results = step1.search_chunks(
            self.connection,
            "How do I write Python programs?",
            top_k=2,
            embedder=self.embedder,
        )

        self.assertEqual(results[0].filename, python_pdf.name)
        self.assertEqual(results[0].page_number, 1)
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM chunk_embeddings"
            ).fetchone()[0],
            2,
        )

    def test_bm25_search_ranks_term_matches_without_embeddings(self) -> None:
        python_pdf = self.root / "python.pdf"
        partial_pdf = self.root / "partial.pdf"
        fruit_pdf = self.root / "fruit.pdf"
        self.make_pdf(
            python_pdf,
            ["Python variable names store values in a program."],
        )
        self.make_pdf(partial_pdf, ["Python is a programming language."])
        self.make_pdf(fruit_pdf, ["Banana fruit smoothie recipe."])
        step1.ingest_pdf(python_pdf, self.connection, embedder=self.embedder)
        step1.ingest_pdf(partial_pdf, self.connection, embedder=self.embedder)
        step1.ingest_pdf(fruit_pdf, self.connection, embedder=self.embedder)
        self.connection.execute("DELETE FROM chunk_embeddings")
        self.connection.commit()

        results = step1.search_chunks_bm25(
            self.connection, "Python variable names", top_k=2
        )

        self.assertEqual(results[0].filename, python_pdf.name)
        self.assertEqual(results[0].page_number, 1)
        self.assertLess(results[0].score, results[1].score)
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM chunk_embeddings"
            ).fetchone()[0],
            0,
        )

    def test_hybrid_search_fuses_semantic_and_bm25_ranks(self) -> None:
        semantic_results = [
            step1.SearchResult("semantic.pdf", 1, 0, 10, False, "semantic", 0.9, 1),
            step1.SearchResult("shared.pdf", 1, 0, 10, False, "shared", 0.8, 2),
        ]
        bm25_results = [
            step1.SearchResult("shared.pdf", 1, 0, 10, False, "shared", -0.1, 2),
            step1.SearchResult("keyword.pdf", 1, 0, 10, False, "keyword", -0.2, 3),
        ]

        with patch(
            "step1.search_chunks", return_value=semantic_results
        ) as semantic_search, patch(
            "step1.search_chunks_bm25", return_value=bm25_results
        ) as bm25_search:
            results = step1.search_chunks_hybrid(
                self.connection, "query", top_k=3
            )

        semantic_search.assert_called_once_with(
            self.connection, "query", top_k=step1.HYBRID_CANDIDATE_K,
            embedder=None,
        )
        bm25_search.assert_called_once_with(
            self.connection, "query", top_k=step1.HYBRID_CANDIDATE_K
        )
        self.assertEqual(
            [result.filename for result in results],
            ["shared.pdf", "semantic.pdf", "keyword.pdf"],
        )
        self.assertAlmostEqual(results[0].score, 1 / 61 + 1 / 62)
        self.assertAlmostEqual(results[1].score, 1 / 61)
        self.assertAlmostEqual(results[2].score, 1 / 62)

    def test_hybrid_candidate_retrieval_reports_component_timings(self) -> None:
        semantic_results = [
            step1.SearchResult("semantic.pdf", 1, 0, 10, False, "semantic", 0.9, 1)
        ]
        bm25_results = [
            step1.SearchResult("keyword.pdf", 1, 0, 10, False, "keyword", -0.1, 2)
        ]
        timings: dict[str, float] = {}

        with patch(
            "step1.search_chunks", return_value=semantic_results
        ), patch("step1.search_chunks_bm25", return_value=bm25_results):
            results = step1.search_chunks_hybrid_candidates(
                self.connection,
                "query",
                candidate_k=2,
                timings=timings,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            set(timings),
            {
                "dense_search_seconds",
                "bm25_search_seconds",
                "rrf_fusion_seconds",
            },
        )
        self.assertTrue(all(value >= 0 for value in timings.values()))

    def test_hybrid_search_fuses_candidates_beyond_requested_top_k(self) -> None:
        semantic_results = [
            step1.SearchResult(
                f"semantic-{rank}.pdf", 1, 0, 10, False, "semantic", 0.9, rank
            )
            for rank in range(1, 4)
        ]
        bm25_results = [
            step1.SearchResult(
                f"keyword-{rank}.pdf", 1, 0, 10, False, "keyword", -0.1, rank + 40
            )
            for rank in range(1, 4)
        ]
        deep_match = step1.SearchResult(
            "deep-match.pdf", 1, 0, 10, False, "deep match", 0, 100
        )
        semantic_results.append(deep_match)
        bm25_results.append(deep_match)

        with patch(
            "step1.search_chunks", return_value=semantic_results
        ), patch("step1.search_chunks_bm25", return_value=bm25_results):
            results = step1.search_chunks_hybrid(
                self.connection, "query", top_k=3
            )

        self.assertEqual(
            [result.filename for result in results],
            ["deep-match.pdf", "semantic-1.pdf", "keyword-1.pdf"],
        )

    def test_hybrid_search_reranks_fused_candidates_and_preserves_metadata(self) -> None:
        semantic_results = [
            step1.SearchResult(
                "first.pdf", 2, 10, 25, True, "first passage", 0.9, 1
            ),
            step1.SearchResult(
                "second.pdf", 3, 30, 48, False, "second passage", 0.8, 2
            ),
        ]
        bm25_results = [
            step1.SearchResult(
                "first.pdf", 2, 10, 25, True, "first passage", -0.1, 1
            ),
            step1.SearchResult(
                "second.pdf", 3, 30, 48, False, "second passage", -0.2, 2
            ),
        ]
        reranker = Mock()
        reranker.score.return_value = [0.1, 0.9]

        with patch(
            "step1.search_chunks", return_value=semantic_results
        ) as semantic_search, patch(
            "step1.search_chunks_bm25", return_value=bm25_results
        ) as bm25_search:
            results = step1.search_chunks_hybrid(
                self.connection,
                "query",
                top_k=1,
                candidate_k=2,
                reranker=reranker,
            )

        semantic_search.assert_called_once_with(
            self.connection, "query", top_k=2, embedder=None
        )
        bm25_search.assert_called_once_with(
            self.connection, "query", top_k=2
        )
        reranker.score.assert_called_once_with(
            "query", ["first passage", "second passage"]
        )
        self.assertEqual(
            results,
            [
                step1.SearchResult(
                    "second.pdf", 3, 30, 48, False, "second passage", 0.9, 2
                )
            ],
        )

    def test_bm25_index_tracks_existing_new_and_updated_chunks(self) -> None:
        pdf_path = self.root / "indexed.pdf"
        self.make_pdf(pdf_path, ["Before update: apples and oranges"])
        result = step1.ingest_pdf(
            pdf_path, self.connection, embedder=self.embedder
        )
        self.assertEqual(
            len(step1.search_chunks_bm25(self.connection, "apples")), 1
        )

        self.connection.execute(
            "UPDATE chunks SET text = ? WHERE document_sha256 = ?",
            ("Updated text contains grapefruits.", result.sha256),
        )
        self.connection.commit()

        self.assertEqual(
            step1.search_chunks_bm25(self.connection, "apples"), []
        )
        self.assertEqual(
            len(step1.search_chunks_bm25(self.connection, "grapefruits")), 1
        )

    def test_cli_bm25_search_does_not_load_embedding_model(self) -> None:
        pdf_path = self.root / "keyword.pdf"
        self.make_pdf(pdf_path, ["Distinctive searchable keyword"])
        step1.ingest_pdf(pdf_path, self.connection, embedder=self.embedder)
        output = StringIO()

        with patch.object(
            step1,
            "LocalEmbedder",
            side_effect=AssertionError("BM25 must not load embeddings"),
        ):
            with redirect_stdout(output):
                exit_code = step1.main(
                    [
                        "--search",
                        "Distinctive",
                        "--search-mode",
                        "bm25",
                        "--database",
                        str(self.database),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("keyword.pdf, page 1", output.getvalue())
        self.assertIn("BM25", output.getvalue())

    def test_cli_hybrid_search_prints_rrf_score(self) -> None:
        pdf_path = self.root / "hybrid.pdf"
        self.make_pdf(pdf_path, ["Python programming and code"])
        step1.ingest_pdf(pdf_path, self.connection, embedder=self.embedder)
        output = StringIO()

        with patch.object(step1, "LocalEmbedder", return_value=self.embedder):
            with redirect_stdout(output):
                exit_code = step1.main(
                    [
                        "--search",
                        "Python coding",
                        "--search-mode",
                        "hybrid",
                        "--database",
                        str(self.database),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("hybrid.pdf, page 1", output.getvalue())
        self.assertIn("RRF", output.getvalue())

    def test_cli_hybrid_reranking_uses_mocked_cross_encoder(self) -> None:
        pdf_path = self.root / "hybrid-reranked.pdf"
        self.make_pdf(pdf_path, ["Python programming and code"])
        step1.ingest_pdf(pdf_path, self.connection, embedder=self.embedder)
        output = StringIO()
        reranker = Mock()
        reranker.score.return_value = [0.75]

        with patch.object(step1, "LocalEmbedder", return_value=self.embedder):
            with patch(
                "step1.CrossEncoderReranker", return_value=reranker
            ) as reranker_factory:
                with redirect_stdout(output):
                    exit_code = step1.main(
                        [
                            "--search",
                            "Python coding",
                            "--search-mode",
                            "hybrid",
                            "--rerank",
                            "--rerank-threads",
                            "2",
                            "--database",
                            str(self.database),
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertIn("hybrid-reranked.pdf, page 1", output.getvalue())
        self.assertIn("Cross-Encoder score 0.75", output.getvalue())
        self.assertEqual(
            output.getvalue().count("hybrid-reranked.pdf, page 1"), 1
        )
        reranker_factory.assert_called_once_with(threads=2)
        reranker.score.assert_called_once()

    def test_cli_search_prints_source_page_and_chunk(self) -> None:
        pdf_path = self.root / "searchable.pdf"
        self.make_pdf(pdf_path, ["Python programming and code"])
        step1.ingest_pdf(pdf_path, self.connection, embedder=self.embedder)
        output = StringIO()

        with patch.object(step1, "LocalEmbedder", return_value=self.embedder):
            with redirect_stdout(output):
                exit_code = step1.main(
                    [
                        "--search",
                        "Python coding",
                        "--database",
                        str(self.database),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("searchable.pdf, page 1", output.getvalue())
        self.assertIn("Python programming and code", output.getvalue())

    def test_interactive_mode_answers_multiple_questions_then_exits(self) -> None:
        python_pdf = self.root / "interactive.pdf"
        self.make_pdf(python_pdf, ["Python programming and code"])
        step1.ingest_pdf(python_pdf, self.connection, embedder=self.embedder)
        output = StringIO()

        with patch("builtins.input", side_effect=["Python programming?", "exit"]):
            with redirect_stdout(output):
                exit_code = step1.run_interactive(
                    self.connection, embedder=self.embedder
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("PDF semantic search is ready.", output.getvalue())
        self.assertIn("interactive.pdf, page 1", output.getvalue())
        self.assertIn("Goodbye.", output.getvalue())

    def test_interactive_hybrid_mode_searches_with_rrf(self) -> None:
        pdf_path = self.root / "interactive-hybrid.pdf"
        self.make_pdf(pdf_path, ["Python programming and code"])
        step1.ingest_pdf(pdf_path, self.connection, embedder=self.embedder)
        output = StringIO()

        with patch("step1.LocalEmbedder", return_value=self.embedder):
            with patch("builtins.input", side_effect=["Python coding", "exit"]):
                with redirect_stdout(output):
                    exit_code = step1.run_interactive(
                        self.connection,
                        search_mode="hybrid",
                    )

        self.assertEqual(exit_code, 0)
        self.assertIn("PDF hybrid search is ready.", output.getvalue())
        self.assertIn("RRF", output.getvalue())
        self.assertIn("interactive-hybrid.pdf, page 1", output.getvalue())

    def test_interactive_hybrid_reranker_is_loaded_once_and_reused(self) -> None:
        reranker = Mock()
        result = step1.SearchResult(
            "reranked.pdf", 1, 0, 10, False, "result", 0.9, 1
        )

        with patch("step1.CrossEncoderReranker", return_value=reranker) as factory:
            with patch(
                "step1.search_chunks_hybrid", return_value=[result]
            ) as hybrid_search:
                with patch(
                    "builtins.input",
                    side_effect=["first query", "second query", "exit"],
                ):
                    with redirect_stdout(StringIO()):
                        exit_code = step1.run_interactive(
                            self.connection,
                            search_mode="hybrid",
                            candidate_k=8,
                            rerank=True,
                            rerank_threads=4,
                            embedder=self.embedder,
                        )

        self.assertEqual(exit_code, 0)
        factory.assert_called_once_with(threads=4)
        self.assertEqual(hybrid_search.call_count, 2)
        self.assertIs(
            hybrid_search.call_args_list[0].kwargs["reranker"], reranker
        )
        self.assertIs(
            hybrid_search.call_args_list[1].kwargs["reranker"], reranker
        )
        self.assertEqual(
            hybrid_search.call_args_list[0].kwargs["candidate_k"], 8
        )

    def test_running_without_arguments_enters_interactive_mode(self) -> None:
        output = StringIO()
        with patch("builtins.input", side_effect=["exit"]):
            with redirect_stdout(output):
                exit_code = step1.main(
                    ["--database", str(self.database)]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("PDF semantic search is ready.", output.getvalue())

    def test_scanned_page_uses_ocr_text_when_available(self) -> None:
        page = Mock()
        page.find_tables.return_value.tables = []
        page.get_text.side_effect = lambda option, **kwargs: (
            "Recognized scanned text"
            if option == "text" and "textpage" in kwargs
            else ""
        )
        page.get_images.return_value = [("image",)]
        page.get_textpage_ocr.return_value = object()

        with patch("step1.shutil.which", return_value="tesseract"):
            text, ocr_used = step1.extract_page_text(page, "eng")

        self.assertEqual(text, "Recognized scanned text")
        self.assertTrue(ocr_used)
        page.get_textpage_ocr.assert_called_once_with(
            language="eng", dpi=300, full=True
        )

    def test_clean_text_removes_common_pdf_artifacts(self) -> None:
        self.assertEqual(
            step1.clean_text("inter-\nnational\u00ad  text \x00\ufffd"),
            "international text •",
        )

    def test_database_cleanup_repairs_artifacts_and_invalidates_embeddings(self) -> None:
        self.connection.execute(
            """
            INSERT INTO documents (sha256, filename, source_path, page_count)
            VALUES ('legacy-hash', 'legacy.pdf', 'legacy.pdf', 1)
            """
        )
        cursor = self.connection.execute(
            """
            INSERT INTO chunks (
                document_sha256, chunk_index, page_number,
                char_start, char_end, ocr_used, text
            ) VALUES ('legacy-hash', 0, 1, 0, 14, 0, 'Heading \ufffd text')
            """
        )
        self.connection.execute(
            """
            INSERT INTO chunk_embeddings
                (chunk_id, model_name, dimensions, vector)
            VALUES (?, 'test-model', 1, ?)
            """,
            (cursor.lastrowid, b"\x00\x00\x80?"),
        )
        self.connection.commit()

        step1.initialize_database(self.connection)

        self.assertEqual(
            self.connection.execute(
                "SELECT text FROM chunks WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()[0],
            "Heading • text",
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id = ?",
                (cursor.lastrowid,),
            ).fetchone()[0],
            0,
        )

    def test_chunker_tracks_source_offsets_and_overlaps(self) -> None:
        text = "One paragraph with enough words to span multiple chunks."
        chunks = list(step1.split_into_chunks(text, 20, 5))

        self.assertGreater(len(chunks), 1)
        for chunk, start, end in chunks:
            self.assertEqual(chunk, text[start:end])
        self.assertLessEqual(chunks[1][1], chunks[0][2])
        self.assertEqual(chunks[0][1], 0)
        self.assertEqual(chunks[-1][2], len(text.rstrip()))


if __name__ == "__main__":
    unittest.main()
