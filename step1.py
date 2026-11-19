from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import sqlite3
import struct
import sys
import time
import unicodedata
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import pymupdf

from reranker import (
    DEFAULT_RERANK_THREADS,
    CrossEncoderReranker,
    Reranker,
)


OCR_TEXT_THRESHOLD = 40
DEFAULT_DATABASE = Path("pdf_chunks.sqlite3")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FTS_INDEX_VERSION = "1"
RRF_K = 60
HYBRID_CANDIDATE_K = 20


@dataclass(frozen=True)
class IngestResult:
    status: str
    sha256: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class SearchResult:
    filename: str
    page_number: int
    char_start: int
    char_end: int
    ocr_used: bool
    text: str
    score: float
    chunk_id: int


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Local semantic search requires fastembed. Install project "
                "dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        cache_dir = (
            Path(os.environ.get("LOCALAPPDATA", Path.home()))
            / "fastembed_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir)
        )

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [
            tuple(float(value) for value in vector)
            for vector in self._model.embed(list(texts))
        ]


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            sha256 TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            source_path TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_sha256 TEXT NOT NULL
                REFERENCES documents (sha256) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            ocr_used INTEGER NOT NULL CHECK (ocr_used IN (0, 1)),
            text TEXT NOT NULL,
            UNIQUE (document_sha256, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS chunks_document_page
            ON chunks (document_sha256, page_number);
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY
                REFERENCES chunks (id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS chunk_embeddings_model
            ON chunk_embeddings (model_name);
        CREATE TABLE IF NOT EXISTS app_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
        AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts (rowid, text)
            VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
        AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts (chunks_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_fts_update
        AFTER UPDATE OF text ON chunks BEGIN
            INSERT INTO chunks_fts (chunks_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
            INSERT INTO chunks_fts (rowid, text)
            VALUES (new.id, new.text);
        END;
        """
    )
    index_version = connection.execute(
        "SELECT value FROM app_metadata WHERE key = 'fts_index_version'"
    ).fetchone()
    if index_version is None or index_version[0] != FTS_INDEX_VERSION:
        connection.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")
        connection.execute(
            """
            INSERT INTO app_metadata (key, value)
            VALUES ('fts_index_version', ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value
            """,
            (FTS_INDEX_VERSION,),
        )
    connection.execute(
        """
        DELETE FROM chunk_embeddings
        WHERE chunk_id IN (
            SELECT id FROM chunks WHERE instr(text, char(65533)) > 0
        )
        """
    )
    connection.execute(
        """
        UPDATE chunks SET text = replace(text, char(65533), '•')
        WHERE instr(text, char(65533)) > 0
        """
    )
    connection.commit()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u00ad", "")
        .replace("\u00a0", " ")
        .replace("\ufffd", "•")
    )
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    cleaned_blocks: list[str] = []
    for block in text.split("\n\n"):
        lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in block.splitlines()
        ]
        cleaned_block = "\n".join(line for line in lines if line)
        if cleaned_block:
            cleaned_blocks.append(cleaned_block)
    return "\n\n".join(cleaned_blocks).strip()


def _find_tesseract() -> str | None:
    executable = shutil.which("tesseract")
    if executable:
        return executable

    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                user_path, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            user_path = ""

        if user_path:
            os.environ["PATH"] = os.pathsep.join(
                (user_path, os.environ.get("PATH", ""))
            )
            executable = shutil.which("tesseract")
            if executable:
                return executable

        for install_dir in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        ):
            candidate = install_dir / "Tesseract-OCR" / "tesseract.exe"
            if candidate.is_file():
                os.environ["PATH"] = os.pathsep.join(
                    (str(candidate.parent), os.environ.get("PATH", ""))
                )
                return shutil.which("tesseract") or str(candidate)

    return None


def _table_markdown(table: pymupdf.table.Table) -> str:
    rows = table.extract()
    if not rows:
        return ""

    formatted_rows = [
        [
            re.sub(r"\s+", " ", clean_text(cell or "")).replace("|", r"\|")
            for cell in row
        ]
        for row in rows
    ]
    column_count = max((len(row) for row in formatted_rows), default=0)
    if column_count == 0:
        return ""

    formatted_rows = [
        row + [""] * (column_count - len(row)) for row in formatted_rows
    ]
    header = formatted_rows[0]
    separator = ["---"] * column_count
    body = formatted_rows[1:]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        + ["| " + " | ".join(row) + " |" for row in body]
    )


def extract_page_text(
    page: pymupdf.Page, ocr_language: str
) -> tuple[str, bool]:
    tables = page.find_tables().tables
    table_regions = [tuple(table.bbox) for table in tables]
    content: list[tuple[float, float, str]] = []

    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, block_text = block[:5]
        if any(
            tx0 <= (x0 + x1) / 2 <= tx1
            and ty0 <= (y0 + y1) / 2 <= ty1
            for tx0, ty0, tx1, ty1 in table_regions
        ):
            continue
        cleaned_block = clean_text(block_text)
        if cleaned_block:
            content.append((y0, x0, cleaned_block))

    for table in tables:
        x0, y0, x1, y1 = table.bbox
        markdown = _table_markdown(table)
        if markdown:
            content.append((y0, x0, markdown))

    text = clean_text(
        "\n\n".join(
            value
            for _, _, value in sorted(
                content, key=lambda item: (item[0], item[1])
            )
        )
    )
    needs_ocr = (
        len(re.sub(r"\s+", "", text)) < OCR_TEXT_THRESHOLD
        and bool(page.get_images(full=True))
    )
    if not needs_ocr:
        return text, False

    if _find_tesseract() is None:
        raise RuntimeError(
            "This page appears to be scanned and requires OCR, but the "
            "Tesseract executable was not found on PATH. Install Tesseract OCR "
            "and its language data, then retry."
        )

    try:
        ocr_page = page.get_textpage_ocr(
            language=ocr_language, dpi=300, full=True
        )
    except (RuntimeError, OSError) as exc:
        raise RuntimeError(
            "PyMuPDF could not OCR this scanned page. Check that Tesseract OCR "
            "and the requested language data are installed."
        ) from exc

    ocr_text = clean_text(page.get_text("text", textpage=ocr_page, sort=True))
    return (ocr_text or text), True


def _chunk_boundaries(text: str, start: int, limit: int) -> int:
    if limit == len(text):
        return limit

    minimum = start + max(1, int((limit - start) * 0.7))
    for separator in ("\n\n", "\n", ". ", " "):
        boundary = text.rfind(separator, minimum, limit)
        if boundary >= minimum:
            return boundary + len(separator)
    return limit


def split_into_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> Iterator[tuple[str, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")

    start = 0
    while start < len(text):
        limit = min(start + chunk_size, len(text))
        end = _chunk_boundaries(text, start, limit)
        while end > start and text[end - 1].isspace():
            end -= 1

        content_start = start
        while content_start < end and text[content_start].isspace():
            content_start += 1
        if content_start < end:
            yield text[content_start:end], content_start, end
        if limit == len(text):
            break

        start = max(start + 1, end - chunk_overlap)
        while start < len(text) and text[start].isspace():
            start += 1


def _validate_embeddings(
    texts: Sequence[str], vectors: Sequence[tuple[float, ...]]
) -> int:
    if len(texts) != len(vectors):
        raise ValueError(
            f"Embedding model returned {len(vectors)} vectors for "
            f"{len(texts)} texts"
        )
    dimensions = len(vectors[0]) if vectors else 0
    if dimensions == 0:
        raise ValueError("Embedding model returned an empty vector")
    for vector in vectors:
        if len(vector) != dimensions or not all(
            math.isfinite(value) for value in vector
        ):
            raise ValueError("Embedding model returned an invalid vector")
    return dimensions


def _store_embedding_batch(
    connection: sqlite3.Connection,
    embedder: Embedder,
    chunk_ids: Sequence[int],
    texts: Sequence[str],
) -> None:
    vectors = embedder.embed(texts)
    dimensions = _validate_embeddings(texts, vectors)
    connection.executemany(
        """
        INSERT INTO chunk_embeddings (chunk_id, model_name, dimensions, vector)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (chunk_id) DO UPDATE SET
            model_name = excluded.model_name,
            dimensions = excluded.dimensions,
            vector = excluded.vector
        """,
        [
            (
                chunk_id,
                embedder.model_name,
                dimensions,
                struct.pack(f"<{dimensions}f", *vector),
            )
            for chunk_id, vector in zip(chunk_ids, vectors)
        ],
    )


def index_missing_embeddings(
    connection: sqlite3.Connection,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    rebuild: bool = False,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if rebuild:
        connection.execute("DELETE FROM chunk_embeddings")
        connection.commit()

    rows = connection.execute(
        """
        SELECT c.id, c.text
        FROM chunks AS c
        LEFT JOIN chunk_embeddings AS e
            ON e.chunk_id = c.id AND e.model_name = ?
        WHERE e.chunk_id IS NULL
        ORDER BY c.id
        """,
        (embedder.model_name,),
    ).fetchall()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        _store_embedding_batch(
            connection,
            embedder,
            [row[0] for row in batch],
            [row[1] for row in batch],
        )
        connection.commit()
    return len(rows)


def search_chunks(
    connection: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("Search query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    active_embedder = embedder or LocalEmbedder()
    index_missing_embeddings(connection, active_embedder)
    query_vectors = active_embedder.embed([query])
    _validate_embeddings([query], query_vectors)
    query_vector = query_vectors[0]
    query_norm = math.sqrt(math.fsum(value * value for value in query_vector))
    if query_norm == 0:
        raise ValueError("Embedding model returned a zero vector for the query")

    rows = connection.execute(
        """
        SELECT d.filename, c.page_number, c.char_start, c.char_end,
               c.ocr_used, c.text, e.dimensions, e.vector, c.id
        FROM chunk_embeddings AS e
        JOIN chunks AS c ON c.id = e.chunk_id
        JOIN documents AS d ON d.sha256 = c.document_sha256
        WHERE e.model_name = ?
        """,
        (active_embedder.model_name,),
    ).fetchall()

    results: list[SearchResult] = []
    for filename, page, start, end, ocr, text, dimensions, blob, chunk_id in rows:
        expected_bytes = dimensions * 4
        if dimensions != len(query_vector) or len(blob) != expected_bytes:
            raise ValueError(
                f"Stored embedding for chunk {chunk_id} is incompatible with "
                f"model {active_embedder.model_name}"
            )
        vector = struct.unpack(f"<{dimensions}f", blob)
        vector_norm = math.sqrt(math.fsum(value * value for value in vector))
        if vector_norm == 0:
            continue
        score = math.fsum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_vector, vector)
        ) / (query_norm * vector_norm)
        results.append(
            SearchResult(
                filename, page, start, end, bool(ocr), text, score, chunk_id
            )
        )

    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]


def search_chunks_bm25(
    connection: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("Search query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    if not terms:
        return []
    match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

    try:
        rows = connection.execute(
            """
            SELECT d.filename, c.page_number, c.char_start, c.char_end,
                   c.ocr_used, c.text, bm25(chunks_fts) AS rank, c.id
            FROM chunks_fts
            JOIN chunks AS c ON c.id = chunks_fts.rowid
            JOIN documents AS d ON d.sha256 = c.document_sha256
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC, c.id ASC
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite FTS5 search failed: {exc}") from exc

    return [
        SearchResult(
            filename, page, start, end, bool(ocr), text, float(rank), chunk_id
        )
        for filename, page, start, end, ocr, text, rank, chunk_id in rows
    ]


def search_chunks_hybrid_candidates(
    connection: sqlite3.Connection,
    query: str,
    *,
    candidate_k: int = HYBRID_CANDIDATE_K,
    embedder: Embedder | None = None,
    timings: dict[str, float] | None = None,
) -> list[SearchResult]:
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")

    start = time.perf_counter()
    semantic_results = search_chunks(
        connection, query, top_k=candidate_k, embedder=embedder
    )
    dense_seconds = time.perf_counter() - start

    start = time.perf_counter()
    bm25_results = search_chunks_bm25(
        connection, query, top_k=candidate_k
    )
    bm25_seconds = time.perf_counter() - start

    start = time.perf_counter()
    fused: dict[int, tuple[SearchResult, float]] = {}

    for ranked_results in (semantic_results, bm25_results):
        for rank, result in enumerate(ranked_results, start=1):
            previous = fused.get(result.chunk_id)
            score = 1 / (RRF_K + rank)
            if previous is None:
                fused[result.chunk_id] = (result, score)
            else:
                fused[result.chunk_id] = (previous[0], previous[1] + score)

    ranked = sorted(
        fused.values(),
        key=lambda item: (-item[1], item[0].chunk_id),
    )
    fused_candidates = [
        replace(result, score=score)
        for result, score in ranked[:candidate_k]
    ]
    rrf_seconds = time.perf_counter() - start
    if timings is not None:
        timings.update(
            {
                "dense_search_seconds": dense_seconds,
                "bm25_search_seconds": bm25_seconds,
                "rrf_fusion_seconds": rrf_seconds,
            }
        )
    return fused_candidates


def search_chunks_hybrid(
    connection: sqlite3.Connection,
    query: str,
    *,
    top_k: int = 5,
    candidate_k: int = HYBRID_CANDIDATE_K,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    fused_candidates = search_chunks_hybrid_candidates(
        connection,
        query,
        candidate_k=candidate_k,
        embedder=embedder,
    )
    if reranker is None:
        return fused_candidates[:top_k]
    if not fused_candidates:
        return []

    scores = reranker.score(
        query, [result.text for result in fused_candidates]
    )
    if len(scores) != len(fused_candidates):
        raise ValueError(
            "Reranker must return one score for each candidate"
        )
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Reranker returned a non-finite score")

    return [
        replace(result, score=score)
        for result, score in sorted(
            zip(fused_candidates, scores),
            key=lambda item: (-item[1], item[0].chunk_id),
        )[:top_k]
    ]


def run_interactive(
    connection: sqlite3.Connection,
    *,
    top_k: int = 5,
    candidate_k: int = HYBRID_CANDIDATE_K,
    search_mode: str = "semantic",
    embedder: Embedder | None = None,
    rerank: bool = False,
    rerank_threads: int = DEFAULT_RERANK_THREADS,
) -> int:
    if search_mode not in {"semantic", "bm25", "hybrid"}:
        raise ValueError("search_mode must be 'semantic', 'bm25', or 'hybrid'")
    if rerank and search_mode != "hybrid":
        raise ValueError("Cross-encoder reranking requires hybrid search")
    if rerank_threads is not None and rerank_threads <= 0:
        raise ValueError("rerank_threads must be greater than zero")
    if search_mode == "hybrid":
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )
    print(
        f"PDF {search_mode} search is ready. "
        "Ask a question, or type 'exit' to quit."
    )
    active_embedder = embedder
    active_reranker: Reranker | None = None
    while True:
        try:
            query = input("\nAsk a question: ").strip()
        except EOFError:
            print()
            return 0

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if not query:
            continue

        try:
            if search_mode == "bm25":
                results = search_chunks_bm25(
                    connection, query, top_k=top_k
                )
            elif search_mode == "hybrid":
                active_embedder = active_embedder or LocalEmbedder()
                if rerank and active_reranker is None:
                    active_reranker = CrossEncoderReranker(
                        threads=rerank_threads
                    )
                results = search_chunks_hybrid(
                    connection,
                    query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    embedder=active_embedder,
                    reranker=active_reranker,
                )
            else:
                active_embedder = active_embedder or LocalEmbedder()
                results = search_chunks(
                    connection,
                    query,
                    top_k=top_k,
                    embedder=active_embedder,
                )
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            print(f"ERROR: {exc}")
            continue

        if not results:
            print("No indexed chunks found. Ingest one or more PDFs first.")
            continue

        for rank, result in enumerate(results, start=1):
            score_label = (
                "Cross-Encoder score"
                if active_reranker is not None
                else {"semantic": "score", "bm25": "BM25", "hybrid": "RRF"}[
                    search_mode
                ]
            )
            print(
                f"\n[{rank}] {result.filename}, page {result.page_number}, "
                f"chars {result.char_start}-{result.char_end}, "
                f"{score_label} {result.score:.6g}"
                f"{', OCR' if result.ocr_used else ''}"
            )
            print(result.text)


def ingest_pdf(
    pdf_path: Path,
    connection: sqlite3.Connection,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    ocr_language: str = "eng",
    embedder: Embedder | None = None,
) -> IngestResult:
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")

    digest = file_sha256(pdf_path)
    existing = connection.execute(
        "SELECT page_count FROM documents WHERE sha256 = ?", (digest,)
    ).fetchone()
    if existing is not None:
        return IngestResult("duplicate", digest, existing[0], 0)

    with pymupdf.open(pdf_path) as document:
        pending_chunks: list[tuple[int, int, int, int, int, str]] = []
        chunk_index = 0
        for page_index, page in enumerate(document):
            page_text, ocr_used = extract_page_text(page, ocr_language)
            for chunk, char_start, char_end in split_into_chunks(
                page_text, chunk_size, chunk_overlap
            ):
                pending_chunks.append(
                    (
                        chunk_index,
                        page_index + 1,
                        char_start,
                        char_end,
                        int(ocr_used),
                        chunk,
                    )
                )
                chunk_index += 1
        page_count = len(document)

    vectors: list[tuple[float, ...]] = []
    active_embedder = embedder
    if pending_chunks:
        active_embedder = active_embedder or LocalEmbedder()
        texts = [chunk[5] for chunk in pending_chunks]
        vectors = active_embedder.embed(texts)
        _validate_embeddings(texts, vectors)

    connection.execute("BEGIN IMMEDIATE")
    try:
        existing = connection.execute(
            "SELECT page_count FROM documents WHERE sha256 = ?", (digest,)
        ).fetchone()
        if existing is not None:
            connection.rollback()
            return IngestResult("duplicate", digest, existing[0], 0)
        connection.execute(
            """
            INSERT INTO documents (sha256, filename, source_path, page_count)
            VALUES (?, ?, ?, ?)
            """,
            (digest, pdf_path.name, str(pdf_path.resolve()), page_count),
        )
        connection.executemany(
            """
            INSERT INTO chunks (
                document_sha256, chunk_index, page_number,
                char_start, char_end, ocr_used, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(digest, *chunk) for chunk in pending_chunks],
        )
        if pending_chunks and active_embedder is not None:
            chunk_ids = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT id FROM chunks
                    WHERE document_sha256 = ?
                    ORDER BY chunk_index
                    """,
                    (digest,),
                )
            ]
            dimensions = _validate_embeddings(
                [chunk[5] for chunk in pending_chunks], vectors
            )
            connection.executemany(
                """
                INSERT INTO chunk_embeddings
                    (chunk_id, model_name, dimensions, vector)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chunk_id,
                        active_embedder.model_name,
                        dimensions,
                        struct.pack(f"<{dimensions}f", *vector),
                    )
                    for chunk_id, vector in zip(chunk_ids, vectors)
                ],
            )
        connection.commit()
        return IngestResult("processed", digest, page_count, len(pending_chunks))
    except Exception:
        connection.rollback()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract, clean, chunk, and store text from PDF files."
    )
    parser.add_argument("pdfs", nargs="*", type=Path, help="PDF files to ingest")
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite database path (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code(s), e.g. eng or eng+fra (default: eng)",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search stored PDF chunks",
    )
    parser.add_argument(
        "--search-mode",
        choices=("semantic", "bm25", "hybrid"),
        default="semantic",
        help="Retrieval engine for --search or interactive search (default: semantic)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=HYBRID_CANDIDATE_K,
        help=(
            "Hybrid candidates retrieved and optionally reranked "
            f"(default: {HYBRID_CANDIDATE_K})"
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank hybrid candidates with a local cross-encoder",
    )
    parser.add_argument(
        "--rerank-threads",
        type=int,
        default=DEFAULT_RERANK_THREADS,
        help="ONNX CPU threads for cross-encoder inference",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild local embeddings for all chunks in the database",
    )
    args = parser.parse_args(argv)

    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("require --chunk-size > --chunk-overlap >= 0")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.candidate_k <= 0:
        parser.error("--candidate-k must be greater than zero")
    if args.search_mode == "hybrid" and args.candidate_k < args.top_k:
        parser.error("--candidate-k must be greater than or equal to --top-k")
    if args.rerank and args.search_mode != "hybrid":
        parser.error("--rerank requires --search-mode hybrid")
    if args.rerank_threads is not None and args.rerank_threads <= 0:
        parser.error("--rerank-threads must be greater than zero")
    if args.search and (args.pdfs or args.reindex):
        parser.error("--search cannot be combined with PDF paths or --reindex")
    if args.reindex and args.pdfs:
        parser.error("--reindex cannot be combined with PDF paths")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with closing(sqlite3.connect(args.database)) as connection:
        initialize_database(connection)
        if not args.pdfs and not args.search and not args.reindex:
            return run_interactive(
                connection,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                search_mode=args.search_mode,
                rerank=args.rerank,
                rerank_threads=args.rerank_threads,
            )

        if args.search or args.reindex:
            try:
                if args.search_mode == "bm25":
                    if args.reindex:
                        parser.error("--reindex applies only to semantic search")
                    results = search_chunks_bm25(
                        connection, args.search, top_k=args.top_k
                    )
                    if not results:
                        print(
                            "No matching chunks found. Ingest one or more PDFs first."
                        )
                        return 1
                    for rank, result in enumerate(results, start=1):
                        print(
                            f"\n[{rank}] {result.filename}, page "
                            f"{result.page_number}, chars {result.char_start}-"
                            f"{result.char_end}, BM25 {result.score:.6g}"
                            f"{', OCR' if result.ocr_used else ''}"
                        )
                        print(result.text)
                    return 0
                embedder = LocalEmbedder()
                if args.reindex:
                    count = index_missing_embeddings(
                        connection, embedder, rebuild=True
                    )
                    print(
                        f"REINDEXED {count} chunks with "
                        f"{embedder.model_name}"
                    )
                    return 0
                if args.search_mode == "hybrid":
                    reranker = (
                        CrossEncoderReranker(threads=args.rerank_threads)
                        if args.rerank
                        else None
                    )
                    results = search_chunks_hybrid(
                        connection,
                        args.search,
                        top_k=args.top_k,
                        candidate_k=args.candidate_k,
                        embedder=embedder,
                        reranker=reranker,
                    )
                else:
                    results = search_chunks(
                        connection,
                        args.search,
                        top_k=args.top_k,
                        embedder=embedder,
                    )
                if not results:
                    print(
                        "No indexed chunks found. Ingest one or more PDFs first."
                    )
                    return 1
                score_label = (
                    "Cross-Encoder score"
                    if args.search_mode == "hybrid" and args.rerank
                    else "RRF"
                    if args.search_mode == "hybrid"
                    else "score"
                )
                for rank, result in enumerate(results, start=1):
                    score = (
                        f"{result.score:.6g}"
                        if args.search_mode == "hybrid"
                        else f"{result.score:.3f}"
                    )
                    print(
                        f"\n[{rank}] {result.filename}, page "
                        f"{result.page_number}, chars {result.char_start}-"
                        f"{result.char_end}, {score_label} {score}"
                        f"{', OCR' if result.ocr_used else ''}"
                    )
                    print(result.text)
                return 0
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

        for pdf_path in args.pdfs:
            try:
                result = ingest_pdf(
                    pdf_path,
                    connection,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    ocr_language=args.ocr_language,
                )
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                print(f"ERROR {pdf_path}: {exc}", file=sys.stderr)
                failures += 1
                continue

            if result.status == "duplicate":
                print(f"SKIPPED {pdf_path}: already uploaded (SHA-256 match)")
            else:
                print(
                    f"STORED {pdf_path}: {result.pages} pages, "
                    f"{result.chunks} chunks"
                )
    return 1 if failures else 0


if __name__ == "__main__":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(errors="backslashreplace")
    raise SystemExit(main())