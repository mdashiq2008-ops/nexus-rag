from __future__ import annotations

import base64
import hashlib
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import step1
from answer_generation import AnswerCitation


_PROJECT_DATABASE = Path(__file__).with_name("pdf_chunks.sqlite3")
DATABASE_PATH = Path(
    os.environ.get("NEXUS_DATABASE_PATH", str(_PROJECT_DATABASE))
).expanduser().resolve()
APP_DATA_DIRECTORY = (
    Path(os.environ.get("LOCALAPPDATA", Path.home())) / "nexus-rag"
)
UPLOADS_DIRECTORY = APP_DATA_DIRECTORY / "streamlit-uploads"


@dataclass(frozen=True)
class IndexedDocument:
    sha256: str
    filename: str
    source_path: Path
    page_count: int
    chunk_count: int
    processed_at: str


def pdf_viewer_html(pdf_bytes: bytes, page_number: int) -> str:
    if page_number <= 0:
        raise ValueError("page_number must be greater than zero")
    encoded_pdf = base64.b64encode(pdf_bytes).decode("ascii")
    return (
        '<iframe title="PDF source page" '
        f'src="data:application/pdf;base64,{encoded_pdf}#page={page_number}" '
        'width="100%" height="760" style="border:0"></iframe>'
    )


def open_database(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    initialized = False
    try:
        step1.initialize_database(connection)
        initialized = True
    finally:
        if not initialized:
            connection.close()
    return connection


def list_indexed_documents(
    connection: sqlite3.Connection,
) -> list[IndexedDocument]:
    rows = connection.execute(
        """
        SELECT d.sha256, d.filename, d.source_path, d.page_count,
               COUNT(c.id), d.processed_at
        FROM documents AS d
        LEFT JOIN chunks AS c ON c.document_sha256 = d.sha256
        GROUP BY d.sha256, d.filename, d.source_path, d.page_count,
                 d.processed_at
        ORDER BY d.filename COLLATE NOCASE
        """
    ).fetchall()
    return [
        IndexedDocument(
            sha256=digest,
            filename=filename,
            source_path=Path(source_path),
            page_count=page_count,
            chunk_count=chunk_count,
            processed_at=processed_at,
        )
        for digest, filename, source_path, page_count, chunk_count, processed_at in rows
    ]


def _safe_pdf_name(upload_name: str) -> str:
    filename = Path(upload_name.replace("\\", "/")).name
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not filename or not filename.lower().endswith(".pdf"):
        raise ValueError("Upload must have a .pdf filename")
    return filename


def ingest_uploaded_pdf(
    upload_name: str,
    pdf_bytes: bytes,
    connection: sqlite3.Connection,
    *,
    uploads_directory: Path = UPLOADS_DIRECTORY,
    embedder: step1.Embedder | None = None,
) -> step1.IngestResult:
    filename = _safe_pdf_name(upload_name)
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    existing = connection.execute(
        "SELECT page_count FROM documents WHERE sha256 = ?",
        (digest,),
    ).fetchone()
    corrupt_chunk_rows = (
        connection.execute(
        """
        SELECT text
        FROM chunks
        WHERE document_sha256 = ?
        """,
        (digest,),
        ).fetchall()
        if existing is not None
        else []
    )
    corrupt_chunks_exist = any(
        re.search(r"\d+\s+\.\s*\d*|\d+\s*,\s*\d*", text)
        for (text,) in corrupt_chunk_rows
    )
    if existing is not None and not corrupt_chunks_exist:
        return step1.IngestResult("duplicate", digest, existing[0], 0)
    existing = connection.execute(
        "SELECT page_count FROM documents WHERE sha256 = ?",
        (digest,),
    ).fetchone()

    uploads_directory.mkdir(parents=True, exist_ok=True)
    destination = uploads_directory / f"{digest}_{filename}"
    temporary_path: Path | None = None
    created_destination = False
    indexed = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=uploads_directory,
            prefix=f".{digest}.",
            suffix=".upload",
            delete=False,
        ) as temporary:
            temporary.write(pdf_bytes)
            temporary_path = Path(temporary.name)

        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise FileExistsError(
                    f"Upload destination contains different content: {destination}"
                )
            temporary_path.unlink()
            temporary_path = None
        else:
            temporary_path.replace(destination)
            temporary_path = None
            created_destination = True

        result = step1.ingest_pdf(
            destination,
            connection,
            embedder=embedder,
            filename=filename,
            reindex_existing=existing is not None,
        )
        if existing is not None and result.status == "processed":
            return step1.IngestResult(
                "duplicate",
                result.sha256,
                result.pages,
                result.chunks,
            )
        if result.status == "duplicate" and created_destination:
            destination.unlink()
            created_destination = False
        indexed = True
        return result
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if created_destination and not indexed and destination.exists():
            destination.unlink()


def delete_indexed_document(
    connection: sqlite3.Connection,
    digest: str,
    *,
    uploads_directory: Path = UPLOADS_DIRECTORY,
    delete_uploaded_file: bool = False,
) -> tuple[str, bool]:
    row = connection.execute(
        "SELECT filename, source_path FROM documents WHERE sha256 = ?",
        (digest,),
    ).fetchone()
    if row is None:
        raise ValueError("The selected document is no longer indexed")

    filename, source_path = row
    managed_path: Path | None = None
    if delete_uploaded_file:
        candidate = Path(source_path).resolve()
        uploads_root = uploads_directory.resolve()
        try:
            candidate.relative_to(uploads_root)
        except ValueError:
            pass
        else:
            managed_path = candidate

    connection.execute("DELETE FROM documents WHERE sha256 = ?", (digest,))
    connection.commit()

    file_deleted = False
    if managed_path is not None and managed_path.is_file():
        managed_path.unlink()
        file_deleted = True
    return filename, file_deleted


def citation_pdf_paths(
    connection: sqlite3.Connection,
    citations: Sequence[AnswerCitation],
) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {}
    for citation in citations:
        path: Path | None = None
        for chunk_id in citation.source.chunk_ids:
            row = connection.execute(
                """
                SELECT d.filename, d.source_path
                FROM chunks AS c
                JOIN documents AS d ON d.sha256 = c.document_sha256
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
            if row is not None and row[0] == citation.source.filename:
                path = Path(row[1])
                break
        paths[citation.citation_id] = path
    return paths
