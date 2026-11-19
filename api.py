from __future__ import annotations

import hashlib
import sqlite3
import shutil
from contextlib import closing
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

import step1
from document_store import (
    DATABASE_PATH,
    UPLOADS_DIRECTORY,
    delete_indexed_document,
    ingest_uploaded_pdf,
    list_indexed_documents,
    open_database,
)
from extractive_generator import ExtractiveAnswerGenerator
from qwen3_llama_cpp import Qwen3LlamaCppConfig
from rag_pipeline import RagPipelineConfig, run_rag_pipeline


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SearchMode = Literal["hybrid", "hybrid_rerank"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    search_mode: SearchMode = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=100)
    max_context_tokens: int = Field(default=2500, ge=256, le=16000)
    rerank_threads: int = Field(default=2, ge=1, le=16)

    @model_validator(mode="after")
    def validate_candidate_pool(self) -> "ChatRequest":
        if self.candidate_k < self.top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )
        return self


class DocumentResponse(BaseModel):
    sha256: str
    filename: str
    page_count: int
    chunk_count: int
    processed_at: str
    pdf_url: str | None
    source_available: bool
    file_size_bytes: int | None = None
    ocr_used: bool = False


class UploadResponse(BaseModel):
    status: Literal["processed", "duplicate"]
    document: DocumentResponse


class AnswerStatementResponse(BaseModel):
    text: str
    citation_ids: list[str]


class CitationResponse(BaseModel):
    citation_id: str
    passage_id: str
    filename: str
    page_number: int
    char_start: int
    char_end: int
    ocr_used: bool
    chunk_ids: list[int]
    pdf_url: str | None


class PassageSourceResponse(BaseModel):
    filename: str
    page_number: int
    char_start: int
    char_end: int
    ocr_used: bool
    chunk_ids: list[int]
    pdf_url: str | None


class PassageResponse(BaseModel):
    rank: int
    text: str
    score: float
    truncated: bool
    sources: list[PassageSourceResponse]


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    answer_mode: Literal["extractive"]
    search_mode: SearchMode
    statements: list[AnswerStatementResponse]
    citations: list[CitationResponse]
    source_passages: list[PassageResponse]


def _word_count(text: str, /) -> int:
    return len(text.split())


def _validate_digest(digest: str) -> None:
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest.lower()
    ):
        raise HTTPException(status_code=404, detail="Document not found")


def _document_url(digest: str, page_number: int | None = None) -> str:
    if page_number is None:
        return f"/api/documents/{digest}/pdf"
    return f"/api/documents/{digest}/pdf?page={page_number}#page={page_number}"


def _source_document_id(
    connection: sqlite3.Connection,
    filename: str,
    chunk_ids: tuple[int, ...],
) -> str | None:
    for chunk_id in chunk_ids:
        row = connection.execute(
            """
            SELECT d.sha256, d.filename, d.source_path
            FROM chunks AS c
            JOIN documents AS d ON d.sha256 = c.document_sha256
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if (
            row is not None
            and row[1] == filename
            and Path(row[2]).is_file()
        ):
            return str(row[0])
    return None


def _reranker_available() -> bool:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        return False
    return TextCrossEncoder is not None


def _llm_status() -> dict[str, object]:
    config = Qwen3LlamaCppConfig()
    configured = str(config.server_executable)
    resolved = shutil.which(configured)
    if resolved is None:
        candidate = Path(configured).expanduser()
        resolved = str(candidate.resolve()) if candidate.is_file() else None
    return {
        "configured": resolved is not None,
        "available": resolved is not None,
        "status": "available" if resolved is not None else "not_installed",
        "mode": "llm_optional",
        "model_repo": config.model_repo,
        "model_file": config.model_file,
    }


def _document_status(
    connection: sqlite3.Connection,
    document: object,
) -> tuple[int | None, bool]:
    source_path = getattr(document, "source_path")
    file_size = source_path.stat().st_size if source_path.is_file() else None
    ocr_row = connection.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM chunks
            WHERE document_sha256 = ? AND ocr_used = 1
        )
        """,
        (getattr(document, "sha256"),),
    ).fetchone()
    return file_size, bool(ocr_row[0]) if ocr_row is not None else False


def _chat_response(
    database_path: Path,
    question: str,
    search_mode: SearchMode,
    top_k: int = 5,
    candidate_k: int = 20,
    max_context_tokens: int = 2500,
    rerank_threads: int = 2,
) -> ChatResponse:
    config = RagPipelineConfig(
        database=database_path,
        rerank=search_mode == "hybrid_rerank",
        top_k=top_k,
        candidate_k=candidate_k,
        max_context_tokens=max_context_tokens,
        rerank_threads=rerank_threads,
    )
    answer = run_rag_pipeline(
        question,
        config=config,
        generator=ExtractiveAnswerGenerator(question),
        token_counter=_word_count,
    )

    with closing(open_database(database_path)) as connection:
        citations = []
        for citation in answer.citations:
            document_id = _source_document_id(
                connection,
                citation.source.filename,
                citation.source.chunk_ids,
            )
            citations.append(
                CitationResponse(
                    citation_id=citation.citation_id,
                    passage_id=citation.passage_id,
                    filename=citation.source.filename,
                    page_number=citation.source.page_number,
                    char_start=citation.source.char_start,
                    char_end=citation.source.char_end,
                    ocr_used=citation.source.ocr_used,
                    chunk_ids=list(citation.source.chunk_ids),
                    pdf_url=(
                        _document_url(
                            document_id,
                            citation.source.page_number,
                        )
                        if document_id is not None
                        else None
                    ),
                )
            )

        passages = []
        for passage in answer.context.passages:
            sources = []
            for source in passage.sources:
                document_id = _source_document_id(
                    connection,
                    source.filename,
                    source.chunk_ids,
                )
                sources.append(
                    PassageSourceResponse(
                        filename=source.filename,
                        page_number=source.page_number,
                        char_start=source.char_start,
                        char_end=source.char_end,
                        ocr_used=source.ocr_used,
                        chunk_ids=list(source.chunk_ids),
                        pdf_url=(
                            _document_url(document_id, source.page_number)
                            if document_id is not None
                            else None
                        ),
                    )
                )
            passages.append(
                PassageResponse(
                    rank=passage.rank,
                    text=passage.text,
                    score=passage.score,
                    truncated=passage.truncated,
                    sources=sources,
                )
            )

    return ChatResponse(
        answer=answer.answer,
        abstained=answer.abstained,
        answer_mode="extractive",
        search_mode=search_mode,
        statements=[
            AnswerStatementResponse(
                text=statement.text,
                citation_ids=list(statement.citation_ids),
            )
            for statement in answer.statements
        ],
        citations=citations,
        source_passages=passages,
    )


def create_app(
    database_path: Path = DATABASE_PATH,
    uploads_directory: Path = UPLOADS_DIRECTORY,
) -> FastAPI:
    application = FastAPI(
        title="NEXUS local RAG API",
        description="Local PDF indexing, hybrid retrieval, and extractive answers.",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/api/health")
    def health() -> dict[str, object]:
        database_status = "ok"
        try:
            with closing(open_database(database_path)) as connection:
                connection.execute("SELECT 1").fetchone()
        except (OSError, sqlite3.Error):
            database_status = "unavailable"
        llm = _llm_status()
        return {
            "status": "ok",
            "database_status": database_status,
            "answer_mode": "extractive",
            "retrieval_mode": "hybrid",
            "reranker_available": _reranker_available(),
            "llm": llm,
        }

    @application.get("/api/status")
    def status() -> dict[str, object]:
        with closing(open_database(database_path)) as connection:
            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            embedding_count = connection.execute(
                "SELECT COUNT(*) FROM chunk_embeddings"
            ).fetchone()[0]
        llm = _llm_status()
        return {
            "backend": "online",
            "database_status": "ok",
            "document_count": document_count,
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "reranker_available": _reranker_available(),
            "llm": llm,
        }

    @application.get(
        "/api/documents",
        response_model=list[DocumentResponse],
    )
    def get_documents() -> list[DocumentResponse]:
        with closing(open_database(database_path)) as connection:
            responses = []
            for document in list_indexed_documents(connection):
                file_size, ocr_used = _document_status(connection, document)
                responses.append(
                    DocumentResponse(
                        sha256=document.sha256,
                        filename=document.filename,
                        page_count=document.page_count,
                        chunk_count=document.chunk_count,
                        processed_at=document.processed_at,
                        pdf_url=(
                            _document_url(document.sha256)
                            if document.source_path.is_file()
                            else None
                        ),
                        source_available=document.source_path.is_file(),
                        file_size_bytes=file_size,
                        ocr_used=ocr_used,
                    )
                )
            return responses

    def index_upload(filename: str, content: bytes):
        digest = hashlib.sha256(content).hexdigest()
        with closing(open_database(database_path)) as connection:
            existing = connection.execute(
                "SELECT 1 FROM documents WHERE sha256 = ?",
                (digest,),
            ).fetchone()
            if existing is not None:
                result = ingest_uploaded_pdf(
                    filename,
                    content,
                    connection,
                    uploads_directory=uploads_directory,
                )
                return result
            embedder = step1.LocalEmbedder()
            return ingest_uploaded_pdf(
                filename,
                content,
                connection,
                uploads_directory=uploads_directory,
                embedder=embedder,
            )

    @application.post(
        "/api/documents",
        response_model=UploadResponse,
        status_code=201,
    )
    async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=415,
                detail="Only PDF uploads are supported",
            )
        if file.size is not None and file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 50 MiB upload limit",
            )
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 50 MiB upload limit",
            )
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
        try:
            result = await run_in_threadpool(index_upload, filename, content)
            with closing(open_database(database_path)) as connection:
                document = next(
                    item
                    for item in list_indexed_documents(connection)
                    if item.sha256 == result.sha256
                )
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UploadResponse(
            status=result.status,
            document=DocumentResponse(
                sha256=document.sha256,
                filename=document.filename,
                page_count=document.page_count,
                chunk_count=document.chunk_count,
                processed_at=document.processed_at,
                pdf_url=(
                    _document_url(document.sha256)
                    if document.source_path.is_file()
                    else None
                ),
                source_available=document.source_path.is_file(),
                file_size_bytes=(
                    document.source_path.stat().st_size
                    if document.source_path.is_file()
                    else None
                ),
                ocr_used=False,
            ),
        )

    @application.delete("/api/documents/{digest}")
    def remove_document(
        digest: str,
        delete_uploaded_file: bool = Query(default=False),
    ) -> dict[str, str | bool]:
        _validate_digest(digest)
        try:
            with closing(open_database(database_path)) as connection:
                filename, file_deleted = delete_indexed_document(
                    connection,
                    digest.lower(),
                    uploads_directory=uploads_directory,
                    delete_uploaded_file=delete_uploaded_file,
                )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "filename": filename,
            "deleted": True,
            "uploaded_file_deleted": file_deleted,
        }

    @application.get("/api/documents/{digest}/pdf")
    def get_document_pdf(
        digest: str,
        page: int = Query(default=1, ge=1),
    ) -> FileResponse:
        _validate_digest(digest)
        with closing(open_database(database_path)) as connection:
            row = connection.execute(
                """
                SELECT filename, source_path, page_count
                FROM documents WHERE sha256 = ?
                """,
                (digest.lower(),),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        filename, source_path, page_count = row
        if page > page_count:
            raise HTTPException(status_code=404, detail="PDF page not found")
        path = Path(source_path)
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="The indexed PDF file is unavailable",
            )
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=filename,
            content_disposition_type="inline",
        )

    @application.post(
        "/api/chat",
        response_model=ChatResponse,
    )
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            return await run_in_threadpool(
                _chat_response,
                database_path,
                request.question,
                request.search_mode,
                request.top_k,
                request.candidate_k,
                request.max_context_tokens,
                request.rerank_threads,
            )
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return application


app = create_app()
