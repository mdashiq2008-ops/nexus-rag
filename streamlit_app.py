from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import step1
from answer_generation import GroundedAnswer
from document_store import (
    DATABASE_PATH,
    UPLOADS_DIRECTORY,
    citation_pdf_paths,
    delete_indexed_document,
    ingest_uploaded_pdf,
    list_indexed_documents,
    open_database,
    pdf_viewer_html,
)
from rag_pipeline import RagPipelineConfig, run_rag_pipeline


def _render_document_manager() -> None:
    st.sidebar.header("PDF library")
    uploads = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploads",
    )
    if st.sidebar.button("Add PDFs to library", disabled=not uploads):
        embedder: step1.Embedder | None = None
        for upload in uploads or []:
            try:
                with closing(open_database()) as connection:
                    digest = hashlib.sha256(upload.getvalue()).hexdigest()
                    existing = connection.execute(
                        "SELECT 1 FROM documents WHERE sha256 = ?",
                        (digest,),
                    ).fetchone()
                    if existing is not None:
                        st.sidebar.info(f"Already indexed: {upload.name}")
                        continue

                with st.spinner(f"Indexing {upload.name}..."):
                    if embedder is None:
                        embedder = step1.LocalEmbedder()
                    with closing(open_database()) as connection:
                        result = ingest_uploaded_pdf(
                            upload.name,
                            upload.getvalue(),
                            connection,
                            embedder=embedder,
                        )
                if result.status == "duplicate":
                    st.sidebar.info(f"Already indexed: {upload.name}")
                else:
                    st.sidebar.success(
                        f"Added {upload.name}: {result.pages} pages, "
                        f"{result.chunks} chunks"
                    )
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
                st.sidebar.error(f"Could not index {upload.name}: {exc}")

    with closing(open_database()) as connection:
        documents = list_indexed_documents(connection)

    st.sidebar.subheader(f"Indexed documents ({len(documents)})")
    if not documents:
        st.sidebar.caption("Upload a PDF to get started.")
        return

    for document in documents:
        st.sidebar.caption(
            f"{document.filename} — {document.page_count} pages, "
            f"{document.chunk_count} passages"
        )
    documents_by_hash = {document.sha256: document for document in documents}
    selected_digest = st.sidebar.selectbox(
        "Select a document to remove",
        options=list(documents_by_hash),
        format_func=lambda digest: documents_by_hash[digest].filename,
        key="document_to_remove",
    )
    with st.sidebar.form("remove_document_form"):
        confirm_delete = st.checkbox("Confirm removal from the index")
        delete_file = st.checkbox(
            "Also delete the PDF if it is an app-managed upload"
        )
        submitted = st.form_submit_button(
            "Remove document",
            disabled=not confirm_delete,
        )
    if submitted:
        try:
            with closing(open_database()) as connection:
                filename, file_deleted = delete_indexed_document(
                    connection,
                    selected_digest,
                    delete_uploaded_file=delete_file,
                )
            suffix = (
                " and deleted its app-managed PDF"
                if file_deleted
                else "; the original PDF file was kept"
            )
            st.sidebar.success(f"Removed {filename} from the index{suffix}.")
            st.rerun()
        except (OSError, ValueError, sqlite3.Error) as exc:
            st.sidebar.error(f"Could not remove document: {exc}")


def _render_answer(
    answer: GroundedAnswer,
    message_id: str,
    database_path: Path,
) -> None:
    if answer.abstained:
        st.warning(answer.answer)
        st.caption("Answer type: Abstention; no supported answer was generated.")
    else:
        st.info(
            "Answer type: Local LLM-generated, then validated as verbatim "
            "extractive evidence."
        )
        for statement_index, statement in enumerate(answer.statements):
            st.text(statement.text)
            for citation_id in statement.citation_ids:
                citation = next(
                    item
                    for item in answer.citations
                    if item.citation_id == citation_id
                )
                with closing(sqlite3.connect(database_path)) as connection:
                    path = citation_pdf_paths(connection, [citation]).get(
                        citation_id
                    )
                source = citation.source
                label = (
                    f"[{citation_id}] {source.filename}, "
                    f"page {source.page_number} — open PDF"
                )
                if path is None or not path.is_file():
                    st.caption(f"{label} (PDF file unavailable)")
                    continue
                if st.button(
                    label,
                    key=(
                        f"citation-{message_id}-{statement_index}-"
                        f"{citation_id}"
                    ),
                ):
                    st.session_state["selected_pdf"] = {
                        "path": str(path),
                        "filename": source.filename,
                        "page_number": source.page_number,
                    }

    with st.expander("Source passage previews"):
        if not answer.context.passages:
            st.caption("No passages were retrieved.")
        for passage in answer.context.passages:
            sources = "; ".join(
                f"{source.filename}, page {source.page_number}"
                for source in passage.sources
            )
            st.markdown(f"**Passage {passage.rank}** — {sources}")
            st.text(passage.text)


def _render_selected_pdf() -> None:
    selected = st.session_state.get("selected_pdf")
    if not isinstance(selected, dict):
        return
    path = Path(selected["path"])
    if not path.is_file():
        st.error("The selected source PDF is no longer available.")
        return
    pdf_bytes = path.read_bytes()
    filename = str(selected["filename"])
    page_number = int(selected["page_number"])
    st.subheader(f"{filename} — page {page_number}")
    st.download_button(
        "Download source PDF",
        data=pdf_bytes,
        file_name=path.name,
        mime="application/pdf",
        key="download-selected-pdf",
    )
    components.html(
        pdf_viewer_html(pdf_bytes, page_number),
        height=780,
        scrolling=True,
    )


def _render_chat(database_path: Path, search_mode: str) -> None:
    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            elif "error" in message:
                st.error(message["error"])
            else:
                _render_answer(
                    message["answer"],
                    message["message_id"],
                    database_path,
                )

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        config = RagPipelineConfig(
            database=database_path,
            rerank=search_mode == "Hybrid + Reranking",
        )
        try:
            with st.spinner("Searching documents and preparing an answer..."):
                answer = run_rag_pipeline(question, config=config)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            st.error(f"Unable to answer this question: {exc}")
            messages.append({"role": "assistant", "error": str(exc)})
        else:
            message_id = hashlib.sha256(
                f"{question}:{len(messages)}".encode("utf-8")
            ).hexdigest()[:16]
            messages.append(
                {
                    "role": "assistant",
                    "answer": answer,
                    "message_id": message_id,
                    "search_mode": search_mode,
                }
            )
            _render_answer(answer, message_id, database_path)


def main() -> None:
    st.set_page_config(page_title="NEXUS RAG", page_icon="N", layout="wide")
    st.title("NEXUS RAG")
    st.caption("Local PDF search with traceable, evidence-grounded answers.")

    search_mode = st.sidebar.radio(
        "Search mode",
        options=("Hybrid", "Hybrid + Reranking"),
        help=(
            "Hybrid uses dense search and BM25 with RRF. Reranking adds the "
            "local cross-encoder."
        ),
    )
    _render_document_manager()

    _render_chat(DATABASE_PATH, search_mode)
    _render_selected_pdf()


if __name__ == "__main__":
    main()
