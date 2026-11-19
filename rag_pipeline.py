from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import step1
from answer_generation import (
    AnswerGenerator,
    GroundedAnswer,
    answer_question,
)
from context_builder import TokenCounter
from qwen3_llama_cpp import (
    Qwen3LlamaCppConfig,
    Qwen3LlamaCppGenerator,
)
from reranker import CrossEncoderReranker


@dataclass(frozen=True)
class RagPipelineConfig:
    database: Path = step1.DEFAULT_DATABASE
    top_k: int = 5
    candidate_k: int = step1.HYBRID_CANDIDATE_K
    rerank: bool = True
    rerank_threads: int = 2
    max_context_tokens: int = 2500
    qwen: Qwen3LlamaCppConfig = field(default_factory=Qwen3LlamaCppConfig)

    def __post_init__(self) -> None:
        if not str(self.database):
            raise ValueError("database path must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.rerank_threads <= 0:
            raise ValueError("rerank_threads must be greater than zero")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero")


def run_rag_pipeline(
    question: str,
    *,
    config: RagPipelineConfig | None = None,
    embedder: step1.Embedder | None = None,
    generator: AnswerGenerator | None = None,
    token_counter: TokenCounter | None = None,
) -> GroundedAnswer:
    if not question.strip():
        raise ValueError("Question must not be empty")
    active_config = config or RagPipelineConfig()

    reranker = (
        CrossEncoderReranker(threads=active_config.rerank_threads)
        if active_config.rerank
        else None
    )
    try:
        active_config.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(active_config.database)) as connection:
            step1.initialize_database(connection)
            ranked_results = step1.search_chunks_hybrid(
                connection,
                question,
                top_k=active_config.top_k,
                candidate_k=active_config.candidate_k,
                embedder=embedder,
                reranker=reranker,
            )
    finally:
        if reranker is not None:
            reranker.close()

    active_generator = generator or Qwen3LlamaCppGenerator(active_config.qwen)
    active_token_counter = token_counter
    if active_token_counter is None:
        if not isinstance(active_generator, Qwen3LlamaCppGenerator):
            raise ValueError(
                "token_counter is required when using a custom generator"
            )
        active_token_counter = active_generator.count_tokens

    try:
        return answer_question(
            question,
            ranked_results,
            generator=active_generator,
            max_context_tokens=active_config.max_context_tokens,
            token_counter=active_token_counter,
        )
    finally:
        if (
            isinstance(active_generator, Qwen3LlamaCppGenerator)
            and not active_generator.config.keep_server_running
        ):
            active_generator.close()


def _print_answer(answer: GroundedAnswer) -> None:
    print(answer.answer)
    if not answer.citations:
        return
    print("\nCitations:")
    for citation in answer.citations:
        source = citation.source
        chunks = ", ".join(str(chunk_id) for chunk_id in source.chunk_ids)
        print(
            f"[{citation.citation_id}] {source.filename}, page "
            f"{source.page_number}, chars {source.char_start}-{source.char_end}"
            f"{f', chunks {chunks}' if chunks else ''}"
            f"{', OCR' if source.ocr_used else ''}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Answer a question using hybrid PDF retrieval, optional "
            "cross-encoder reranking, and grounded local generation."
        )
    )
    parser.add_argument("question", help="Question to answer from indexed PDFs")
    parser.add_argument(
        "--database",
        type=Path,
        default=step1.DEFAULT_DATABASE,
        help=f"SQLite database path (default: {step1.DEFAULT_DATABASE})",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=step1.HYBRID_CANDIDATE_K,
        help="Fused candidates sent to the reranker",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Use RRF results without cross-encoder reranking",
    )
    parser.add_argument("--rerank-threads", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument(
        "--server-executable",
        type=str,
        help="Override the llama-server executable path",
    )
    args = parser.parse_args(argv)

    qwen_config = Qwen3LlamaCppConfig(
        server_executable=(
            args.server_executable
            if args.server_executable is not None
            else Qwen3LlamaCppConfig().server_executable
        )
    )
    try:
        config = RagPipelineConfig(
            database=args.database,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank=not args.no_rerank,
            rerank_threads=args.rerank_threads,
            max_context_tokens=args.max_context_tokens,
            qwen=qwen_config,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        answer = run_rag_pipeline(args.question, config=config)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_answer(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
