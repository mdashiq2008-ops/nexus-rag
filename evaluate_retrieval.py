from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pymupdf

import step1
from reranker import (
    CrossEncoderReranker,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_THREADS,
)


DEFAULT_BENCHMARK = Path(__file__).parent / "benchmarks" / "retrieval_benchmark.json"
DEFAULT_CANDIDATE_KS = (10, 20, 40)
RERANK_BATCH_SIZE = DEFAULT_RERANK_BATCH_SIZE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _passage_key(passage: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        passage["pdf"],
        passage["page"],
        passage["char_start"],
        passage["char_end"],
    )


def _result_key(result: step1.SearchResult) -> tuple[str, int, int, int]:
    return (
        result.filename,
        result.page_number,
        result.char_start,
        result.char_end,
    )


def _dcg(grades: list[int]) -> float:
    return math.fsum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
        if grade > 0
    )


def score_results(
    results: list[step1.SearchResult],
    question: dict[str, Any],
    *,
    top_k: int,
    latency_seconds: float,
) -> dict[str, Any]:
    relevant = {
        _passage_key(passage): passage["grade"]
        for passage in question["targets"]
    }
    ranked = results[:top_k]
    grades = [relevant.get(_result_key(result), 0) for result in ranked]
    hits = sum(grade > 0 for grade in grades)
    first_relevant = next(
        (rank for rank, grade in enumerate(grades, start=1) if grade > 0),
        None,
    )
    ideal_grades = sorted(relevant.values(), reverse=True)[:top_k]
    ideal_dcg = _dcg(ideal_grades)
    return {
        "recall_at_k": hits / len(relevant) if relevant else 0.0,
        "mrr_at_k": 1 / first_relevant if first_relevant else 0.0,
        "ndcg_at_k": _dcg(grades) / ideal_dcg if ideal_dcg else 0.0,
        "top1_accuracy": float(bool(grades and grades[0] > 0)),
        "latency_seconds": latency_seconds,
        "first_relevant_rank": first_relevant,
        "ranked_results": [
            {
                "rank": rank,
                "pdf": result.filename,
                "page": result.page_number,
                "char_start": result.char_start,
                "char_end": result.char_end,
                "score": result.score,
                "relevance_grade": grades[rank - 1],
            }
            for rank, result in enumerate(ranked, start=1)
        ],
    }


def evaluate(
    benchmark: dict[str, Any],
    pdf_dir: Path,
    *,
    top_k: int,
    candidate_ks: tuple[int, ...] = DEFAULT_CANDIDATE_KS,
    batch_size: int = RERANK_BATCH_SIZE,
    threads: int = DEFAULT_RERANK_THREADS,
) -> dict[str, Any]:
    if top_k <= 0 or not candidate_ks or any(
        candidate_k < top_k for candidate_k in candidate_ks
    ):
        raise ValueError("Require candidate_k >= top_k > 0")
    if any(candidate_k <= 0 for candidate_k in candidate_ks):
        raise ValueError("candidate_k values must be greater than zero")
    if len(set(candidate_ks)) != len(candidate_ks):
        raise ValueError("candidate_k values must be unique")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if threads is not None and threads <= 0:
        raise ValueError("threads must be greater than zero")
    if not benchmark["questions"]:
        raise ValueError("Benchmark must contain at least one question")

    documents: list[tuple[Path, dict[str, Any]]] = []
    for document in benchmark["documents"]:
        path = pdf_dir / document["pdf"]
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark PDF not found: {path}")
        if _sha256(path) != document["sha256"]:
            raise ValueError(f"Benchmark PDF hash does not match: {path.name}")
        with pymupdf.open(path) as pdf:
            if len(pdf) != document["pages"]:
                raise ValueError(
                    f"Page count mismatch for {path.name}: "
                    f"expected {document['pages']}, got {len(pdf)}"
                )
        documents.append((path, document))

    with closing(sqlite3.connect(":memory:")) as connection:
        step1.initialize_database(connection)
        embedder_start = time.perf_counter()
        embedder = step1.LocalEmbedder()
        embedder_load_seconds = time.perf_counter() - embedder_start

        indexing_start = time.perf_counter()
        for path, _ in documents:
            step1.ingest_pdf(path, connection, embedder=embedder)
        indexing_seconds = time.perf_counter() - indexing_start

        target_keys = {
            _passage_key(passage)
            for question in benchmark["questions"]
            for passage in question["targets"]
        }
        indexed_keys = {
            (filename, page, start, end)
            for filename, page, start, end in connection.execute(
                """
                SELECT d.filename, c.page_number, c.char_start, c.char_end
                FROM chunks AS c
                JOIN documents AS d ON d.sha256 = c.document_sha256
                """
            )
        }
        missing_targets = target_keys - indexed_keys
        if missing_targets:
            missing_details = "\n".join(
                f"{pdf} page {page} chars {start}-{end}"
                for pdf, page, start, end in sorted(missing_targets)
            )
            raise ValueError(
                "Annotated passages did not match the indexed chunks:\n"
                f"{missing_details}"
            )

        reranker_start = time.perf_counter()
        reranker = CrossEncoderReranker(threads=threads)
        reranker_load_seconds = time.perf_counter() - reranker_start

        first_question = benchmark["questions"][0]["question"]
        warm_candidates = step1.search_chunks_hybrid_candidates(
            connection,
            first_question,
            candidate_k=max(candidate_ks),
            embedder=embedder,
        )
        warm_pairs = (
            [(first_question, warm_candidates[0].text)]
            if warm_candidates
            else []
        )
        reranker.score_pairs(warm_pairs, batch_size=batch_size)

        experiments: dict[str, Any] = {}
        for candidate_k in candidate_ks:
            retrieved: list[dict[str, Any]] = []
            for question in benchmark["questions"]:
                stage_timings: dict[str, float] = {}
                retrieval_start = time.perf_counter()
                candidates = step1.search_chunks_hybrid_candidates(
                    connection,
                    question["question"],
                    candidate_k=candidate_k,
                    embedder=embedder,
                    timings=stage_timings,
                )
                stage_timings["retrieval_seconds"] = (
                    time.perf_counter() - retrieval_start
                )
                retrieved.append(
                    {
                        "question": question,
                        "candidates": candidates,
                        "timings": stage_timings,
                    }
                )

            rrf_rows: list[dict[str, Any]] = []
            for row in retrieved:
                question = row["question"]
                rrf_rows.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        **score_results(
                            row["candidates"],
                            question,
                            top_k=top_k,
                            latency_seconds=row["timings"][
                                "retrieval_seconds"
                            ],
                        ),
                        "stage_timings": row["timings"],
                    }
                )

            pairs: list[tuple[str, str]] = []
            pair_ranges: list[tuple[int, int]] = []
            for row in retrieved:
                start = len(pairs)
                query = row["question"]["question"]
                pairs.extend(
                    (query, candidate.text)
                    for candidate in row["candidates"]
                )
                pair_ranges.append((start, len(pairs)))

            rerank_start = time.perf_counter()
            scores, reranker_timings = reranker.score_pairs_profiled(
                pairs,
                batch_size=batch_size,
            )
            rerank_batch_seconds = time.perf_counter() - rerank_start
            if len(scores) != len(pairs):
                raise ValueError(
                    "Reranker must return one score for each candidate pair"
                )
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("Reranker returned a non-finite score")

            reranked_rows: list[dict[str, Any]] = []
            for row, (start, end) in zip(retrieved, pair_ranges):
                question = row["question"]
                ranked = [
                    result
                    for result, _ in sorted(
                        zip(row["candidates"], scores[start:end]),
                        key=lambda item: (-item[1], item[0].chunk_id),
                    )
                ]
                pair_count = end - start
                allocated_rerank_seconds = (
                    rerank_batch_seconds * pair_count / len(pairs)
                    if pairs
                    else 0.0
                )
                reranked_rows.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        **score_results(
                            ranked,
                            question,
                            top_k=top_k,
                            latency_seconds=(
                                row["timings"]["retrieval_seconds"]
                                + allocated_rerank_seconds
                            ),
                        ),
                        "stage_timings": {
                            **row["timings"],
                            "reranking_allocated_seconds": (
                                allocated_rerank_seconds
                            ),
                        },
                    }
                )

            def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
                return {
                    "recall_at_k": statistics.mean(
                        row["recall_at_k"] for row in rows
                    ),
                    "mrr_at_k": statistics.mean(
                        row["mrr_at_k"] for row in rows
                    ),
                    "ndcg_at_k": statistics.mean(
                        row["ndcg_at_k"] for row in rows
                    ),
                    "top1_accuracy": statistics.mean(
                        row["top1_accuracy"] for row in rows
                    ),
                    "mean_latency_seconds": statistics.mean(
                        row["latency_seconds"] for row in rows
                    ),
                    "median_latency_seconds": statistics.median(
                        row["latency_seconds"] for row in rows
                    ),
                }

            rrf_summary = summarize(rrf_rows)
            reranked_summary = summarize(reranked_rows)
            question_count = len(benchmark["questions"])
            rrf_summary["stage_timings_mean_seconds"] = {
                stage: statistics.mean(
                    row["stage_timings"][stage] for row in rrf_rows
                )
                for stage in (
                    "dense_search_seconds",
                    "bm25_search_seconds",
                    "rrf_fusion_seconds",
                    "retrieval_seconds",
                )
            }
            reranked_summary["stage_timings"] = {
                "dense_search_mean_seconds_per_question": statistics.mean(
                    row["stage_timings"]["dense_search_seconds"]
                    for row in reranked_rows
                ),
                "bm25_search_mean_seconds_per_question": statistics.mean(
                    row["stage_timings"]["bm25_search_seconds"]
                    for row in reranked_rows
                ),
                "rrf_fusion_mean_seconds_per_question": statistics.mean(
                    row["stage_timings"]["rrf_fusion_seconds"]
                    for row in reranked_rows
                ),
                "tokenization_batch_seconds": reranker_timings[
                    "tokenization_seconds"
                ],
                "cross_encoder_inference_batch_seconds": reranker_timings[
                    "inference_seconds"
                ],
                "reranking_batch_wall_seconds": rerank_batch_seconds,
                "reranking_amortized_seconds_per_question": (
                    rerank_batch_seconds / question_count
                ),
                "reranked_pair_count": len(pairs),
                "batch_size": batch_size,
            }
            experiments[str(candidate_k)] = {
                "summary": {
                    "A": rrf_summary,
                    "B": reranked_summary,
                },
                "per_question": {
                    "A": rrf_rows,
                    "B": reranked_rows,
                },
            }

    return {
        "benchmark_version": benchmark["version"],
        "question_count": len(benchmark["questions"]),
        "top_k": top_k,
        "candidate_ks": list(candidate_ks),
        "rerank_batch_size": batch_size,
        "rerank_threads": threads,
        "documents": [document["pdf"] for _, document in documents],
        "models": {
            "embedder": step1.DEFAULT_EMBEDDING_MODEL,
            "reranker": DEFAULT_RERANKER_MODEL,
        },
        "setup_seconds": {
            "embedder_load": embedder_load_seconds,
            "indexing": indexing_seconds,
            "reranker_load": reranker_load_seconds,
        },
        "metric_definitions": {
            "recall_at_k": "Relevant annotated passages retrieved divided by all relevant annotated passages.",
            "mrr_at_k": "Reciprocal rank of the first relevant passage, or zero when absent.",
            "ndcg_at_k": "Normalized DCG using annotated relevance grades and gain 2^grade - 1.",
            "top1_accuracy": "Fraction of questions with a relevant passage at rank one.",
            "latency": (
                "Warm retrieval wall time; for reranked results, the batch "
                "reranking wall time is allocated across questions in "
                "proportion to their candidate-pair counts. Model loading and "
                "indexing are excluded."
            ),
        },
        "experiments": experiments,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval with and without cross-encoder reranking."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing the PDF files listed in the benchmark",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--candidate-k",
        type=int,
        action="append",
        dest="candidate_ks",
        help="Candidate count to evaluate; repeat to compare several (default: 10, 20, 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=RERANK_BATCH_SIZE,
        help=f"Cross-encoder inference batch size (default: {RERANK_BATCH_SIZE})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_RERANK_THREADS,
        help="ONNX intra-op and inter-op thread count for cross-encoder inference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for complete per-question JSON results",
    )
    args = parser.parse_args(argv)

    with args.benchmark.open(encoding="utf-8") as source:
        benchmark = json.load(source)
    top_k = (
        args.top_k if args.top_k is not None else benchmark["top_k"]
    )
    candidate_ks = tuple(args.candidate_ks or DEFAULT_CANDIDATE_KS)
    results = evaluate(
        benchmark,
        args.pdf_dir,
        top_k=top_k,
        candidate_ks=candidate_ks,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    print(
        f"Questions: {results['question_count']}; "
        f"top_k={top_k}; "
        f"candidate_ks={','.join(str(k) for k in candidate_ks)}; "
        f"batch_size={args.batch_size}; threads={args.threads or 'default'}"
    )
    print(
        f"Setup (s): embedder={results['setup_seconds']['embedder_load']:.2f}, "
        f"index={results['setup_seconds']['indexing']:.2f}, "
        f"reranker={results['setup_seconds']['reranker_load']:.2f}"
    )
    print(
        f"Experiment | Recall@{top_k} | MRR@{top_k} | nDCG@{top_k} | Top-1 | "
        "Mean latency (s) | Median latency (s)"
    )
    for candidate_k, experiment in results["experiments"].items():
        for label, name in (("A", "RRF"), ("B", "RRF+CrossEncoder")):
            summary = experiment["summary"][label]
            print(
                f"k={candidate_k:>2} {name:15} | "
                f"{summary['recall_at_k']:.3f} | "
                f"{summary['mrr_at_k']:.3f} | "
                f"{summary['ndcg_at_k']:.3f} | "
                f"{summary['top1_accuracy']:.3f} | "
                f"{summary['mean_latency_seconds']:.4f} | "
                f"{summary['median_latency_seconds']:.4f}"
            )
        timings = experiment["summary"]["B"]["stage_timings"]
        print(
            f"  k={candidate_k}: dense={timings['dense_search_mean_seconds_per_question']:.4f}s/q, "
            f"BM25={timings['bm25_search_mean_seconds_per_question']:.4f}s/q, "
            f"RRF={timings['rrf_fusion_mean_seconds_per_question']:.4f}s/q, "
            f"tokenize={timings['tokenization_batch_seconds']:.2f}s/batch, "
            f"ONNX={timings['cross_encoder_inference_batch_seconds']:.2f}s/batch "
            f"({timings['reranked_pair_count']} pairs)"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as destination:
            json.dump(results, destination, indent=2, ensure_ascii=False)
            destination.write("\n")
        print(f"Detailed results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
