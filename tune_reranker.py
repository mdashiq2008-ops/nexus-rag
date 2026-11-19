from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pymupdf

import step1
from reranker import CrossEncoderReranker


DEFAULT_BENCHMARK = Path(__file__).parent / "benchmarks" / "retrieval_benchmark.json"
DEFAULT_OUTPUT = Path(__file__).parent / "benchmarks" / "reranker_tuning_results.json"
BATCH_SIZES = (1, 2, 4, 8, 16)
THREAD_COUNTS = (1, 2, 4)
TUNING_CANDIDATE_K = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_working_set_bytes() -> int:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_memory_info.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory_info(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def _prepare_pairs(
    benchmark: dict[str, Any],
    pdf_dir: Path,
) -> tuple[list[tuple[str, str]], float, list[str]]:
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

    pairs: list[tuple[str, str]] = []
    with closing(sqlite3.connect(":memory:")) as connection:
        step1.initialize_database(connection)
        embedder = step1.LocalEmbedder()
        indexing_start = time.perf_counter()
        for path, _ in documents:
            step1.ingest_pdf(path, connection, embedder=embedder)
        indexing_seconds = time.perf_counter() - indexing_start

        target_keys = {
            (
                passage["pdf"],
                passage["page"],
                passage["char_start"],
                passage["char_end"],
            )
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
        if missing_targets := target_keys - indexed_keys:
            raise ValueError(
                "Annotated passages did not match the indexed chunks: "
                f"{sorted(missing_targets)[:5]}"
            )

        for question in benchmark["questions"]:
            candidates = step1.search_chunks_hybrid_candidates(
                connection,
                question["question"],
                candidate_k=TUNING_CANDIDATE_K,
                embedder=embedder,
            )
            if not candidates:
                raise ValueError(
                    f"No candidates for tuning question {question['id']}"
                )
            pairs.append((question["question"], candidates[0].text))

    return pairs, indexing_seconds, [document["pdf"] for _, document in documents]


def _run_worker(
    pairs_path: Path,
    *,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    pairs = [
        (query, document)
        for query, document in json.loads(
            pairs_path.read_text(encoding="utf-8")
        )
    ]
    load_start = time.perf_counter()
    reranker = CrossEncoderReranker(threads=threads)
    load_seconds = time.perf_counter() - load_start

    reranker.score_pairs(pairs[:1], batch_size=1)
    cpu_start = time.process_time()
    inference_start = time.perf_counter()
    scores, stages = reranker.score_pairs_profiled(
        pairs,
        batch_size=batch_size,
    )
    wall_seconds = time.perf_counter() - inference_start
    cpu_seconds = time.process_time() - cpu_start
    if len(scores) != len(pairs):
        raise ValueError("Reranker did not score every tuning pair")

    return {
        "batch_size": batch_size,
        "threads": threads,
        "pair_count": len(pairs),
        "model_load_seconds": load_seconds,
        "tokenization_seconds": stages["tokenization_seconds"],
        "inference_seconds": stages["inference_seconds"],
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "peak_working_set_bytes": _peak_working_set_bytes(),
    }


def _run_configuration(
    pairs_path: Path,
    *,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--pairs-file",
        str(pairs_path),
        "--batch-size",
        str(batch_size),
        "--threads",
        str(threads),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Tuning worker failed for batch_size={batch_size}, "
            f"threads={threads}:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def tune(
    benchmark: dict[str, Any],
    pdf_dir: Path,
    *,
    output: Path,
) -> dict[str, Any]:
    pairs, indexing_seconds, documents = _prepare_pairs(benchmark, pdf_dir)
    if len(pairs) != len(benchmark["questions"]):
        raise ValueError("Expected one tuning pair for every benchmark question")

    configurations: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="nexus-reranker-tuning-") as directory:
        pairs_path = Path(directory) / "pairs.json"
        pairs_path.write_text(
            json.dumps(pairs, ensure_ascii=False),
            encoding="utf-8",
        )
        for batch_size in BATCH_SIZES:
            for threads in THREAD_COUNTS:
                configurations.append(
                    _run_configuration(
                        pairs_path,
                        batch_size=batch_size,
                        threads=threads,
                    )
                )

    results = {
        "benchmark_version": benchmark["version"],
        "question_count": len(benchmark["questions"]),
        "sample_method": (
            "One top RRF candidate per question from candidate_k=10; "
            "the same query-passage pairs are scored for every configuration."
        ),
        "candidate_k": TUNING_CANDIDATE_K,
        "documents": documents,
        "indexing_seconds": indexing_seconds,
        "configurations": configurations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tune cross-encoder batch size and ONNX CPU threads."
    )
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pairs-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        if (
            args.pairs_file is None
            or args.batch_size is None
            or args.threads is None
        ):
            parser.error("worker mode requires pairs file, batch size, and threads")
        result = _run_worker(
            args.pairs_file,
            batch_size=args.batch_size,
            threads=args.threads,
        )
        print(json.dumps(result))
        return 0

    if args.pdf_dir is None:
        parser.error("--pdf-dir is required")
    with args.benchmark.open(encoding="utf-8") as source:
        benchmark = json.load(source)
    results = tune(benchmark, args.pdf_dir, output=args.output)

    print(
        f"Questions: {results['question_count']}; "
        f"pairs/configuration: {results['question_count']}; "
        f"indexing: {results['indexing_seconds']:.2f}s"
    )
    print(
        "Batch | Threads | Tokenize (s) | ONNX (s) | Wall (s) | "
        "CPU (s) | Peak working set (MiB)"
    )
    for row in results["configurations"]:
        print(
            f"{row['batch_size']:>5} | {row['threads']:>7} | "
            f"{row['tokenization_seconds']:>12.3f} | "
            f"{row['inference_seconds']:>8.3f} | "
            f"{row['wall_seconds']:>8.3f} | "
            f"{row['process_cpu_seconds']:>7.3f} | "
            f"{row['peak_working_set_bytes'] / 1024**2:>23.1f}"
        )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
