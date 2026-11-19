from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol, Sequence, cast


DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_BATCH_SIZE = 1
DEFAULT_RERANK_THREADS = 2


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class _TimedSession:
    def __init__(self, session: Any, timings: dict[str, float]) -> None:
        self._session = session
        self._timings = timings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._session.run(*args, **kwargs)
        finally:
            self._timings["inference_seconds"] += time.perf_counter() - started


class _ProfiledModel(Protocol):
    model: Any

    def load_onnx_model(self) -> None: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        threads: int | None = DEFAULT_RERANK_THREADS,
    ) -> None:
        if threads is not None and threads <= 0:
            raise ValueError("threads must be greater than zero")
        active_threads = threads or DEFAULT_RERANK_THREADS
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires FastEmbed >=0.9.1. "
                "Install project dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        cache_dir = (
            Path(os.environ.get("LOCALAPPDATA", Path.home()))
            / "fastembed_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any | None = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=active_threads,
        )

    def close(self) -> None:
        self._model = None

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Cross-encoder has been closed")
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return self.score_pairs(
            [(query, document) for document in documents],
            batch_size=DEFAULT_RERANK_BATCH_SIZE,
        )

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ) -> list[float]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not pairs:
            return []
        model = self._require_model()
        return [
            float(score)
            for score in model.rerank_pairs(
                list(pairs),
                batch_size=batch_size,
                parallel=None,
            )
        ]

    def score_pairs_profiled(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ) -> tuple[list[float], dict[str, float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        timings = {"tokenization_seconds": 0.0, "inference_seconds": 0.0}
        if not pairs:
            return [], timings

        model = cast(_ProfiledModel, self._require_model().model)
        if model.model is None:
            model.load_onnx_model()

        original_session = model.model
        original_tokenize = getattr(model, "tokenize")
        model_attributes = vars(model)
        had_tokenize_attribute = "tokenize" in model_attributes
        tokenize_attribute = model_attributes.get("tokenize")

        def timed_tokenize(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_tokenize(*args, **kwargs)
            finally:
                timings["tokenization_seconds"] += (
                    time.perf_counter() - started
                )

        setattr(model, "tokenize", timed_tokenize)
        model.model = _TimedSession(original_session, timings)
        try:
            scores = self.score_pairs(pairs, batch_size=batch_size)
        finally:
            model.model = original_session
            if not had_tokenize_attribute:
                delattr(model, "tokenize")
            else:
                setattr(model, "tokenize", tokenize_attribute)
        return scores, timings
