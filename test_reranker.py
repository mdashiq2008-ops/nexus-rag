import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from fastembed.rerank.cross_encoder import TextCrossEncoder

from reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_loads_model_once_and_converts_scores_to_floats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "fastembed_cache"
            with patch.dict("os.environ", {"LOCALAPPDATA": temp_dir}):
                with patch(
                    "fastembed.rerank.cross_encoder.TextCrossEncoder"
                ) as cross_encoder:
                    model = cross_encoder.return_value
                    model.rerank_pairs.side_effect = [
                        iter([0.25, 0.75]),
                        iter([0.5]),
                    ]
                    reranker = CrossEncoderReranker()

                    self.assertEqual(
                        reranker.score("query", ["first", "second"]),
                        [0.25, 0.75],
                    )
                    self.assertEqual(
                        reranker.score("another query", ["third"]),
                        [0.5],
                    )

            cross_encoder.assert_called_once_with(
                model_name=DEFAULT_RERANKER_MODEL,
                cache_dir=str(cache_dir),
                threads=2,
            )
            self.assertEqual(model.rerank_pairs.call_count, 2)
            model.rerank_pairs.assert_any_call(
                [("query", "first"), ("query", "second")],
                batch_size=1,
                parallel=None,
            )

    def test_passes_thread_count_to_fastembed(self) -> None:
        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as model:
            CrossEncoderReranker(threads=2)

        model.assert_called_once()
        self.assertEqual(model.call_args.kwargs["threads"], 2)

    def test_close_releases_cross_encoder_and_prevents_reuse(self) -> None:
        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as model:
            reranker = CrossEncoderReranker()
            reranker.close()

            with self.assertRaisesRegex(RuntimeError, "has been closed"):
                reranker.score("query", ["passage"])

        model.return_value.rerank_pairs.assert_not_called()

    def test_rejects_nonpositive_thread_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "threads"):
            CrossEncoderReranker(threads=0)

    def test_empty_candidates_do_not_call_model(self) -> None:
        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as model:
            reranker = CrossEncoderReranker()

            self.assertEqual(reranker.score("query", []), [])
            self.assertEqual(reranker.score_pairs([]), [])

        model.return_value.rerank_pairs.assert_not_called()

    def test_batch_scoring_preserves_pair_order_and_batch_size(self) -> None:
        pairs = [("query one", "first"), ("query two", "second")]
        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as model:
            model.return_value.rerank_pairs.return_value = iter([0.4, 0.8])
            reranker = CrossEncoderReranker()

            scores = reranker.score_pairs(pairs, batch_size=2)

        self.assertEqual(scores, [0.4, 0.8])
        model.return_value.rerank_pairs.assert_called_once_with(
            pairs,
            batch_size=2,
            parallel=None,
        )

    def test_batch_scoring_rejects_nonpositive_batch_size(self) -> None:
        with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as model:
            reranker = CrossEncoderReranker()

            with self.assertRaisesRegex(ValueError, "batch_size"):
                reranker.score_pairs([("query", "passage")], batch_size=0)

        model.return_value.rerank_pairs.assert_not_called()

    def test_profiled_batch_scoring_separates_tokenization_and_inference(self) -> None:
        class FakeSession:
            def run(self, _output_names, _onnx_input):
                return (
                    [0.25, 0.75]
                    if _output_names is None and _onnx_input is None
                    else []
                )

        class FakeOnnxModel:
            def __init__(self) -> None:
                self.model = FakeSession()

            def tokenize(self, pairs):
                return list(pairs)

        class FakeModel:
            def __init__(self) -> None:
                self.model = FakeOnnxModel()

            def rerank_pairs(self, pairs, batch_size=64, parallel=None):
                if batch_size <= 0 or parallel is not None:
                    raise ValueError("unexpected batch configuration")
                self.model.tokenize(pairs)
                return iter(self.model.model.run(None, None))

        fake_model = FakeModel()
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = cast(TextCrossEncoder, fake_model)

        scores, timings = reranker.score_pairs_profiled(
            [("first query", "first passage"), ("second query", "second passage")],
            batch_size=2,
        )

        self.assertEqual(scores, [0.25, 0.75])
        self.assertGreaterEqual(timings["tokenization_seconds"], 0)
        self.assertGreaterEqual(timings["inference_seconds"], 0)
        self.assertIsInstance(fake_model.model.model, FakeSession)
        self.assertNotIn("tokenize", fake_model.model.__dict__)


if __name__ == "__main__":
    unittest.main()
