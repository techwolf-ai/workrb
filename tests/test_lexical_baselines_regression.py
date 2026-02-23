"""Regression tests for lexical baseline models.

Validates that each lexical baseline model produces expected metric values
when evaluated on the JobTitleSimilarityRanking task (English, test split,
105 queries x 2,619 targets).

To re-bootstrap expected values after an intentional change:
    1. Set all values in EXPECTED_METRICS to 0.0
    2. Run: uv run pytest tests/test_lexical_baselines_regression.py -v -s
    3. Copy printed actual values into EXPECTED_METRICS
    4. Re-run without -s to confirm all tests pass
"""

from collections.abc import Callable

import pytest

from workrb.models.lexical_baselines import (
    BM25Model,
    EditDistanceModel,
    RandomRankingModel,
    TfIdfModel,
)
from workrb.tasks.abstract.base import Language
from workrb.tasks.ranking.job_similarity import JobTitleSimilarityRanking

# ---------------------------------------------------------------------------
# Expected metric values per model variant.
# These were recorded from a run on the JobTitleSimilarityRanking task
# (English, test split, 105 queries x 2,619 targets).
# See module docstring for how to re-bootstrap after intentional changes.
# ---------------------------------------------------------------------------
EXPECTED_METRICS: dict[str, dict[str, float]] = {
    "BM25-lower": {
        "map": 0.28,
        "rp@5": 0.52,
        "rp@10": 0.45,
        "mrr": 0.71,
    },
    "BM25-cased": {
        "map": 0.28,
        "rp@5": 0.51,
        "rp@10": 0.45,
        "mrr": 0.71,
    },
    "TfIdf-word-lower": {
        "map": 0.28,
        "rp@5": 0.54,
        "rp@10": 0.47,
        "mrr": 0.71,
    },
    "TfIdf-word-cased": {
        "map": 0.28,
        "rp@5": 0.54,
        "rp@10": 0.47,
        "mrr": 0.71,
    },
    "TfIdf-char-lower": {
        "map": 0.34,
        "rp@5": 0.58,
        "rp@10": 0.51,
        "mrr": 0.73,
    },
    "TfIdf-char-cased": {
        "map": 0.34,
        "rp@5": 0.58,
        "rp@10": 0.51,
        "mrr": 0.73,
    },
    "EditDistance-lower": {
        "map": 0.23,
        "rp@5": 0.42,
        "rp@10": 0.38,
        "mrr": 0.62,
    },
    "EditDistance-cased": {
        "map": 0.24,
        "rp@5": 0.45,
        "rp@10": 0.41,
        "mrr": 0.63,
    },
    "RandomRanking": {
        "map": 0.01,
        "rp@5": 0.01,
        "rp@10": 0.01,
        "mrr": 0.04,
    },
}

# Model factories: callables that return a fresh model instance.
# Using factories (not pre-instantiated objects) ensures RandomRankingModel
# is freshly seeded on each test invocation, guaranteeing reproducibility.
MODEL_VARIANTS: list[tuple[str, Callable]] = [
    ("BM25-lower", lambda: BM25Model(lowercase=True)),
    ("BM25-cased", lambda: BM25Model(lowercase=False)),
    ("TfIdf-word-lower", lambda: TfIdfModel(lowercase=True, tokenization="word")),
    ("TfIdf-word-cased", lambda: TfIdfModel(lowercase=False, tokenization="word")),
    ("TfIdf-char-lower", lambda: TfIdfModel(lowercase=True, tokenization="char")),
    ("TfIdf-char-cased", lambda: TfIdfModel(lowercase=False, tokenization="char")),
    ("EditDistance-lower", lambda: EditDistanceModel(lowercase=True)),
    ("EditDistance-cased", lambda: EditDistanceModel(lowercase=False)),
    ("RandomRanking", lambda: RandomRankingModel(seed=42)),
]


@pytest.fixture(scope="module")
def job_similarity_task():
    """Create the JobTitleSimilarityRanking task (loaded once per module)."""
    return JobTitleSimilarityRanking(split="test", languages=["en"])


@pytest.mark.parametrize(
    ("model_name", "model_factory"),
    MODEL_VARIANTS,
    ids=[name for name, _ in MODEL_VARIANTS],
)
class TestLexicalBaselineRegression:
    """Regression tests asserting metric values for each lexical baseline."""

    def test_metrics_match_expected(self, job_similarity_task, model_name, model_factory):
        """Assert that evaluation metrics match pre-recorded expected values."""
        model = model_factory()
        results = job_similarity_task.evaluate(model, dataset_id=Language.EN.value)

        expected = EXPECTED_METRICS[model_name]
        print(f"\n[{model_name}] actual metrics: {results}")

        for metric_name, expected_value in expected.items():
            actual_value = results[metric_name]
            assert actual_value == pytest.approx(expected_value, abs=1e-2), (
                f"{model_name} metric '{metric_name}': "
                f"expected {expected_value}, got {actual_value}"
            )

    def test_all_default_metrics_present(self, job_similarity_task, model_name, model_factory):
        """Assert that all default ranking metrics appear in results."""
        model = model_factory()
        results = job_similarity_task.evaluate(model, dataset_id=Language.EN.value)

        for metric_name in job_similarity_task.default_metrics:
            assert metric_name in results, (
                f"{model_name}: missing default metric '{metric_name}' in results"
            )
