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
        "map": 0.27571023549983364,
        "rp@5": 0.5165079365079365,
        "rp@10": 0.4534807256235827,
        "mrr": 0.7105598099130719,
    },
    "BM25-cased": {
        "map": 0.27507750200419107,
        "rp@5": 0.5146031746031746,
        "rp@10": 0.45252834467120184,
        "mrr": 0.7105598099130719,
    },
    "TfIdf-word-lower": {
        "map": 0.28485413562322776,
        "rp@5": 0.5412698412698412,
        "rp@10": 0.47307256235827666,
        "mrr": 0.7111908432787992,
    },
    "TfIdf-word-cased": {
        "map": 0.28485413562322776,
        "rp@5": 0.5412698412698412,
        "rp@10": 0.47307256235827666,
        "mrr": 0.7111908432787992,
    },
    "TfIdf-char-lower": {
        "map": 0.33555610544690023,
        "rp@5": 0.584920634920635,
        "rp@10": 0.5117195767195767,
        "mrr": 0.7272701297287621,
    },
    "TfIdf-char-cased": {
        "map": 0.33555610544690023,
        "rp@5": 0.584920634920635,
        "rp@10": 0.5117195767195767,
        "mrr": 0.7272701297287621,
    },
    "EditDistance-lower": {
        "map": 0.22953525249793413,
        "rp@5": 0.42269841269841263,
        "rp@10": 0.3778987150415721,
        "mrr": 0.6176868635378816,
    },
    "EditDistance-cased": {
        "map": 0.24233603296868966,
        "rp@5": 0.4503174603174603,
        "rp@10": 0.4056538170823885,
        "mrr": 0.6348209280137445,
    },
    "RandomRanking": {
        "map": 0.01167258539227986,
        "rp@5": 0.009523809523809525,
        "rp@10": 0.010714285714285714,
        "mrr": 0.041147312519647754,
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
        results = job_similarity_task.evaluate(model, language=Language.EN)

        expected = EXPECTED_METRICS[model_name]
        print(f"\n[{model_name}] actual metrics: {results}")

        for metric_name, expected_value in expected.items():
            actual_value = results[metric_name]
            assert actual_value == pytest.approx(expected_value, abs=1e-3), (
                f"{model_name} metric '{metric_name}': "
                f"expected {expected_value}, got {actual_value}"
            )

    def test_all_default_metrics_present(self, job_similarity_task, model_name, model_factory):
        """Assert that all default ranking metrics appear in results."""
        model = model_factory()
        results = job_similarity_task.evaluate(model, language=Language.EN)

        for metric_name in job_similarity_task.default_metrics:
            assert metric_name in results, (
                f"{model_name}: missing default metric '{metric_name}' in results"
            )
