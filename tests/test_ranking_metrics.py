"""Tests for ranking metrics (workrb.metrics.ranking)."""

import math

import numpy as np
import pytest
import torch

from workrb.metrics.ranking import calculate_ranking_metrics


def _prediction_matrix_from_order(order: list[int], n_targets: int) -> np.ndarray:
    """Build a 1-query prediction matrix that ranks targets in the given order (best first)."""
    scores = np.zeros((1, n_targets), dtype=float)
    for rank, idx in enumerate(order):
        scores[0, idx] = float(n_targets - rank)
    return scores


# Shared test fixtures: 2 queries x 5 targets
# Query 0 sorted order: [0, 2, 4, 3, 1]  (scores: 0.9, 0.1, 0.8, 0.2, 0.5)
# Query 1 sorted order: [3, 1, 2, 0, 4]  (scores: 0.3, 0.7, 0.5, 0.9, 0.1)
PREDICTION_MATRIX = np.array(
    [
        [0.9, 0.1, 0.8, 0.2, 0.5],
        [0.3, 0.7, 0.5, 0.9, 0.1],
    ]
)
POS_LABEL_IDXS = [[2, 4], [1, 3]]


class TestNDCGAtK:
    """NDCG@k with hand-computed expected values."""

    def test_ndcg_at_k_hand_computed(self):
        """
        Query 0: sorted [0,2,4,3,1], relevant {2,4}
          top-3: [0,2,4] -> DCG = 0 + 1/log2(3) + 1/log2(4) = 0.63093 + 0.5 = 1.13093
          IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
          NDCG@3 = 1.13093 / 1.63093 = 0.69342

        Query 1: sorted [3,1,2,0,4], relevant {1,3}
          top-3: [3,1,2] -> DCG = 1/log2(2) + 1/log2(3) + 0 = 1.0 + 0.63093 = 1.63093
          IDCG = 1.0 + 0.63093 = 1.63093
          NDCG@3 = 1.0

        Mean NDCG@3 = (0.69342 + 1.0) / 2 = 0.84671
        """
        results = calculate_ranking_metrics(PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg@3"])
        assert results["ndcg@3"] == pytest.approx(0.84671, abs=1e-4)

    def test_ndcg_at_k_1(self):
        """
        Query 0: top-1 is idx 0, not relevant -> NDCG@1 = 0.0
        Query 1: top-1 is idx 3, relevant -> NDCG@1 = 1.0
        Mean = 0.5
        """
        results = calculate_ranking_metrics(PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg@1"])
        assert results["ndcg@1"] == pytest.approx(0.5, abs=1e-4)


class TestNDCGFullList:
    """NDCG over the full ranked list (no @k)."""

    def test_ndcg_full_list(self):
        """Full-list NDCG should equal NDCG@5 for a 5-column matrix."""
        results = calculate_ranking_metrics(PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg"])
        assert results["ndcg"] == pytest.approx(0.84671, abs=1e-4)

    def test_ndcg_full_equals_ndcg_at_n_targets(self):
        """Ndcg and ndcg@n_targets must be identical."""
        results = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg", "ndcg@5"]
        )
        assert results["ndcg"] == pytest.approx(results["ndcg@5"])


class TestNDCGEdgeCases:
    """Edge cases for NDCG computation."""

    def test_perfect_ranking(self):
        """When relevant items are ranked first, NDCG = 1.0."""
        prediction_matrix = np.array([[0.1, 0.9, 0.8]])  # sorted: [1, 2, 0]
        pos_label_idxs = [[1, 2]]
        results = calculate_ranking_metrics(prediction_matrix, pos_label_idxs, metrics=["ndcg@3"])
        assert results["ndcg@3"] == pytest.approx(1.0, abs=1e-4)

    def test_worst_ranking(self):
        """
        Relevant items ranked last.
        prediction_matrix = [[0.9, 0.1, 0.2]]  sorted: [0, 2, 1]
        relevant: {1, 2}
        DCG@3 = 0 + 1/log2(3) + 1/log2(4) = 0.63093 + 0.5 = 1.13093
        IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
        NDCG = 0.69342
        """
        prediction_matrix = np.array([[0.9, 0.1, 0.2]])
        pos_label_idxs = [[1, 2]]
        results = calculate_ranking_metrics(prediction_matrix, pos_label_idxs, metrics=["ndcg@3"])
        assert results["ndcg@3"] == pytest.approx(0.69342, abs=1e-4)

    def test_no_relevant_items(self):
        """Queries with no positives are skipped; empty result -> 0.0."""
        prediction_matrix = np.array([[0.5, 0.3, 0.9]])
        pos_label_idxs = [[]]
        results = calculate_ranking_metrics(prediction_matrix, pos_label_idxs, metrics=["ndcg@3"])
        assert results["ndcg@3"] == pytest.approx(0.0, abs=1e-4)

    def test_all_items_relevant(self):
        """When all items are relevant, any ordering gives NDCG = 1.0."""
        prediction_matrix = np.array([[0.5, 0.3, 0.9]])
        pos_label_idxs = [[0, 1, 2]]
        results = calculate_ranking_metrics(prediction_matrix, pos_label_idxs, metrics=["ndcg@3"])
        assert results["ndcg@3"] == pytest.approx(1.0, abs=1e-4)

    def test_k_greater_than_n_targets(self):
        """K > n_targets should work and equal ndcg over the full list."""
        results_large_k = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg@10"]
        )
        results_full = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg@5"]
        )
        assert results_large_k["ndcg@10"] == pytest.approx(results_full["ndcg@5"])

    def test_single_query_single_target(self):
        """1x1 matrix with the single item being relevant -> NDCG = 1.0."""
        prediction_matrix = np.array([[0.5]])
        pos_label_idxs = [[0]]
        results = calculate_ranking_metrics(prediction_matrix, pos_label_idxs, metrics=["ndcg@1"])
        assert results["ndcg@1"] == pytest.approx(1.0, abs=1e-4)


class TestNDCGInputTypes:
    """Test NDCG with different input types."""

    def test_torch_tensor_input(self):
        """torch.Tensor input should produce the same result as numpy."""
        results_np = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg@3"]
        )
        results_torch = calculate_ranking_metrics(
            torch.tensor(PREDICTION_MATRIX), POS_LABEL_IDXS, metrics=["ndcg@3"]
        )
        assert results_torch["ndcg@3"] == pytest.approx(results_np["ndcg@3"])


class TestRankingMetricsGeneral:
    """General tests for the ranking metrics module."""

    def test_unknown_metric_raises(self):
        """Unknown metric names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ranking metric"):
            calculate_ranking_metrics(PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["nonexistent"])

    def test_existing_metrics_still_work(self):
        """Smoke test: all existing metrics return float values without error."""
        results = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["map", "mrr", "recall@3", "hit@3", "rp@3"],
        )
        for metric_name in ["map", "mrr", "recall@3", "hit@3", "rp@3"]:
            assert isinstance(results[metric_name], float)
            assert 0.0 <= results[metric_name] <= 1.0


class TestNDCGGradedRelevance:
    """nDCG with graded relevance (pos_label_relevance not None)."""

    def test_explicit_relevance_one_matches_binary(self):
        """Passing all-1.0 relevance must match the binary fallback exactly."""
        binary = calculate_ranking_metrics(PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["ndcg@3"])
        graded = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["ndcg@3"],
            pos_label_relevance=[[1.0, 1.0], [1.0, 1.0]],
        )
        assert graded["ndcg@3"] == pytest.approx(binary["ndcg@3"])

    def test_higher_grade_first_outscores_lower_grade_first(self):
        """Ranking the higher-graded positive first must beat the reverse."""
        preds_good = _prediction_matrix_from_order([0, 1, 2, 3], n_targets=4)
        preds_bad = _prediction_matrix_from_order([1, 0, 2, 3], n_targets=4)
        good = calculate_ranking_metrics(
            prediction_matrix=preds_good,
            pos_label_idxs=[[0, 1]],
            pos_label_relevance=[[3.0, 1.0]],
            metrics=["ndcg@4"],
        )
        bad = calculate_ranking_metrics(
            prediction_matrix=preds_bad,
            pos_label_idxs=[[0, 1]],
            pos_label_relevance=[[3.0, 1.0]],
            metrics=["ndcg@4"],
        )
        assert good["ndcg@4"] == pytest.approx(1.0)
        assert bad["ndcg@4"] < good["ndcg@4"]

    def test_graded_perfect_ranking_returns_one(self):
        """Ideal ordering of grades [3, 2, 1] gives nDCG = 1."""
        preds = _prediction_matrix_from_order([0, 1, 2, 3], n_targets=4)
        result = calculate_ranking_metrics(
            prediction_matrix=preds,
            pos_label_idxs=[[0, 1, 2]],
            pos_label_relevance=[[3.0, 2.0, 1.0]],
            metrics=["ndcg@4"],
        )
        assert result["ndcg@4"] == pytest.approx(1.0)

    def test_graded_known_value(self):
        """Hand-computed value with grades 2 and 1 over 3 targets.

        Positives: idx 0 grade 2, idx 2 grade 1. Ranking [0, 1, 2].
        gains @ ranks 0,1,2 = (2^2-1, 0, 2^1-1) = (3, 0, 1)
        discounts = 1/log2(2), 1/log2(3), 1/log2(4)
        DCG = 3 + 0 + 0.5 = 3.5
        Ideal grades [2, 1] -> gains (3, 1), discounts (1, 1/log2(3))
        IDCG = 3 + 1/log2(3)
        """
        preds = _prediction_matrix_from_order([0, 1, 2], n_targets=3)
        result = calculate_ranking_metrics(
            prediction_matrix=preds,
            pos_label_idxs=[[0, 2]],
            pos_label_relevance=[[2.0, 1.0]],
            metrics=["ndcg@3"],
        )
        dcg = 3.0 + 1.0 / math.log2(4)
        idcg = 3.0 + 1.0 / math.log2(3)
        assert result["ndcg@3"] == pytest.approx(dcg / idcg)

    def test_relevance_ignored_by_binary_metrics(self):
        """Binary metrics (map, mrr, ...) ignore pos_label_relevance entirely."""
        without = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["map", "mrr", "recall@3"]
        )
        with_rel = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["map", "mrr", "recall@3"],
            pos_label_relevance=[[3.0, 1.0], [2.0, 2.0]],
        )
        for metric in ["map", "mrr", "recall@3"]:
            assert with_rel[metric] == pytest.approx(without[metric])

    def test_misaligned_relevance_raises(self):
        """A relevance list whose lengths don't match pos_label_idxs raises."""
        with pytest.raises(ValueError, match="zip"):
            calculate_ranking_metrics(
                PREDICTION_MATRIX,
                POS_LABEL_IDXS,
                metrics=["ndcg@3"],
                pos_label_relevance=[[1.0], [1.0, 1.0]],
            )


class TestBinaryRelevanceThreshold:
    """Threshold filters the binary positive set without affecting nDCG."""

    def test_default_threshold_keeps_all_listed_positives(self):
        """Default 1e-9 threshold passes any non-zero grade — same as no relevance."""
        without_rel = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["map", "mrr", "recall@3"]
        )
        with_rel = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["map", "mrr", "recall@3"],
            pos_label_relevance=[[3.0, 1.0], [2.0, 2.0]],
        )
        for metric in ["map", "mrr", "recall@3"]:
            assert with_rel[metric] == pytest.approx(without_rel[metric])

    def test_threshold_drops_subthreshold_positives_for_binary(self):
        """A threshold of 2 drops grade-1 items from the binary positive set."""
        # Query 0 sorted: [0, 2, 4, 3, 1]. Positives: {2 (grade 3), 4 (grade 1)}.
        # With threshold=2, only idx 2 stays positive for binary metrics.
        # MRR: rank of first surviving positive (idx 2) is 2 -> 0.5
        # Query 1 sorted: [3, 1, 2, 0, 4]. Positives: {1 (grade 3), 3 (grade 1)}.
        # With threshold=2, only idx 1 stays. Rank of idx 1 = 2 -> 0.5
        result = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["mrr"],
            pos_label_relevance=[[3.0, 1.0], [3.0, 1.0]],
            binary_relevance_threshold=2.0,
        )
        assert result["mrr"] == pytest.approx(0.5)

    def test_threshold_does_not_affect_ndcg(self):
        """NDCG always sees the full graded list regardless of threshold."""
        relevance = [[3.0, 1.0], [3.0, 1.0]]
        low = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["ndcg@5"],
            pos_label_relevance=relevance,
            binary_relevance_threshold=1e-9,
        )
        high = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["ndcg@5"],
            pos_label_relevance=relevance,
            binary_relevance_threshold=2.0,
        )
        assert low["ndcg@5"] == pytest.approx(high["ndcg@5"])

    def test_threshold_ignored_when_relevance_is_none(self):
        """Threshold is a no-op when no graded relevance is supplied."""
        ref = calculate_ranking_metrics(
            PREDICTION_MATRIX, POS_LABEL_IDXS, metrics=["map", "mrr", "recall@3"]
        )
        with_threshold = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["map", "mrr", "recall@3"],
            binary_relevance_threshold=99.0,  # would drop everything if it applied
        )
        for metric in ["map", "mrr", "recall@3"]:
            assert with_threshold[metric] == pytest.approx(ref[metric])

    def test_threshold_above_all_grades_empties_binary_set(self):
        """A threshold above every grade leaves no binary positives -> 0.0."""
        result = calculate_ranking_metrics(
            PREDICTION_MATRIX,
            POS_LABEL_IDXS,
            metrics=["map", "mrr", "recall@3"],
            pos_label_relevance=[[3.0, 1.0], [3.0, 1.0]],
            binary_relevance_threshold=10.0,
        )
        for metric in ["map", "mrr", "recall@3"]:
            assert result[metric] == 0.0
