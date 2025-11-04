"""Ranking metrics implementation."""

from collections.abc import Sequence

import numpy as np
import torch


def calculate_ranking_metrics(
    prediction_matrix: torch.Tensor | np.ndarray,
    pos_label_idxs: list[list[int]],
    metrics: Sequence[str] = (
        "map",
        "rp@10",
    ),
) -> dict[str, float]:
    """
    Calculate ranking metrics for evaluation.

    Args:
        prediction_matrix: Similarity/prediction matrix of shape (n_queries, n_targets)
        pos_label_idxs: List of lists containing positive label indices for each query
        metrics: List of metric names to compute

    Returns
    -------
        Dictionary mapping metric names to values
    """
    # Convert to numpy if needed
    if isinstance(prediction_matrix, torch.Tensor):
        prediction_matrix = prediction_matrix.cpu().numpy()

    # Sort indices by prediction scores (descending)
    sorted_indices = np.argsort(-prediction_matrix, axis=1)

    results = {}

    def _metric_k_split(metric: str) -> tuple[str, int | None]:
        """Split metric name into base metric and k."""
        assert len(metric.strip()) == len(metric), f"Metric must not contain whitespace: '{metric}'"

        metric_parts = metric.split("@")
        assert len(metric_parts) <= 2, f"Metric must be in the format 'metric@k': '{metric}'"

        base_metric = metric_parts[0]
        k = int(metric_parts[1]) if len(metric_parts) == 2 else None
        return base_metric, k

    for metric in metrics:
        base_metric, k = _metric_k_split(metric)

        if metric == "map":
            results[metric] = _calculate_map(sorted_indices, pos_label_idxs)

        elif base_metric == "rp":
            assert k is not None, "k must be provided for rp@k metrics"
            results[metric] = _calculate_rp_at_k(sorted_indices, pos_label_idxs, k)

        elif metric == "mrr":
            results[metric] = _calculate_mrr(sorted_indices, pos_label_idxs)

        elif base_metric == "recall":
            assert k is not None, "k must be provided for recall@k metrics"
            results[metric] = _calculate_recall_at_k(sorted_indices, pos_label_idxs, k)

        elif base_metric == "hit":
            assert k is not None, "k must be provided for hit@k metrics"
            results[metric] = _calculate_hit_at_k(sorted_indices, pos_label_idxs, k)

        else:
            raise ValueError(f"Unknown ranking metric '{metric}'")

    return results


def _calculate_map(sorted_indices: np.ndarray, pos_label_idxs: list[list[int]]) -> float:
    """Calculate Mean Average Precision."""
    ap_scores = []

    for i, pos_labels in enumerate(pos_label_idxs):
        if len(pos_labels) == 0:
            continue

        # Create binary relevance vector
        relevance = np.zeros(len(sorted_indices[i]))
        for pos_idx in pos_labels:
            if pos_idx < len(relevance):
                relevance[pos_idx] = 1

        # Reorder relevance by predicted ranking
        ranked_relevance = relevance[sorted_indices[i]]

        # Calculate AP for this query
        ap = 0.0
        num_relevant = 0

        for k in range(len(ranked_relevance)):
            if ranked_relevance[k] == 1:
                num_relevant += 1
                ap += num_relevant / (k + 1)

        if num_relevant > 0:
            ap /= num_relevant
            ap_scores.append(ap)

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def _calculate_mrr(sorted_indices: np.ndarray, pos_label_idxs: list[list[int]]) -> float:
    """Calculate Mean Reciprocal Rank."""
    rr_scores = []

    for i, pos_labels in enumerate(pos_label_idxs):
        if len(pos_labels) == 0:
            continue

        # Find rank of first relevant item
        first_relevant_rank = None
        for rank, idx in enumerate(sorted_indices[i]):
            if idx in pos_labels:
                first_relevant_rank = rank + 1  # 1-indexed
                break

        if first_relevant_rank is not None:
            rr_scores.append(1.0 / first_relevant_rank)
        else:
            rr_scores.append(0.0)

    return float(np.mean(rr_scores)) if rr_scores else 0.0


def _calculate_recall_at_k(
    sorted_indices: np.ndarray, pos_label_idxs: list[list[int]], k: int
) -> float:
    """Calculate Recall@K."""
    recall_scores = []

    for i, pos_labels in enumerate(pos_label_idxs):
        if len(pos_labels) == 0:
            continue

        # Get top-k predictions
        top_k = set(sorted_indices[i][:k])

        # Count relevant items in top-k
        relevant_in_top_k = len(set(pos_labels) & top_k)

        # Recall = relevant_in_top_k / total_relevant
        recall = relevant_in_top_k / len(pos_labels)
        recall_scores.append(recall)

    return float(np.mean(recall_scores)) if recall_scores else 0.0


def _calculate_hit_at_k(
    sorted_indices: np.ndarray, pos_label_idxs: list[list[int]], k: int
) -> float:
    """Calculate Hit@K (Hit Rate)."""
    hit_scores = []

    for i, pos_labels in enumerate(pos_label_idxs):
        if len(pos_labels) == 0:
            continue

        # Get top-k predictions
        top_k = set(sorted_indices[i][:k])

        # Check if any relevant item is in top-k
        hit = 1.0 if len(set(pos_labels) & top_k) > 0 else 0.0
        hit_scores.append(hit)

    return float(np.mean(hit_scores)) if hit_scores else 0.0


def _calculate_rp_at_k(
    sorted_indices: np.ndarray, pos_label_idxs: list[list[int]], k: int
) -> float:
    """Calculate R-Precision@K."""
    rp_scores = []

    for i, pos_labels in enumerate(pos_label_idxs):
        r = len(pos_labels)
        if r == 0:
            continue  # Skip queries with no relevant documents

        # Get top-k predictions
        top_k = set(sorted_indices[i][:k])

        # Count relevant items in top-k
        relevant_in_top_k = len(set(pos_labels) & top_k)

        # R-precision@k = relevant_in_top_k / min(k, r)
        rp = relevant_in_top_k / min(k, r)
        rp_scores.append(rp)

    return float(np.mean(rp_scores)) if rp_scores else 0.0
