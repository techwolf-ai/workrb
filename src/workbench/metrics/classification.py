"""Classification metrics implementation."""

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    predictions: torch.Tensor | np.ndarray | list[list[float]],
    true_labels: torch.Tensor | np.ndarray | list[list[int]],
    threshold_mode: Literal["argmax", "threshold"] = "argmax",
    threshold: float | None = None,
    metrics: Sequence[str] = (
        "f1_macro",
        "precision_macro",
        "recall_macro",
    ),
) -> dict[str, float]:
    """
    Calculate classification metrics for multi-class, binary, and multilabel classification.

    All inputs are strictly treated as 2D arrays internally for consistent processing.

    Classification Modes:
        - **threshold_mode="argmax"**: Multi-class or binary classification
          Uses argmax to select the predicted class from probabilities
        - **threshold_mode="threshold"**: Multilabel classification
          Applies threshold to each label independently

    Args:
        predictions: Model prediction scores/probabilities as 2D array of shape (n_samples, n_labels).
            Can be provided as torch.Tensor, numpy array, or list of lists.
            For either of binary, multi-class, multilabel: each row contains class probabilities
        true_labels: Ground truth labels as 2D array of shape (n_samples, n_labels).
            Can be provided as torch.Tensor, numpy array, or list of lists.
            For multi-class: one-hot encoded vectors (single 1 per row)
            For multilabel: binary indicator matrix (multiple 1s per row possible)
        threshold_mode: How to convert predictions to predicted classes:
            - "argmax": Use argmax for multi-class (selects highest scoring class)
            - "threshold": Use threshold for multilabel (predicts 1 if score >= threshold)
        threshold: Threshold value for binarizing predictions when threshold_mode="threshold".
            Default is None. Should be provided if threshold_mode="threshold".
        metrics: Sequence of metric names to compute (threshold-dependent metrics). Supported values:
            - "f1_macro": Macro-averaged F1 score
            - "f1_micro": Micro-averaged F1 score
            - "f1_weighted": Weighted F1 score
            - "f1_samples": Per-sample F1 score (multilabel only)
            - "precision_macro", "precision_micro", "precision_weighted": Precision variants
            - "recall_macro", "recall_micro", "recall_weighted": Recall variants
            - "accuracy": Overall accuracy

        Note:
            Threshold-independent metrics like ROC AUC are exposed via
            calculate_classification_roc_auc(). If "roc_auc" is included
            here, it will be computed by delegating to that function for
            backward compatibility.

    Returns
    -------
        Dictionary mapping metric names to float values.

    Examples
    --------
        Multi-class classification:
            >>> predictions = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
            >>> true_labels = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot encoded
            >>> result = calculate_classification_metrics(
            ...     predictions, true_labels, threshold_mode="argmax"
            ... )

        Binary classification:
            >>> predictions = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> true_labels = np.array([[0, 1], [1, 0]])  # One-hot encoded
            >>> result = calculate_classification_metrics(
            ...     predictions, true_labels, threshold_mode="argmax"
            ... )

        Multilabel classification:
            >>> predictions = np.array([[0.9, 0.8, 0.1], [0.3, 0.6, 0.7]])
            >>> true_labels = np.array([[1, 1, 0], [0, 1, 1]])  # Binary indicator matrix
            >>> result = calculate_classification_metrics(
            ...     predictions, true_labels, threshold_mode="threshold", threshold=0.5
            ... )
    """
    # Convert inputs to 2D numpy arrays
    predictions = _to_2d_numpy(predictions)
    true_labels = _to_2d_numpy(true_labels)

    # Validate shapes match
    if predictions.shape != true_labels.shape:
        raise ValueError(
            f"Shape mismatch: predictions shape {predictions.shape} != "
            f"true_labels shape {true_labels.shape}"
        )

    # Compute predicted classes as a 2D indicator matrix for all modes
    if threshold_mode == "argmax":  # One-hot of argmax
        argmax_indices = np.argmax(predictions, axis=1)
        predicted_classes_for_metrics = np.zeros_like(true_labels, dtype=int)
        predicted_classes_for_metrics[np.arange(len(argmax_indices)), argmax_indices] = 1
        true_labels_for_metrics = true_labels

    elif threshold_mode == "threshold":  # Threshold per label
        assert threshold is not None, "Threshold must be provided if threshold_mode='threshold'"
        predicted_classes_for_metrics = (predictions >= threshold).astype(int)
        true_labels_for_metrics = true_labels

    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    results = {}

    # Calculate each requested metric
    for metric in metrics:
        if metric == "accuracy":
            results[metric] = _calculate_accuracy(
                true_labels_for_metrics, predicted_classes_for_metrics
            )

        elif metric.startswith("f1"):
            average = _parse_averaging_mode(metric, "f1")
            results[metric] = f1_score(
                true_labels_for_metrics,
                predicted_classes_for_metrics,
                average=average,
                zero_division=0,
            )

        elif metric.startswith("precision"):
            average = _parse_averaging_mode(metric, "precision")
            results[metric] = precision_score(
                true_labels_for_metrics,
                predicted_classes_for_metrics,
                average=average,
                zero_division=0,
            )

        elif metric.startswith("recall"):
            average = _parse_averaging_mode(metric, "recall")
            results[metric] = recall_score(
                true_labels_for_metrics,
                predicted_classes_for_metrics,
                average=average,
                zero_division=0,
            )

        elif metric.startswith("roc_auc"):
            # Backward-compat: delegate to dedicated ROC AUC function
            results[metric] = calculate_classification_roc_auc(
                predictions=predictions,
                true_labels=true_labels,
                threshold_mode=threshold_mode,
                average="macro",
            )
        elif metric == "roc_auc_micro":
            results[metric] = calculate_classification_roc_auc(
                predictions=predictions,
                true_labels=true_labels,
                threshold_mode=threshold_mode,
                average="micro",
            )

        else:
            logger.warning(f"Unknown classification metric '{metric}'")

    return results


def calculate_classification_roc_auc(
    predictions: torch.Tensor | np.ndarray | list[list[float]],
    true_labels: torch.Tensor | np.ndarray | list[list[int]],
    threshold_mode: Literal["argmax", "threshold"] = "argmax",
    average: Literal["macro", "micro"] = "macro",
) -> float:
    """
    Calculate ROC AUC for classification using 2D inputs.

    Args:
        predictions: Raw prediction scores as 2D array (n_samples, n_labels)
        true_labels: Ground truth as 2D indicator array (n_samples, n_labels)
        threshold_mode: "argmax" for (multi-class/binary) or "threshold" for multilabel
        average: Averaging strategy ("macro" or "micro")

    Returns
    -------
        ROC AUC score
    """
    predictions_2d = _to_2d_numpy(predictions)
    true_labels_2d = _to_2d_numpy(true_labels)
    return _calculate_roc_auc(true_labels_2d, predictions_2d, threshold_mode, average=average)


def optimize_threshold(
    predictions: np.ndarray | list[list[float]],
    labels: np.ndarray | list[list[int]],
    metric: str = "f1_macro",
    thresholds: np.ndarray | None = None,
) -> tuple[float, list[float], list[float]]:
    """
    Optimize classification threshold for multilabel predictions.

    This function searches for the optimal threshold by evaluating a range of
    threshold values and selecting the one that maximizes the specified metric.

    A single threshold is optimized that applies to all labels simultaneously.

    Args:
        predictions: Prediction scores/probabilities as 2D array of shape (n_samples, n_labels).
        labels: True labels as 2D binary indicator matrix of shape (n_samples, n_labels).
        metric: Metric to optimize ('f1_macro', 'f1_micro', 'accuracy', etc.)
        thresholds: Array of thresholds to try. If None, uses default range [0.1, 0.2, ..., 0.9].

    Returns
    -------
        Tuple of (best_threshold, scores_for_thresholds, thresholds_tested)
            - best_threshold: The optimal threshold value
            - scores_for_thresholds: List of metric scores for each threshold
            - thresholds_tested: List of threshold values that were tested

    Examples
    --------
        Multilabel classification:
            >>> predictions = np.array([[0.9, 0.8, 0.1], [0.3, 0.6, 0.7]])
            >>> labels = np.array([[1, 1, 0], [0, 1, 1]])
            >>> best_th, scores, ths = optimize_threshold(predictions, labels, metric="f1_macro")
    """
    # Convert inputs to 2D numpy arrays
    predictions = _to_2d_numpy(predictions)
    labels = _to_2d_numpy(labels)

    # Default threshold range
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    scores = []
    for threshold in thresholds:
        result = calculate_classification_metrics(
            predictions=predictions,
            true_labels=labels,
            threshold_mode="threshold",
            threshold=threshold,
            metrics=[metric],
        )
        scores.append(result.get(metric, 0.0))

    # Find best threshold
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]

    return best_threshold, scores, thresholds.tolist()


def _to_2d_numpy(array: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """
    Convert input to 2D numpy array.

    Args:
        array: Input as torch.Tensor, numpy array, or list

    Returns
    -------
        2D numpy array
    """
    # Convert torch tensor to numpy
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()

    # Convert to numpy array if it's a list
    if isinstance(array, list):
        array = np.array(array)

    # Ensure 2D
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    return array


def _parse_averaging_mode(metric: str, base_metric: str) -> str:
    """
    Parse averaging mode from metric name.

    Args:
        metric: Full metric name (e.g., "f1_macro", "precision_weighted")
        base_metric: Base metric name (e.g., "f1", "precision")

    Returns
    -------
        Averaging mode (e.g., "macro", "micro", "weighted", "samples")
    """
    if metric == base_metric:  # default
        return "macro"

    # Extract suffix after base_metric_
    prefix = f"{base_metric}_"
    suffix = (
        metric[len(prefix) :] if metric.startswith(prefix) and len(metric) > len(prefix) else ""
    )
    return suffix


def _calculate_accuracy(true_labels: np.ndarray, predicted_classes: np.ndarray) -> float:
    """
    Calculate accuracy for classification.

    Args:
        true_labels: Ground truth labels
        predicted_classes: Predicted class labels
        is_multilabel: Whether this is multilabel classification

    Returns
    -------
        Accuracy score
    """
    # Use subset accuracy when labels are multi-hot; otherwise fallback to sklearn
    is_multilabel = (
        true_labels.ndim == 2 and true_labels.shape[1] > 1 and (true_labels.sum(axis=1) != 1).any()
    )
    if is_multilabel:
        return float(np.mean(np.all(predicted_classes == true_labels, axis=1)))
    # Collapse one-hot to 1D indices for multiclass/binary
    return accuracy_score(np.argmax(true_labels, axis=1), np.argmax(predicted_classes, axis=1))


def _calculate_roc_auc(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    threshold_mode: str,
    average: str = "macro",
) -> float:
    """
    Calculate ROC AUC score.

    For multi-class (threshold_mode="argmax"), converts to one-vs-rest.
    For multilabel (threshold_mode="threshold"), calculates per-label AUC.

    Args:
        true_labels: Ground truth labels as 2D array
        predictions: Raw prediction scores as 2D array
        threshold_mode: "argmax" for multiclass or "threshold" for multilabel
        average: Averaging strategy ("macro" or "micro")

    Returns
    -------
        ROC AUC score
    """
    if threshold_mode == "argmax":
        # Multi-class/binary: use sklearn's built-in handling
        # For binary (2 classes), we can pass 2D directly
        # For multi-class (>2 classes), sklearn will use one-vs-rest
        if true_labels.shape[1] == 2:
            # Binary classification: use the positive class probabilities
            return roc_auc_score(true_labels[:, 1], predictions[:, 1])

        # Multi-class: use one-vs-rest strategy
        true_class_indices = np.argmax(true_labels, axis=1)
        return roc_auc_score(true_class_indices, predictions, average=average, multi_class="ovr")

    # Multilabel: calculate per-label AUC
    return roc_auc_score(true_labels, predictions, average=average)
