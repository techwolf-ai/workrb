"""Classification task implementation."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Literal

from workrb.metrics.classification import calculate_classification_metrics, optimize_threshold
from workrb.tasks.abstract.base import (
    BaseTaskGroup,
    DatasetSplit,
    LabelType,
    Language,
    Task,
    TaskType,
)
from workrb.types import ModelInputType

if TYPE_CHECKING:
    import torch

    from workrb.models.base import ModelInterface

logger = logging.getLogger(__name__)


class ClassificationTaskGroup(BaseTaskGroup, str, Enum):
    """Classification task group enum."""

    _prefix = "cls_"

    JOB2SKILL = f"{_prefix}job2skill"


class ClassificationDataset:
    """Structure for classification datasets (both binary, multi-class, and multi-label)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[list[int]],
        label_space: list[str],
        language: Language,
    ):
        """Initialize classification dataset with validation.

        Args:
            texts: List of input text strings
            labels: List with list of class indices corresponding to each text.
                Contains just 1 item per list for single-label classification.
            label_space: List of class names/labels (e.g., ["skill1", "skill2", "skill3"])
            language: Language enum
        """
        self.texts = self._postprocess_texts(texts)
        self.labels = self._postprocess_labels(labels)
        self.label_space = self._postprocess_texts(label_space)
        self.language = language
        self.validate_dataset()

    def validate_dataset(
        self,
        allow_duplicate_texts: bool = True,
    ):
        """Check the dataset."""
        if not allow_duplicate_texts:
            texts_non_unique = [text for text, cnt in Counter(self.texts).items() if cnt > 1]
            assert len(texts_non_unique) == 0, (
                f"Texts must be unique. Texts appearing multiple times: {texts_non_unique} "
            )

        # Check all labels are valid indices into label_space
        label_set = set()
        for label_list in self.labels:
            assert isinstance(label_list, list), (
                f"Label list must be a list, got: {type(label_list)}, value: {label_list}"
            )
            label_set.update(label_list)
            for idx in label_list:
                assert idx < len(self.label_space), (
                    f"Label {idx} is not in label space (size {len(self.label_space)})"
                )
                assert isinstance(idx, int), f"Label {idx} is not an integer"

        # Check more than 1 class available
        if len(label_set) <= 1:
            raise ValueError(
                f"At least 2 classes are required for classification dataset, only got labels: {label_set}"
            )

        # Check lengths match
        assert len(self.texts) == len(self.labels), (
            f"Number of texts ({len(self.texts)}) != number of labels ({len(self.labels)})"
        )

    def _postprocess_labels(
        self, labels: list[int] | list[list[int]]
    ) -> list[int] | list[list[int]]:
        """Postprocess labels."""
        return labels

    def _postprocess_texts(self, texts: list[str]) -> list[str]:
        """Postprocess texts."""
        # Remove whitespaces
        texts = [text.strip() for text in texts]
        return texts

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the dataset."""
        return {
            "num_samples": len(self.texts),
            "num_classes": len(self.label_space),
        }

    def get_labels_as_indicator_matrix(self) -> list[list[int]]:
        """
        Convert labels to binary indicator matrix format for metrics computation.

        Returns
        -------
            Binary indicator matrix where each row corresponds to a sample and each column
            to a class. For single-label: one-hot encoded. For multi-label: multi-hot encoded.
        """
        num_classes = len(self.label_space)
        indicator_matrix: list[list[int]] = []

        for label in self.labels:
            row = [0] * num_classes
            if isinstance(label, list):
                # Multi-label: set all positive class indices to 1
                for idx in label:
                    row[idx] = 1
            else:
                # Single-label: one-hot encoding
                row[label] = 1
            indicator_matrix.append(row)

        return indicator_matrix


class ClassificationTask(Task):
    """
    Abstract base class for classification tasks.

    Supports both binary and multi-class classification.
    Tasks should implement load_monolingual_data() to return ClassificationDataset.
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    @property
    def default_metrics(self) -> list[str]:
        return ["f1_macro", "roc_auc", "precision_macro", "recall_macro"]

    @property
    @abstractmethod
    def threshold_mode(self) -> Literal["argmax", "threshold"]:
        """Threshold mode for classification."""

    @property
    @abstractmethod
    def best_threshold_on_val_data(self) -> bool:
        """Whether to optimize the threshold on validation data."""

    @property
    @abstractmethod
    def val_optim_metric(self) -> str:
        """Metric to optimize the threshold on validation data."""

    @property
    @abstractmethod
    def threshold(self) -> float | None:
        """Threshold to use for classification."""

    @abstractmethod
    def get_output_space_size(self, language: Language) -> int:
        """Number of output classes for this classification task."""

    @property
    @abstractmethod
    def input_type(self) -> ModelInputType:
        """Input type for texts in the classification task."""

    @abstractmethod
    def load_monolingual_data(
        self, split: DatasetSplit, language: Language
    ) -> ClassificationDataset:
        """Load dataset for a specific language."""

    def get_size_oneliner(self, language: Language) -> str:
        """Get dataset summary to display for progress."""
        dataset: ClassificationDataset = self.lang_datasets[language]
        return f"{len(dataset.texts)} samples, {len(dataset.label_space)} classes"

    def evaluate(
        self,
        model: ModelInterface,
        metrics: list[str] | None = None,
        language: Language = Language.EN,
    ) -> dict[str, float]:
        """
        Evaluate the model with threshold optimization.

        For binary classification, this method:
        1. Optimizes threshold on validation data
        2. Applies optimized threshold to test predictions
        3. Calculates metrics on test data

        Args:
            model: Model implementing classification interface
            metrics: List of metrics to compute
            language: Language code for evaluation

        Returns
        -------
            Dictionary containing metric scores and evaluation metadata
        """
        if metrics is None:
            metrics = self.default_metrics

        # Get evaluation dataset
        eval_dataset: ClassificationDataset = self.lang_datasets[language]

        # Validate model output if it has a fixed classification label space
        model_label_space = model.classification_label_space
        if model_label_space is not None:
            # Model has fixed label space (e.g., classification head)
            if len(model_label_space) != self.get_output_space_size(language):
                raise ValueError(
                    f"Model output size mismatch: model has {len(model_label_space)} outputs, "
                    f"but task requires {self.get_output_space_size(language)} outputs."
                )
            # Validate label order matches (critical for correct evaluation)
            self._validate_model_label_space(model_label_space, eval_dataset)

        best_threshold = (
            self.get_threshold_on_val_data(model, language)
            if self.best_threshold_on_val_data
            else self.threshold
        )

        # Evaluate on the actual dataset (test or val)
        eval_predictions: torch.Tensor = model.compute_classification(
            texts=eval_dataset.texts,
            targets=eval_dataset.label_space,
            input_type=self.input_type,
        )

        # Calculate classification metrics
        # Convert labels to binary indicator matrix format expected by metrics
        true_labels_matrix = eval_dataset.get_labels_as_indicator_matrix()
        metric_results = calculate_classification_metrics(
            predictions=eval_predictions,
            true_labels=true_labels_matrix,
            threshold=best_threshold,
            metrics=metrics,
            threshold_mode=self.threshold_mode,
        )
        return metric_results

    def _validate_model_label_space(
        self,
        model_label_space: list[str],
        eval_dataset: ClassificationDataset,
    ):
        """Validate model label space matches task label space."""
        # Model has explicit label space - validate it matches task
        if len(model_label_space) != len(eval_dataset.label_space):
            raise ValueError(
                f"Model label space size mismatch: model has {len(model_label_space)} labels "
                f"but task has {len(eval_dataset.label_space)} labels."
            )

        # Check if labels match exactly in order
        if model_label_space != eval_dataset.label_space:
            # Check if it's just ordering issue or missing labels
            model_labels_set = set(model_label_space)
            task_labels_set = set(eval_dataset.label_space)

            if model_labels_set != task_labels_set:
                missing_in_task = model_labels_set - task_labels_set
                missing_in_model = task_labels_set - model_labels_set
                error_msg = "Model and task have different label sets.\n"
                if missing_in_task:
                    error_msg += (
                        f"  Labels in model but not in task: {list(missing_in_task)[:5]}...\n"
                    )
                if missing_in_model:
                    error_msg += (
                        f"  Labels in task but not in model: {list(missing_in_model)[:5]}...\n"
                    )
                raise ValueError(error_msg)

            # Same labels, different order
            logger.warning(
                f"Model label order doesn't match task label order for {self.name}. "
                f"First mismatch at index 0: model='{model_label_space[0]}' vs task='{eval_dataset.label_space[0]}'. "
                f"Model must be retrained with task's label order or predictions must be reordered."
            )
            raise ValueError(
                "Model label order doesn't match task label order. "
                "This will cause incorrect evaluation results. "
                "The model must use the exact same label ordering as the task."
            )

    def get_threshold_on_val_data(self, model: ModelInterface, language: Language) -> float:
        """Get the best threshold on validation data."""
        # Step 1: Optimize threshold on validation data
        # Load validation data (even if we're evaluating on test)
        logger.info(f"Optimizing threshold on validation data for {language}...")
        val_dataset = self.load_monolingual_data(DatasetSplit.VAL, language)
        val_predictions = model.compute_classification(
            texts=val_dataset.texts,
            targets=val_dataset.label_space,
            input_type=self.input_type,
        )

        # Optimize threshold
        # Convert labels to binary indicator matrix format expected by optimize_threshold
        val_labels_matrix = val_dataset.get_labels_as_indicator_matrix()
        best_threshold, _, _ = optimize_threshold(
            predictions=val_predictions,
            labels=val_labels_matrix,
            metric=self.val_optim_metric,
        )
        return best_threshold


class MultiLabelClassificationTask(ClassificationTask):
    """
    Abstract base class for multi-label classification tasks, where samples can have multiple labels.
    """

    threshold_mode: Literal["argmax", "threshold"] = "threshold"

    def __init__(
        self,
        threshold: float | None = None,
        best_threshold_on_val_data: bool = True,
        val_optim_metric: str = "f1_macro",
        **kwargs,
    ):
        """Initialize classification task."""
        super().__init__(**kwargs)

        self._best_threshold_on_val_data: bool = best_threshold_on_val_data
        self._val_optim_metric: str = val_optim_metric
        self._threshold: float | None = threshold

        # Validate inputs
        if self.threshold is None:
            assert self.best_threshold_on_val_data, (
                "Threshold must be provided if best_threshold_on_val_data is False"
            )
        if self.best_threshold_on_val_data:
            assert self.threshold is None, (
                "Threshold must be None if best_threshold_on_val_data is True"
            )

    @property
    def best_threshold_on_val_data(self) -> bool:
        """Whether to optimize the threshold on validation data."""
        return self._best_threshold_on_val_data

    @property
    def val_optim_metric(self) -> str:
        """Metric to optimize the threshold on validation data."""
        return self._val_optim_metric

    @property
    def threshold(self) -> float | None:
        """Threshold to use for classification."""
        return self._threshold

    @property
    def label_type(self) -> LabelType:
        """Multi-label classification (multiple labels per sample)."""
        return LabelType.MULTI_LABEL


class MulticlassClassificationTask(ClassificationTask):
    """
    Abstract base class for multi-class classification tasks, where each sample has exactly one label.
    """

    threshold_mode: Literal["argmax", "threshold"] = "argmax"
    best_threshold_on_val_data: bool = False
    threshold: float | None = None

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize classification task.

        Args:
            **kwargs: Arguments passed to parent Task class (languages, split, etc.)
        """
        super().__init__(**kwargs)

    @property
    def label_type(self) -> LabelType:
        """Multi-class classification (single label per sample)."""
        return LabelType.SINGLE_LABEL
