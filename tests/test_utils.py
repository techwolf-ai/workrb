"""
Test utilities for WorkRB tests.

This module provides common utilities for testing, including:
- ToyTaskMixin: Reduces dataset size for fast testing
- Helper functions for creating toy tasks
"""

from typing import Any

from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language, Task, TaskType
from workrb.tasks.abstract.classification_base import ClassificationDataset, ClassificationTask
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup


class GeneralRankingTestTask(Task):
    """General test task."""

    def name(self):
        return "General Test Task"

    def description(self):
        return "General test task"

    @property
    def supported_query_languages(self):
        return [Language.EN, Language.DE]

    @property
    def supported_target_languages(self):
        return [Language.EN, Language.DE]

    @property
    def task_group(self) -> RankingTaskGroup:
        return RankingTaskGroup.JOB_NORMALIZATION

    @property
    def label_type(self) -> LabelType:
        return LabelType.SINGLE_LABEL

    @property
    def default_metrics(self) -> list[str]:
        return ["map"]

    def load_monolingual_data(self, language: Language, split: DatasetSplit) -> Any:
        return {}

    def evaluate(self, model, metrics=None, language: Language = Language.EN) -> dict[str, float]:
        return {}


class ToyTaskMixin:
    """
    Mixin to limit dataset size for toy benchmarks.

    This mixin overrides load_monolingual_data to return a limited subset
    of the full dataset, making tests run faster while still exercising
    the complete pipeline.
    """

    max_queries: int = 2
    max_targets: int = 3

    def _limit_dataset(self, dataset: RankingDataset) -> RankingDataset:
        """
        Limit the dataset size to toy proportions.

        Strategy:
        1. Take first N queries
        2. Collect all unique target indices referenced by those queries
        3. Limit to M targets if we have more
        4. Build new target space with only those targets
        5. Remap indices

        This ensures all queries have valid labels.

        Args:
            dataset: Full RankingDataset from parent class

        Returns
        -------
            Limited RankingDataset with fewer queries and targets
        """
        # Step 1: Limit queries
        limited_queries = dataset.query_texts[: self.max_queries]
        limited_target_indices = dataset.target_indices[: self.max_queries]

        # Step 2: Collect all unique target indices referenced by those queries
        all_referenced_targets = set()
        for label_list in limited_target_indices:
            all_referenced_targets.update(label_list)

        # Convert to sorted list for consistent ordering
        referenced_targets_list = sorted(all_referenced_targets)

        # Step 3: Limit to max_targets if we have more
        if len(referenced_targets_list) > self.max_targets:
            referenced_targets_list = referenced_targets_list[: self.max_targets]

        # Step 4: Build new target space with only referenced targets
        limited_target_space = [dataset.target_space[idx] for idx in referenced_targets_list]

        # Step 5: Create mapping from old indices to new indices
        old_to_new_idx = {
            old_idx: new_idx for new_idx, old_idx in enumerate(referenced_targets_list)
        }

        # Step 6: Remap label indices to new target space
        remapped_indices = []
        filtered_queries = []

        for query, label_list in zip(limited_queries, limited_target_indices, strict=False):
            # Remap labels - only keep those in our limited target space
            new_labels = [old_to_new_idx[idx] for idx in label_list if idx in old_to_new_idx]

            # Only keep queries that have at least one valid label
            if new_labels:
                remapped_indices.append(new_labels)
                filtered_queries.append(query)

        return RankingDataset(
            query_texts=filtered_queries,
            target_indices=remapped_indices,
            target_space=limited_target_space,
            language=dataset.language,
        )


class ToyClassificationTaskMixin:
    """
    Mixin to limit classification dataset size for toy benchmarks.

    This mixin overrides load_monolingual_data to return a limited subset
    of the full classification dataset, making tests run faster.
    """

    max_samples: int = 10  # Maximum number of samples to include

    def _limit_classification_dataset(
        self, dataset: ClassificationDataset
    ) -> ClassificationDataset:
        """
        Limit the classification dataset size to toy proportions.

        Strategy:
        1. Take first N samples
        2. Keep label space unchanged

        Args:
            dataset: Full ClassificationDataset from parent class

        Returns
        -------
            Limited ClassificationDataset with fewer samples
        """
        # Limit to max_samples
        limited_texts = dataset.texts[: self.max_samples]
        limited_labels = dataset.labels[: self.max_samples]

        return ClassificationDataset(
            texts=limited_texts,
            labels=limited_labels,
            label_space=dataset.label_space,  # Keep full label space
            language=dataset.language,
        )


def create_toy_task_class(
    base_task_class: type[Task],
) -> type[Task]:
    """
    Dynamically create a toy version of a task class.

    Args:
        base_task_class: The original task class from the registry

    Returns
    -------
        A new class that inherits from appropriate ToyMixin and the base class
    """
    # Determine if it's a ranking or classification task
    if issubclass(base_task_class, RankingTask):

        class ToyRankingTask(ToyTaskMixin, base_task_class):
            """Dynamically created toy ranking task."""

            def load_monolingual_data(
                self, split: DatasetSplit, language: Language
            ) -> RankingDataset:
                full_dataset = super().load_monolingual_data(split=split, language=language)
                return self._limit_dataset(full_dataset)

        return_cls = ToyRankingTask

    elif issubclass(base_task_class, ClassificationTask):

        class ToyClassificationTask(ToyClassificationTaskMixin, base_task_class):
            """Dynamically created toy classification task."""

            def load_monolingual_data(
                self, split: DatasetSplit, language: Language
            ) -> ClassificationDataset:
                full_dataset = super().load_monolingual_data(split=split, language=language)
                return self._limit_classification_dataset(full_dataset)

        return_cls = ToyClassificationTask

    else:
        raise ValueError(
            f"Task class {base_task_class} is not in a supported task type: {list(TaskType)}"
        )

    # Set a meaningful name for debugging
    return_cls.__name__ = f"Toy{base_task_class.__name__}"
    return_cls.__qualname__ = f"Toy{base_task_class.__name__}"

    return return_cls
