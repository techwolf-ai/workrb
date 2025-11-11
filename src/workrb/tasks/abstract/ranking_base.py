"""Ranking task implementation."""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING

import torch

from workrb.metrics.ranking import calculate_ranking_metrics
from workrb.tasks.abstract.base import BaseTaskGroup, DatasetSplit, Language, Task, TaskType
from workrb.types import ModelInputType

if TYPE_CHECKING:
    from workrb.models.base import ModelInterface


class RankingTaskGroup(BaseTaskGroup, str, Enum):
    _prefix = "rank_"

    JOB_NORMALIZATION = f"{_prefix}job_normalization"
    JOB2SKILL = f"{_prefix}job2skill"
    SKILL2JOB = f"{_prefix}skill2job"
    SKILL_NORMALIZATION = f"{_prefix}skill_normalization"
    SKILL_EXTRACTION = f"{_prefix}skill_extraction"
    SKILLSIM = f"{_prefix}skillsim"


class RankingDataset:
    """Structure for monolingualranking datasets."""

    def __init__(
        self,
        query_texts: list[str],
        target_indices: list[list[int]],
        target_space: list[str],
        language: Language,
    ):
        """Initialize ranking dataset with validation.

        Args:
            query: List of query strings
            target_label: List of lists containing indices into the target vocabulary
            target: List of target vocabulary strings
        """
        self.query_texts = self._postprocess_texts(query_texts)
        self.target_indices = self._postprocess_indices(target_indices)
        self.target_space = self._postprocess_texts(target_space)
        self.language = language
        self.validate_dataset()

    def validate_dataset(
        self,
        allow_duplicate_queries: bool = True,
        allow_duplicate_targets: bool = False,
    ):
        """Check the dataset."""
        if not allow_duplicate_queries:
            queries_non_unique = [
                query_text for query_text, cnt in Counter(self.query_texts).items() if cnt > 1
            ]
            assert len(queries_non_unique) == 0, (
                f"Query texts must be unique. Query texts appearing multiple times: {queries_non_unique} "
            )

        if not allow_duplicate_targets:
            targets_non_unique = [
                target_text for target_text, cnt in Counter(self.target_space).items() if cnt > 1
            ]
            assert len(targets_non_unique) == 0, (
                f"Target texts must be unique. Target texts appearing multiple times: {targets_non_unique} "
            )

        # Check no target_indices outside of target_space or non-int
        for idx_list in self.target_indices:
            for idx in idx_list:
                assert idx < len(self.target_space), (
                    f"Target index {idx} is not in target space {self.target_space}"
                )
                assert isinstance(idx, int), f"Target index {idx} is not an integer"

    def _postprocess_indices(self, indices: list[list[int]]) -> list[list[int]]:
        """Postprocess indices."""
        # Remove duplicates in target_label
        indices = [list(set(label_list)) for label_list in indices]
        return indices

    def _postprocess_texts(self, texts: list[str]) -> list[str]:
        """Postprocess texts."""
        # Remove whitespaces
        texts = [text.strip() for text in texts]
        return texts


class RankingTask(Task):
    """
    Abstract base class for ranking tasks.

    Supports both legacy ModelInterface and new ESCO-based approach.
    New tasks should implement load_val() and load_test() methods.
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.RANKING

    @property
    def default_metrics(self) -> list[str]:
        return ["map", "rp@10", "mrr"]

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize ranking task.

        Args:
            mode: Evaluation mode ("test" or "val")
            language: Language code
            **kwargs: Additional arguments for legacy compatibility
        """
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def query_input_type(self) -> ModelInputType:
        """Input type for query texts in the ranking task."""

    @property
    @abstractmethod
    def target_input_type(self) -> ModelInputType:
        """Input type for target texts in the ranking task."""

    @abstractmethod
    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """Load dataset for a specific language."""

    def get_size_oneliner(self, language: Language) -> str:
        """Get dataset summary to display for progress."""
        return f"{len(self.lang_datasets[language].query_texts)} queries x {len(self.lang_datasets[language].target_space)} targets"

    def evaluate(
        self,
        model: ModelInterface,
        metrics: list[str] | None = None,
        language: Language = Language.EN,
    ) -> dict[str, float]:
        """
        Evaluate the model on this ranking task.

        Args:
            model: Model implementing ModelInterface (must have compute_rankings method)
            metrics: List of metrics to compute. If None, uses default_metrics
            language: Language code for evaluation

        Returns
        -------
            Dictionary containing metric scores and evaluation metadata
        """
        if metrics is None:
            metrics = self.default_metrics

        # Use new dataset if available
        dataset = self.lang_datasets[language]
        queries = dataset.query_texts
        targets = dataset.target_space
        labels = dataset.target_indices

        # Get model predictions (similarity matrix)
        prediction_matrix = model.compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=self.query_input_type,
            target_input_type=self.target_input_type,
        )

        # Convert to numpy if needed
        if isinstance(prediction_matrix, torch.Tensor):
            prediction_matrix = prediction_matrix.cpu().numpy()

        metric_results = calculate_ranking_metrics(
            prediction_matrix=prediction_matrix, pos_label_idxs=labels, metrics=metrics
        )

        return metric_results
