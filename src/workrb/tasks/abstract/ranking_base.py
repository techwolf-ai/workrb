"""Ranking task implementation."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING

import torch

from workrb.metrics.ranking import calculate_ranking_metrics
from workrb.tasks.abstract.base import (
    BaseTaskGroup,
    DatasetConfigNotSupported,
    DatasetSplit,
    Task,
    TaskType,
)
from workrb.types import ModelInputType

if TYPE_CHECKING:
    from workrb.models.base import ModelInterface

logger = logging.getLogger(__name__)


class RankingTaskGroup(BaseTaskGroup, str, Enum):
    _prefix = "rank_"

    JOB_NORMALIZATION = f"{_prefix}job_normalization"
    JOB2SKILL = f"{_prefix}job2skill"
    SEMANTIC_SIMILARITY = f"{_prefix}semantic_similarity"
    SKILL2JOB = f"{_prefix}skill2job"
    SKILL_NORMALIZATION = f"{_prefix}skill_normalization"
    SKILL_EXTRACTION = f"{_prefix}skill_extraction"
    CANDIDATE_RANKING = f"{_prefix}candidate_ranking"


class DuplicateStrategy(str, Enum):
    """Strategy for handling duplicate queries or targets in a RankingDataset.

    ALLOW: Silently accept duplicates without any action.
    RAISE: Raise an error if duplicates are found.
    RESOLVE: Deterministically deduplicate. For targets, keep first occurrence and remap
        indices. For queries, merge target_indices via union for identical query texts.
    """

    ALLOW = "allow"
    RAISE = "raise"
    RESOLVE = "resolve"


class RankingDataset:
    """Structure for ranking datasets."""

    def __init__(
        self,
        query_texts: list[str],
        target_indices: list[list[int]],
        target_space: list[str],
        dataset_id: str,
        target_relevance: list[list[float]] | None = None,
        duplicate_query_strategy: DuplicateStrategy = DuplicateStrategy.RESOLVE,
        duplicate_target_strategy: DuplicateStrategy = DuplicateStrategy.RESOLVE,
    ):
        """Initialize ranking dataset with validation.

        Parameters
        ----------
        query_texts : list[str]
            List of query strings.
        target_indices : list[list[int]]
            List of lists containing indices into the target vocabulary. Items not
            listed for a query are treated as having relevance 0 (unjudged /
            irrelevant) by graded metrics.
        target_space : list[str]
            List of target vocabulary strings.
        dataset_id : str
            Unique identifier for this dataset.
        target_relevance : list[list[float]] or None, optional
            Optional graded relevance per positive, aligned 1-to-1 with
            ``target_indices``. When ``None``, every entry in ``target_indices`` is
            treated as binary relevance 1.0. Used by graded metrics such as
            ``ndcg@k``; binary metrics (``map``, ``mrr``, ``recall@k``, ``hit@k``,
            ``rp@k``) ignore this field. Values must be non-negative; the scale is
            up to the task (e.g. {1, 2, 3} or {0.0..1.0}).
        duplicate_query_strategy : DuplicateStrategy
            How to handle duplicate query texts. ALLOW silently accepts them,
            RAISE raises on duplicates, RESOLVE merges their target_indices via union.
        duplicate_target_strategy : DuplicateStrategy
            How to handle duplicate target texts. ALLOW silently accepts them,
            RAISE raises on duplicates, RESOLVE keeps first occurrence and remaps indices.
        """
        self.query_texts = self._postprocess_texts(query_texts)
        self.target_indices, self.target_relevance = self._postprocess_indices(
            target_indices, target_relevance
        )
        self.target_space = self._postprocess_texts(target_space)
        self.dataset_id = dataset_id

        # Resolve duplicates (targets first to remap indices, then queries to merge)
        if duplicate_target_strategy == DuplicateStrategy.RESOLVE:
            self._resolve_duplicate_targets()
        if duplicate_query_strategy == DuplicateStrategy.RESOLVE:
            self._resolve_duplicate_queries()

        self._validate_dataset(duplicate_query_strategy, duplicate_target_strategy)

    def _resolve_duplicate_targets(self) -> None:
        """Deduplicate target_space, keeping first occurrence and remapping indices."""
        seen: dict[str, int] = {}
        old_to_new: dict[int, int] = {}
        new_target_space: list[str] = []
        duplicates: list[str] = []

        for old_idx, text in enumerate(self.target_space):
            if text in seen:
                old_to_new[old_idx] = seen[text]
                duplicates.append(text)
            else:
                new_idx = len(new_target_space)
                seen[text] = new_idx
                old_to_new[old_idx] = new_idx
                new_target_space.append(text)

        if duplicates:
            logger.warning(
                "Resolved %d duplicate targets in dataset '%s': %s",
                len(duplicates),
                self.dataset_id,
                duplicates,
            )
            self.target_space = new_target_space
            if self.target_relevance is None:
                self.target_indices = [
                    sorted(set(old_to_new[idx] for idx in idx_list))
                    for idx_list in self.target_indices
                ]
            else:
                # Remap indices, dedup, and keep relevance from the first occurrence
                # of each remapped index so (idx, rel) pairs stay aligned.
                new_indices: list[list[int]] = []
                new_relevance: list[list[float]] = []
                for idx_list, rel_list in zip(
                    self.target_indices, self.target_relevance, strict=True
                ):
                    seen_idx: dict[int, float] = {}
                    for idx, rel in zip(idx_list, rel_list, strict=True):
                        new_idx = old_to_new[idx]
                        if new_idx not in seen_idx:
                            seen_idx[new_idx] = rel
                    sorted_pairs = sorted(seen_idx.items())
                    new_indices.append([idx for idx, _ in sorted_pairs])
                    new_relevance.append([rel for _, rel in sorted_pairs])
                self.target_indices = new_indices
                self.target_relevance = new_relevance

    def _resolve_duplicate_queries(self) -> None:
        """Deduplicate query_texts, merging target_indices via union.

        When ``target_relevance`` is set, the merge keeps the relevance from the
        first query occurrence for each index; relevance values from later
        duplicates of the same (query, index) pair are dropped.
        """
        seen: dict[str, int] = {}
        new_queries: list[str] = []
        new_pairs: list[dict[int, float]] = []
        duplicates: list[str] = []

        graded = self.target_relevance is not None
        relevance_iter = (
            self.target_relevance
            if graded
            else [[1.0] * len(idx_list) for idx_list in self.target_indices]
        )

        for query, idx_list, rel_list in zip(
            self.query_texts, self.target_indices, relevance_iter, strict=True
        ):
            if query in seen:
                pos = seen[query]
                for idx, rel in zip(idx_list, rel_list, strict=True):
                    if idx not in new_pairs[pos]:
                        new_pairs[pos][idx] = rel
                duplicates.append(query)
            else:
                seen[query] = len(new_queries)
                new_queries.append(query)
                new_pairs.append(dict(zip(idx_list, rel_list, strict=True)))

        if duplicates:
            logger.warning(
                "Resolved %d duplicate queries (merged target_indices) in dataset '%s': %s",
                len(duplicates),
                self.dataset_id,
                duplicates,
            )
            self.query_texts = new_queries
            new_indices: list[list[int]] = []
            new_relevance: list[list[float]] = []
            for pairs in new_pairs:
                sorted_pairs = sorted(pairs.items())
                new_indices.append([idx for idx, _ in sorted_pairs])
                new_relevance.append([rel for _, rel in sorted_pairs])
            self.target_indices = new_indices
            if graded:
                self.target_relevance = new_relevance

    def _validate_dataset(
        self,
        duplicate_query_strategy: DuplicateStrategy,
        duplicate_target_strategy: DuplicateStrategy,
    ) -> None:
        """Validate the dataset after construction and optional deduplication."""
        if len(self.query_texts) == 0:
            raise DatasetConfigNotSupported(
                f"Dataset '{self.dataset_id}' has 0 queries. "
                "This typically means that on dynamic dataset loading, no data is available for this language/configuration. "
            )
        if len(self.target_space) == 0:
            raise DatasetConfigNotSupported(
                f"Dataset '{self.dataset_id}' has 0 targets. "
                "This typically means that on dynamic dataset loading, no data is available for this language/configuration. "
            )

        if duplicate_query_strategy == DuplicateStrategy.RAISE:
            queries_non_unique = [
                query_text for query_text, cnt in Counter(self.query_texts).items() if cnt > 1
            ]
            assert len(queries_non_unique) == 0, (
                f"Query texts must be unique. Query texts appearing multiple times: {queries_non_unique} "
            )

        if duplicate_target_strategy == DuplicateStrategy.RAISE:
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

        # Check target_relevance alignment and non-negativity
        if self.target_relevance is not None:
            assert len(self.target_relevance) == len(self.target_indices), (
                f"target_relevance has {len(self.target_relevance)} queries but "
                f"target_indices has {len(self.target_indices)}"
            )
            for q_i, (rel_list, idx_list) in enumerate(
                zip(self.target_relevance, self.target_indices, strict=True)
            ):
                assert len(rel_list) == len(idx_list), (
                    f"target_relevance[{q_i}] has length {len(rel_list)} but "
                    f"target_indices[{q_i}] has length {len(idx_list)}"
                )
                for rel in rel_list:
                    assert rel >= 0, f"Negative relevance value {rel} at query {q_i}"

    def _postprocess_indices(
        self,
        indices: list[list[int]],
        relevance: list[list[float]] | None,
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        """Postprocess indices and aligned relevance, dropping duplicate indices.

        Indices are sorted; relevance is permuted in lockstep so each (idx, rel)
        pair stays aligned. When duplicate indices appear within a query, the
        relevance from the first occurrence is kept.
        """
        if relevance is None:
            return [sorted(set(label_list)) for label_list in indices], None

        assert len(relevance) == len(indices), (
            f"target_relevance has {len(relevance)} queries but target_indices has {len(indices)}"
        )
        deduped_indices: list[list[int]] = []
        deduped_relevance: list[list[float]] = []
        for idx_list, rel_list in zip(indices, relevance, strict=True):
            assert len(idx_list) == len(rel_list), (
                f"target_indices and target_relevance must align per query "
                f"(got {len(idx_list)} vs {len(rel_list)})"
            )
            seen: dict[int, float] = {}
            for idx, rel in zip(idx_list, rel_list, strict=True):
                if idx not in seen:
                    seen[idx] = float(rel)
            sorted_pairs = sorted(seen.items())
            deduped_indices.append([idx for idx, _ in sorted_pairs])
            deduped_relevance.append([rel for _, rel in sorted_pairs])
        return deduped_indices, deduped_relevance

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

    @property
    def binary_relevance_threshold(self) -> float:
        """Minimum graded relevance for an item to count as a positive.

        Used by binary metrics (``map``, ``mrr``, ``recall@k``, ``hit@k``,
        ``rp@k``) when the dataset provides ``target_relevance``: items with
        relevance ``>= threshold`` are treated as positives, items below it
        are dropped from the binary positive set but still contribute to
        graded metrics such as ``ndcg@k``.

        Default is ``1e-9`` so any listed item with a non-zero grade counts
        as a positive — equivalent to today's behavior where every entry in
        ``target_indices`` is a positive. Override on the task to express
        a stricter threshold (e.g. ``2.0`` on a 0-3 scale).

        Has no effect when ``target_relevance`` is ``None``.
        """
        return 1e-9

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize ranking task.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to parent Task class.
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
    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load dataset for specific ID and split.

        For tasks that are a union of monolingual datasets: dataset_id equals
        language code.

        For other tasks: dataset_id can encode arbitrary information.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.
        split : DatasetSplit
            Dataset split to load.

        Returns
        -------
        RankingDataset
            RankingDataset object.
        """

    def get_size_oneliner(self, dataset_id: str) -> str:
        """Get dataset summary to display for progress.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        str
            Human-readable size string.
        """
        dataset = self.datasets[dataset_id]
        return f"{len(dataset.query_texts)} queries x {len(dataset.target_space)} targets"

    def evaluate(
        self,
        model: ModelInterface,
        metrics: list[str] | None = None,
        dataset_id: str = "en",
    ) -> dict[str, float]:
        """Evaluate the model on this ranking task.

        Parameters
        ----------
        model : ModelInterface
            Model implementing ModelInterface (must have compute_rankings method).
        metrics : list[str] or None, optional
            List of metrics to compute. If None, uses default_metrics.
        dataset_id : str, optional
            Dataset identifier to evaluate on. Default is "en".

        Returns
        -------
        dict[str, float]
            Dictionary containing metric scores and evaluation metadata.
        """
        if metrics is None:
            metrics = self.default_metrics

        # Retrieve dataset by ID
        dataset = self.datasets[dataset_id]
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
            prediction_matrix = prediction_matrix.cpu().float().numpy()

        # Calculate metrics. When the dataset provides graded relevance, binary
        # metrics consume only positives with relevance >= binary_relevance_threshold;
        # nDCG still sees the full graded label list.
        metric_results = calculate_ranking_metrics(
            prediction_matrix=prediction_matrix,
            pos_label_idxs=labels,
            metrics=metrics,
            pos_label_relevance=dataset.target_relevance,
            binary_relevance_threshold=self.binary_relevance_threshold,
        )

        return metric_results
