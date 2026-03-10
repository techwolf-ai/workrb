"""Tests for DuplicateStrategy in RankingDataset."""

import pytest

from workrb.tasks.abstract.ranking_base import DuplicateStrategy, RankingDataset


class TestDuplicateStrategyRaise:
    """Test that RAISE strategy raises on duplicates."""

    def test_raise_on_duplicate_targets(self):
        with pytest.raises(AssertionError, match="Target texts must be unique"):
            RankingDataset(
                query_texts=["q1"],
                target_indices=[[0, 1]],
                target_space=["a", "b", "a"],
                dataset_id="test",
                duplicate_target_strategy=DuplicateStrategy.RAISE,
            )

    def test_raise_on_duplicate_queries(self):
        with pytest.raises(AssertionError, match="Query texts must be unique"):
            RankingDataset(
                query_texts=["q1", "q1"],
                target_indices=[[0], [1]],
                target_space=["a", "b"],
                dataset_id="test",
                duplicate_query_strategy=DuplicateStrategy.RAISE,
            )

    def test_no_raise_without_duplicates(self):
        ds = RankingDataset(
            query_texts=["q1", "q2"],
            target_indices=[[0], [1]],
            target_space=["a", "b"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.RAISE,
            duplicate_target_strategy=DuplicateStrategy.RAISE,
        )
        assert ds.query_texts == ["q1", "q2"]
        assert ds.target_space == ["a", "b"]


class TestDuplicateStrategyAllow:
    """Test that ALLOW strategy silently accepts duplicates."""

    def test_allow_duplicate_targets(self):
        ds = RankingDataset(
            query_texts=["q1"],
            target_indices=[[0, 2]],
            target_space=["a", "b", "a"],
            dataset_id="test",
            duplicate_target_strategy=DuplicateStrategy.ALLOW,
        )
        assert ds.target_space == ["a", "b", "a"]
        assert ds.target_indices == [[0, 2]]

    def test_allow_duplicate_queries(self):
        ds = RankingDataset(
            query_texts=["q1", "q1"],
            target_indices=[[0], [1]],
            target_space=["a", "b"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.ALLOW,
        )
        assert ds.query_texts == ["q1", "q1"]


class TestResolveDuplicateTargets:
    """Test deterministic deduplication of target_space."""

    def test_keeps_first_occurrence(self):
        ds = RankingDataset(
            query_texts=["q1"],
            target_indices=[[0, 2]],
            target_space=["a", "b", "a"],
            dataset_id="test",
            duplicate_target_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.target_space == ["a", "b"]

    def test_remaps_indices(self):
        # target_space: ["x", "y", "x", "z"] -> ["x", "y", "z"]
        # old indices: 0->0, 1->1, 2->0, 3->2
        ds = RankingDataset(
            query_texts=["q1", "q2"],
            target_indices=[[0, 2, 3], [1, 2]],
            target_space=["x", "y", "x", "z"],
            dataset_id="test",
            duplicate_target_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.target_space == ["x", "y", "z"]
        # [0, 2, 3] -> [0, 0, 2] -> deduplicated & sorted: [0, 2]
        assert ds.target_indices[0] == [0, 2]
        # [1, 2] -> [1, 0] -> sorted: [0, 1]
        assert ds.target_indices[1] == [0, 1]

    def test_preserves_order(self):
        ds = RankingDataset(
            query_texts=["q1"],
            target_indices=[[0]],
            target_space=["c", "a", "b", "a", "c"],
            dataset_id="test",
            duplicate_target_strategy=DuplicateStrategy.RESOLVE,
        )
        # First occurrences in order: c, a, b
        assert ds.target_space == ["c", "a", "b"]

    def test_no_change_without_duplicates(self):
        ds = RankingDataset(
            query_texts=["q1"],
            target_indices=[[0, 1, 2]],
            target_space=["a", "b", "c"],
            dataset_id="test",
            duplicate_target_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.target_space == ["a", "b", "c"]
        assert ds.target_indices == [[0, 1, 2]]


class TestResolveDuplicateQueries:
    """Test deterministic deduplication of query_texts."""

    def test_merges_target_indices(self):
        ds = RankingDataset(
            query_texts=["q1", "q1"],
            target_indices=[[0], [1]],
            target_space=["a", "b", "c"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.query_texts == ["q1"]
        assert ds.target_indices == [[0, 1]]

    def test_union_deduplicates_indices(self):
        ds = RankingDataset(
            query_texts=["q1", "q1"],
            target_indices=[[0, 1], [1, 2]],
            target_space=["a", "b", "c"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.query_texts == ["q1"]
        assert ds.target_indices == [[0, 1, 2]]

    def test_preserves_order_of_first_occurrence(self):
        ds = RankingDataset(
            query_texts=["q2", "q1", "q2"],
            target_indices=[[0], [1], [2]],
            target_space=["a", "b", "c"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.query_texts == ["q2", "q1"]
        assert ds.target_indices == [[0, 2], [1]]

    def test_no_change_without_duplicates(self):
        ds = RankingDataset(
            query_texts=["q1", "q2"],
            target_indices=[[0], [1]],
            target_space=["a", "b"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.query_texts == ["q1", "q2"]
        assert ds.target_indices == [[0], [1]]


class TestResolveBothDuplicates:
    """Test combined target + query deduplication."""

    def test_resolve_targets_then_queries(self):
        # Targets: ["x", "y", "x"] -> ["x", "y"], remap 2->0
        # Queries: ["q1", "q1"] with indices [[0, 2], [1]] -> after target remap [[0], [1]]
        #   -> merged: [[0, 1]]
        ds = RankingDataset(
            query_texts=["q1", "q1"],
            target_indices=[[0, 2], [1]],
            target_space=["x", "y", "x"],
            dataset_id="test",
            duplicate_query_strategy=DuplicateStrategy.RESOLVE,
            duplicate_target_strategy=DuplicateStrategy.RESOLVE,
        )
        assert ds.target_space == ["x", "y"]
        assert ds.query_texts == ["q1"]
        assert ds.target_indices == [[0, 1]]


class TestDefaultBehavior:
    """Test that defaults resolve both query and target duplicates."""

    def test_default_resolves_duplicate_queries(self):
        ds = RankingDataset(
            query_texts=["q1", "q1"],
            target_indices=[[0], [1]],
            target_space=["a", "b"],
            dataset_id="test",
        )
        assert ds.query_texts == ["q1"]
        assert ds.target_indices == [[0, 1]]

    def test_default_resolves_duplicate_targets(self):
        ds = RankingDataset(
            query_texts=["q1"],
            target_indices=[[0, 1]],
            target_space=["a", "a"],
            dataset_id="test",
        )
        assert ds.target_space == ["a"]
        assert ds.target_indices == [[0]]
