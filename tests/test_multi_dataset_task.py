"""
Test multi-dataset tasks that return multiple dataset IDs per language.

This test suite validates that tasks can override languages_to_dataset_ids()
to return multiple dataset identifiers for each language, supporting use cases
like MELO benchmark where datasets encode additional metadata beyond language.
"""

import time

import pytest

from workrb.models import BiEncoderModel
from workrb.results import (
    BenchmarkMetadata,
    BenchmarkResults,
    MetricsResult,
    TaskResultMetadata,
    TaskResults,
)
from workrb.run import _filter_pending_work
from workrb.tasks import ESCOJob2SkillRanking, RankingDataset
from workrb.types import (
    DatasetLanguages,
    Language,
    LanguageAggregationMode,
    get_language_grouping_key,
)


class TestMultiDatasetTask:
    """Test tasks that return multiple dataset IDs per language."""

    def test_languages_to_dataset_ids_multiple_per_language(self):
        """Test task that returns multiple dataset IDs per language."""

        # Create a custom task class that overrides languages_to_dataset_ids
        class MultiDatasetTask(ESCOJob2SkillRanking):
            def languages_to_dataset_ids(self, languages: list[Language]) -> list[str]:
                """Map languages to multiple dataset IDs with custom logic."""
                dataset_ids = []
                lang_set = set(languages)

                # English -> 4 datasets
                if Language.EN in lang_set:
                    dataset_ids.extend(["en1", "en2", "en3_sea", "en3_land"])

                # French -> 2 datasets
                if Language.FR in lang_set:
                    dataset_ids.extend(["fr1", "fr2"])

                # German -> 1 dataset
                if Language.DE in lang_set:
                    dataset_ids.append("de")

                # Spanish -> 3 datasets
                if Language.ES in lang_set:
                    dataset_ids.extend(["es1", "es2", "es3_air"])

                # Cross-language datasets when both French and German are present
                if Language.FR in lang_set and Language.DE in lang_set:
                    dataset_ids.extend(["fr_de_land", "fr_de_sea"])

                return dataset_ids

            def load_dataset(self, dataset_id: str, split):
                """Mock load_dataset to avoid loading real data."""
                # For testing, we just need to verify the dataset_ids are correct
                # Return a minimal mock dataset structure
                return RankingDataset(
                    query_texts=["mock query"],
                    target_indices=[[0]],
                    target_space=["mock target"],
                    dataset_id=dataset_id,
                )

        # Test 1: English only
        task_en = MultiDatasetTask(split="val", languages=["en"])
        assert task_en.dataset_ids == ["en1", "en2", "en3_sea", "en3_land"]
        assert len(task_en.datasets) == 4
        assert all(dataset_id in task_en.datasets for dataset_id in task_en.dataset_ids)

        # Test 2: French only
        task_fr = MultiDatasetTask(split="val", languages=["fr"])
        assert task_fr.dataset_ids == ["fr1", "fr2"]
        assert len(task_fr.datasets) == 2

        # Test 3: German only
        task_de = MultiDatasetTask(split="val", languages=["de"])
        assert task_de.dataset_ids == ["de"]
        assert len(task_de.datasets) == 1

        # Test 4: Spanish only
        task_es = MultiDatasetTask(split="val", languages=["es"])
        assert task_es.dataset_ids == ["es1", "es2", "es3_air"]
        assert len(task_es.datasets) == 3

        # Test 5: French + German (includes cross-language datasets)
        task_fr_de = MultiDatasetTask(split="val", languages=["fr", "de"])
        assert set(task_fr_de.dataset_ids) == {
            "fr1",
            "fr2",
            "de",
            "fr_de_land",
            "fr_de_sea",
        }
        assert len(task_fr_de.datasets) == 5

        # Test 6: Multiple languages
        task_multi = MultiDatasetTask(split="val", languages=["en", "fr", "es"])
        expected = ["en1", "en2", "en3_sea", "en3_land", "fr1", "fr2", "es1", "es2", "es3_air"]
        assert task_multi.dataset_ids == expected
        assert len(task_multi.datasets) == 9

    def test_multi_dataset_task_with_biencoder(self):
        """Test that multi-dataset tasks work with actual model evaluation."""

        class ToyMultiDatasetTask(ESCOJob2SkillRanking):
            def languages_to_dataset_ids(self, languages: list[Language]) -> list[str]:
                """Return multiple dataset IDs per language."""
                dataset_ids = []
                if Language.EN in languages:
                    dataset_ids.extend(["en1", "en2"])
                return dataset_ids

            def load_dataset(self, dataset_id: str, split):
                """Load minimal toy dataset."""
                from workrb.tasks.abstract.ranking_base import RankingDataset

                # Create tiny datasets for testing
                return RankingDataset(
                    query_texts=["Software Engineer", "Data Scientist"],
                    target_indices=[[0, 1], [1, 2]],
                    target_space=["Python", "Machine Learning", "SQL"],
                    dataset_id=dataset_id,
                )

        # Create task with multiple datasets
        task = ToyMultiDatasetTask(split="val", languages=["en"])
        assert task.dataset_ids == ["en1", "en2"]

        # Verify we can evaluate on each dataset
        model = BiEncoderModel("all-MiniLM-L6-v2")

        # Evaluate on first dataset
        results_en1 = task.evaluate(model, dataset_id="en1")
        assert "map" in results_en1
        assert 0 <= results_en1["map"] <= 1

        # Evaluate on second dataset
        results_en2 = task.evaluate(model, dataset_id="en2")
        assert "map" in results_en2
        assert 0 <= results_en2["map"] <= 1

    def test_multi_dataset_task_evaluation_all_datasets(self):
        """Test that evaluation pipeline processes all datasets."""

        class ToyMultiDatasetTask(ESCOJob2SkillRanking):
            def languages_to_dataset_ids(self, languages: list[Language]) -> list[Language]:
                """Return 2 datasets for English."""
                dataset_ids = []
                if Language.EN in languages:
                    dataset_ids.extend(["en_region_a", "en_region_b"])
                return dataset_ids

            def load_dataset(self, dataset_id: str, split):
                """Load minimal toy dataset."""
                from workrb.tasks.abstract.ranking_base import RankingDataset

                return RankingDataset(
                    query_texts=["test query"],
                    target_indices=[[0]],
                    target_space=["test target"],
                    dataset_id=dataset_id,
                )

        task = ToyMultiDatasetTask(split="val", languages=["en"])

        # Verify dataset_ids are correct
        assert task.dataset_ids == ["en_region_a", "en_region_b"]

        # Verify both datasets are loaded
        assert "en_region_a" in task.datasets
        assert "en_region_b" in task.datasets

        # Verify dataset objects have correct dataset_id
        assert task.datasets["en_region_a"].dataset_id == "en_region_a"
        assert task.datasets["en_region_b"].dataset_id == "en_region_b"


def _make_benchmark_results(
    dataset_entries: list[tuple[str, str, dict[str, float], list[str], list[str]]],
) -> BenchmarkResults:
    """Build a BenchmarkResults with controlled MetricsResult entries.

    Parameters
    ----------
    dataset_entries : list of tuples
        Each tuple is ``(task_name, dataset_id, metrics_dict, input_languages, output_languages)``.
    """
    task_results: dict[str, TaskResults] = {}
    for task_name, dataset_id, metrics_dict, inp_langs, out_langs in dataset_entries:
        if task_name not in task_results:
            task_results[task_name] = TaskResults(
                metadata=TaskResultMetadata(
                    task_group="test_group",
                    task_type="ranking",
                    label_type="single_label",
                    description="test",
                    split="val",
                ),
                datasetid_results={},
            )
        task_results[task_name].datasetid_results[dataset_id] = MetricsResult(
            evaluation_time=0.1,
            metrics_dict=metrics_dict,
            input_languages=inp_langs,
            output_languages=out_langs,
        )
    return BenchmarkResults(
        task_results=task_results,
        metadata=BenchmarkMetadata(
            model_name="test_model",
            total_evaluation_time=1.0,
            timestamp=time.time(),
            num_tasks=len(task_results),
            languages=["en"],
        ),
    )


class TestGetDatasetLanguages:
    """Tests for Task.get_dataset_languages()."""

    def test_default_monolingual(self):
        """Default implementation returns matching DatasetLanguages for standard language IDs."""

        class MonoTask(ESCOJob2SkillRanking):
            def load_dataset(self, dataset_id, split):
                return RankingDataset(
                    query_texts=["q"],
                    target_indices=[[0]],
                    target_space=["t"],
                    dataset_id=dataset_id,
                )

        task = MonoTask(split="val", languages=["en"])
        result = task.get_dataset_languages("en")
        assert result == DatasetLanguages(
            input_languages=frozenset({Language.EN}),
            output_languages=frozenset({Language.EN}),
        )

    def test_raises_for_non_standard_ids(self):
        """Default implementation raises NotImplementedError for arbitrary dataset IDs."""

        class ArbitraryIdTask(ESCOJob2SkillRanking):
            def languages_to_dataset_ids(self, languages):
                return ["custom_dataset_1"]

            def load_dataset(self, dataset_id, split):
                return RankingDataset(
                    query_texts=["q"],
                    target_indices=[[0]],
                    target_space=["t"],
                    dataset_id=dataset_id,
                )

        task = ArbitraryIdTask(split="val", languages=["en"])
        with pytest.raises(NotImplementedError, match="not a valid language code"):
            task.get_dataset_languages("custom_dataset_1")


class TestAggregationModes:
    """Tests for _aggregate_per_language aggregation modes."""

    def test_monolingual_only_with_monolingual_datasets(self):
        """MONOLINGUAL_ONLY correctly groups monolingual datasets by language."""
        br = _make_benchmark_results(
            [
                ("task1", "en", {"map": 0.8}, ["en"], ["en"]),
                ("task1", "de", {"map": 0.6}, ["de"], ["de"]),
                ("task2", "en", {"map": 0.9}, ["en"], ["en"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.MONOLINGUAL_ONLY,
        )
        result_str = {str(k): v for k, v in result.items()}
        assert "mean_per_language/en/map/mean" in result_str
        assert "mean_per_language/de/map/mean" in result_str
        # en: mean of 0.8 and 0.9 = 0.85
        assert result_str["mean_per_language/en/map/mean"] == pytest.approx(0.85)
        # de: single value 0.6
        assert result_str["mean_per_language/de/map/mean"] == pytest.approx(0.6)

    def test_monolingual_only_skips_crosslingual_dataset(self):
        """MONOLINGUAL_ONLY skips cross-lingual datasets."""
        br = _make_benchmark_results(
            [
                ("task1", "en_de", {"map": 0.7}, ["en"], ["de"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.MONOLINGUAL_ONLY,
        )
        assert result == {}

    def test_monolingual_only_skips_multilingual_dataset(self):
        """MONOLINGUAL_ONLY skips multilingual datasets."""
        br = _make_benchmark_results(
            [
                ("task1", "multi", {"map": 0.7}, ["en", "fr"], ["en", "fr"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.MONOLINGUAL_ONLY,
        )
        assert result == {}

    def test_crosslingual_group_input_languages(self):
        """CROSSLINGUAL_GROUP_INPUT_LANGUAGES groups by input language."""
        br = _make_benchmark_results(
            [
                ("task1", "en_to_de", {"map": 0.7}, ["en"], ["de"]),
                ("task1", "en_to_fr", {"map": 0.9}, ["en"], ["fr"]),
                ("task1", "de_to_fr", {"map": 0.5}, ["de"], ["fr"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES,
        )
        result_str = {str(k): v for k, v in result.items()}
        assert "mean_per_language/en/map/mean" in result_str
        assert "mean_per_language/de/map/mean" in result_str
        # en: mean of 0.7 and 0.9 = 0.8
        assert result_str["mean_per_language/en/map/mean"] == pytest.approx(0.8)
        # de: single value 0.5
        assert result_str["mean_per_language/de/map/mean"] == pytest.approx(0.5)

    def test_crosslingual_group_input_skips_multi_input(self):
        """CROSSLINGUAL_GROUP_INPUT_LANGUAGES skips datasets with multiple input langs."""
        br = _make_benchmark_results(
            [
                ("task1", "multi_in", {"map": 0.7}, ["en", "fr"], ["de"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES,
        )
        assert result == {}

    def test_crosslingual_group_output_languages(self):
        """CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES groups by output language."""
        br = _make_benchmark_results(
            [
                ("task1", "en_to_de", {"map": 0.7}, ["en"], ["de"]),
                ("task1", "fr_to_de", {"map": 0.9}, ["fr"], ["de"]),
                ("task1", "en_to_fr", {"map": 0.5}, ["en"], ["fr"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES,
        )
        result_str = {str(k): v for k, v in result.items()}
        assert "mean_per_language/de/map/mean" in result_str
        assert "mean_per_language/fr/map/mean" in result_str
        # de: mean of 0.7 and 0.9 = 0.8
        assert result_str["mean_per_language/de/map/mean"] == pytest.approx(0.8)
        # fr: single value 0.5
        assert result_str["mean_per_language/fr/map/mean"] == pytest.approx(0.5)

    def test_crosslingual_group_output_skips_multi_output(self):
        """CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES skips datasets with multiple output langs."""
        br = _make_benchmark_results(
            [
                ("task1", "multi_out", {"map": 0.7}, ["en"], ["de", "fr"]),
            ]
        )
        result = br._aggregate_per_language(
            aggregation_mode=LanguageAggregationMode.CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES,
        )
        assert result == {}


class TestGetLanguageGroupingKey:
    """Tests for the standalone get_language_grouping_key() function."""

    def test_monolingual_returns_language(self):
        """Monolingual dataset returns the shared language."""
        assert (
            get_language_grouping_key(["en"], ["en"], LanguageAggregationMode.MONOLINGUAL_ONLY)
            == "en"
        )

    def test_monolingual_skips_crosslingual(self):
        """MONOLINGUAL_ONLY returns None for cross-lingual datasets."""
        assert (
            get_language_grouping_key(["en"], ["de"], LanguageAggregationMode.MONOLINGUAL_ONLY)
            is None
        )

    def test_monolingual_skips_multilingual(self):
        """MONOLINGUAL_ONLY returns None for multilingual datasets."""
        assert (
            get_language_grouping_key(
                ["en", "fr"], ["en", "fr"], LanguageAggregationMode.MONOLINGUAL_ONLY
            )
            is None
        )

    def test_group_input_returns_input_language(self):
        """CROSSLINGUAL_GROUP_INPUT_LANGUAGES returns the singleton input language."""
        assert (
            get_language_grouping_key(
                ["en"], ["de"], LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES
            )
            == "en"
        )

    def test_group_input_skips_multi_input(self):
        """CROSSLINGUAL_GROUP_INPUT_LANGUAGES returns None for multiple input languages."""
        assert (
            get_language_grouping_key(
                ["en", "fr"], ["de"], LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES
            )
            is None
        )

    def test_group_output_returns_output_language(self):
        """CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES returns the singleton output language."""
        assert (
            get_language_grouping_key(
                ["en"], ["de"], LanguageAggregationMode.CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES
            )
            == "de"
        )

    def test_group_output_skips_multi_output(self):
        """CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES returns None for multiple output languages."""
        assert (
            get_language_grouping_key(
                ["en"], ["de", "fr"], LanguageAggregationMode.CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES
            )
            is None
        )


class _MockTask:
    """Minimal mock task for testing _filter_pending_work."""

    def __init__(self, name: str, dataset_languages_map: dict[str, DatasetLanguages]):
        self.name = name
        self._dataset_languages_map = dataset_languages_map

    def get_dataset_languages(self, dataset_id: str) -> DatasetLanguages:
        return self._dataset_languages_map[dataset_id]


class TestExecutionModeFiltering:
    """Tests for _filter_pending_work()."""

    def test_lazy_monolingual_skips_crosslingual(self):
        """LAZY + MONOLINGUAL_ONLY skips cross-lingual datasets."""
        task = _MockTask(
            "task1",
            {
                "en": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.EN}),
                ),
                "en_de": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.DE}),
                ),
            },
        )
        pending = [(task, "en"), (task, "en_de")]
        result = _filter_pending_work(pending, LanguageAggregationMode.MONOLINGUAL_ONLY)
        assert len(result) == 1
        assert result[0][1] == "en"

    def test_lazy_group_input_keeps_crosslingual_singleton_input(self):
        """CROSSLINGUAL_GROUP_INPUT_LANGUAGES keeps cross-lingual datasets with singleton input."""
        task = _MockTask(
            "task1",
            {
                "en_de": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.DE}),
                ),
                "multi_in": DatasetLanguages(
                    input_languages=frozenset({Language.EN, Language.FR}),
                    output_languages=frozenset({Language.DE}),
                ),
            },
        )
        pending = [(task, "en_de"), (task, "multi_in")]
        result = _filter_pending_work(
            pending, LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES
        )
        assert len(result) == 1
        assert result[0][1] == "en_de"

    def test_monolingual_only_mixed_task_keeps_only_monolingual(self):
        """MONOLINGUAL_ONLY keeps monolingual datasets and filters all cross-lingual ones.

        Simulates a MELO-like task with monolingual datasets (en, de) alongside
        several cross-lingual datasets (en_de, fr_de) — only the monolingual
        ones should survive filtering.
        """
        task = _MockTask(
            "melo_task",
            {
                "en": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.EN}),
                ),
                "de": DatasetLanguages(
                    input_languages=frozenset({Language.DE}),
                    output_languages=frozenset({Language.DE}),
                ),
                "en_de": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.DE}),
                ),
                "fr_de": DatasetLanguages(
                    input_languages=frozenset({Language.FR}),
                    output_languages=frozenset({Language.DE}),
                ),
                "multilingual": DatasetLanguages(
                    input_languages=frozenset({Language.EN, Language.DE, Language.FR}),
                    output_languages=frozenset({Language.EN, Language.DE, Language.FR}),
                ),
            },
        )
        pending = [
            (task, "en"),
            (task, "de"),
            (task, "en_de"),
            (task, "fr_de"),
            (task, "multilingual"),
        ]
        result = _filter_pending_work(pending, LanguageAggregationMode.MONOLINGUAL_ONLY)
        kept_ids = [dataset_id for _, dataset_id in result]
        assert kept_ids == ["en", "de"]

    def test_group_input_mixed_task_keeps_singleton_input(self):
        """CROSSLINGUAL_GROUP_INPUT_LANGUAGES keeps datasets with a single input language.

        Same MELO-like task: monolingual and single-input cross-lingual datasets
        survive, but the multilingual one (multiple input languages) is filtered.
        """
        task = _MockTask(
            "melo_task",
            {
                "en": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.EN}),
                ),
                "en_de": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.DE}),
                ),
                "fr_de": DatasetLanguages(
                    input_languages=frozenset({Language.FR}),
                    output_languages=frozenset({Language.DE}),
                ),
                "multilingual": DatasetLanguages(
                    input_languages=frozenset({Language.EN, Language.DE, Language.FR}),
                    output_languages=frozenset({Language.EN, Language.DE, Language.FR}),
                ),
            },
        )
        pending = [
            (task, "en"),
            (task, "en_de"),
            (task, "fr_de"),
            (task, "multilingual"),
        ]
        result = _filter_pending_work(
            pending, LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES
        )
        kept_ids = [dataset_id for _, dataset_id in result]
        assert kept_ids == ["en", "en_de", "fr_de"]

    def test_empty_pending_work(self):
        """Empty pending work returns empty list."""
        result = _filter_pending_work([], LanguageAggregationMode.MONOLINGUAL_ONLY)
        assert result == []

    def test_all_datasets_filtered(self):
        """All datasets incompatible with the mode results in empty list."""
        task = _MockTask(
            "task1",
            {
                "en_de": DatasetLanguages(
                    input_languages=frozenset({Language.EN}),
                    output_languages=frozenset({Language.DE}),
                ),
                "fr_es": DatasetLanguages(
                    input_languages=frozenset({Language.FR}),
                    output_languages=frozenset({Language.ES}),
                ),
            },
        )
        pending = [(task, "en_de"), (task, "fr_es")]
        result = _filter_pending_work(pending, LanguageAggregationMode.MONOLINGUAL_ONLY)
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
