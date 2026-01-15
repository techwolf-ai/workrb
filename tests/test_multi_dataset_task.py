"""
Test multi-dataset tasks that return multiple dataset IDs per language.

This test suite validates that tasks can override languages_to_dataset_ids()
to return multiple dataset identifiers for each language, supporting use cases
like MELO benchmark where datasets encode additional metadata beyond language.
"""

import pytest

from workrb.models import BiEncoderModel
from workrb.tasks import ESCOJob2SkillRanking, RankingDataset
from workrb.types import Language


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
