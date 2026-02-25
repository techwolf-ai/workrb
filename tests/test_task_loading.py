"""Test loading of all concrete task implementations."""

from typing import Literal

import pytest

from tests.test_utils import GeneralRankingTestTask
from workrb.tasks import (
    ESCOJob2SkillClassification,
    ESCOJob2SkillRanking,
    ESCOSkill2JobRanking,
    ESCOSkillNormRanking,
    HouseSkillExtractRanking,
    JobBERTJobNormRanking,
    JobTitleSimilarityRanking,
    MELORanking,
    MELSRanking,
    SkillMatch1kSkillSimilarityRanking,
    SkillSkapeExtractRanking,
    TechSkillExtractRanking,
)
from workrb.tasks.abstract.base import DatasetSplit, Language, TaskType
from workrb.types import DatasetLanguages


class StubSupportedLanguagesRankingTask(GeneralRankingTestTask):
    """Stub task class for testing language support."""

    @property
    def supported_query_languages(self):
        return [Language.EN, Language.DE]

    @property
    def supported_target_languages(self):
        return [Language.EN, Language.DE]

    @property
    def task_type(self):
        return TaskType.RANKING


def test_ranking_tasks_init_en_splits():
    """Test instantiation of all ranking task classes."""
    ranking_tasks = [
        ("ESCOJob2SkillRanking", ESCOJob2SkillRanking),
        ("ESCOSkill2JobRanking", ESCOSkill2JobRanking),
        ("ESCOSkillNormRanking", ESCOSkillNormRanking),
        ("JobNormRanking", JobBERTJobNormRanking),
        ("JobTitleSimilarityRanking", JobTitleSimilarityRanking),
        ("MELORanking", MELORanking),
        ("MELSRanking", MELSRanking),
        ("SkillExtractHouseRanking", HouseSkillExtractRanking),
        ("SkillExtractTechRanking", TechSkillExtractRanking),
        ("SkillExtractSkillSkapeRanking", SkillSkapeExtractRanking),
        ("SkillSimilarityRanking", SkillMatch1kSkillSimilarityRanking),
    ]

    tasks_with_only_test_set = [
        "JobTitleSimilarityRanking",
        "MELORanking",
        "MELSRanking",
    ]

    results = {"success": [], "failures": []}
    languages = [Language.EN]
    splits = [split for split in DatasetSplit]

    nb_total = 0
    for split in splits:
        for task_name, task_class in ranking_tasks:
            if split != DatasetSplit.TEST and task_name in tasks_with_only_test_set:
                continue
            nb_total += 1
            try:
                # Try to instantiate with minimal parameters
                task = task_class(split=split, languages=languages)
                results["success"].append((task_name, task.name))
            except ImportError as e:
                results["failures"].append((task_name, f"Error: {e}"))

    # Print results
    print(f"\n{'=' * 60}")
    print("RANKING TASK LOADING TEST RESULTS")
    print(f"{'=' * 60}")

    if results["success"]:
        print(f"\n✓ SUCCESSFULLY LOADED ({len(results['success'])}):")
        for task_name, display_name in results["success"]:
            print(f"  • {task_name}: {display_name}")

    if results["failures"]:
        print(f"\n⚠️ FAILED TO LOAD ({len(results['failures'])}):")
        for task_name, reason in results["failures"]:
            print(f"  • {task_name}: {reason}")

    # Test should pass if we have some successes and no unexpected failures
    assert len(results["success"]) == nb_total, "Not all ranking tasks could be loaded successfully"
    assert len(results["failures"]) == 0, f"Tasks failed to load: {results['failures']}"

    print("\n🎉 Ranking task loading test passed!")


def test_classification_tasks_init_en_splits():
    """Test instantiation of all classification task classes."""
    classification_tasks = [
        ("JobSkillClassification", ESCOJob2SkillClassification),
    ]

    results = {"success": [], "failures": []}
    languages = [Language.EN]
    splits = [split for split in DatasetSplit]

    nb_total = 0
    for split in splits:
        for task_name, task_class in classification_tasks:
            nb_total += 1
            try:
                # Try to instantiate with minimal parameters
                task = task_class(split=split, languages=languages)
                results["success"].append((task_name, task.name))
            except (ImportError, FileNotFoundError) as e:
                results["failures"].append((task_name, f"Error: {e}"))

    # Print results
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION TASK LOADING TEST RESULTS")
    print(f"{'=' * 60}")

    if results["success"]:
        print(f"\n✓ SUCCESSFULLY LOADED ({len(results['success'])}):")
        for task_name, display_name in results["success"]:
            print(f"  • {task_name}: {display_name}")

    if results["failures"]:
        print(f"\n⚠️ FAILED TO LOAD ({len(results['failures'])}):")
        for task_name, reason in results["failures"]:
            print(f"  • {task_name}: {reason}")

    # Test should pass if we have some successes
    # Note: Failures are expected if data files are not present
    print(f"\n📊 Summary: {len(results['success'])}/{nb_total} tasks loaded successfully")

    # Only fail if ALL tasks fail
    assert len(results["success"]) > 0, "All classification tasks failed to load"

    print("\n🎉 Classification task loading test passed!")


class TestTaskLanguageSupport:
    faulty_lang_input = ["i am not a language input"]
    supported_languages = [Language.EN.value, Language.DE.value]
    unsupported_languages = [Language.ES.value, Language.IT.value]
    mixed_faulty_lang_input = [faulty_lang_input[0], Language.EN.value]
    split = DatasetSplit.TEST

    @pytest.mark.parametrize("split", [DatasetSplit.TEST, DatasetSplit.VAL])
    def test_init_task_with_error_on_unsupported_languages(
        self, split, unsupported_lang_mode: Literal["error", "skip"] = "error"
    ):
        """Test that passing unsupported languages raises an error."""
        with pytest.raises(ValueError):
            StubSupportedLanguagesRankingTask(
                languages=self.unsupported_languages,
                split=self.split,
                unsupported_lang_mode=unsupported_lang_mode,
            )

        with pytest.raises(ValueError, match="not a valid Language"):
            StubSupportedLanguagesRankingTask(
                languages=self.faulty_lang_input,
                split=self.split,
                unsupported_lang_mode=unsupported_lang_mode,
            )

        with pytest.raises(ValueError, match="not a valid Language"):
            StubSupportedLanguagesRankingTask(
                languages=self.mixed_faulty_lang_input,
                split=self.split,
                unsupported_lang_mode=unsupported_lang_mode,
            )

        StubSupportedLanguagesRankingTask(
            languages=self.supported_languages,
            split=self.split,
            unsupported_lang_mode=unsupported_lang_mode,
        )

    @pytest.mark.parametrize("split", [DatasetSplit.TEST, DatasetSplit.VAL])
    def test_init_task_with_skip_on_unsupported_languages(
        self, split, unsupported_lang_mode: Literal["error", "skip"] = "skip"
    ):
        """Test that passing unsupported languages raises an error."""
        StubSupportedLanguagesRankingTask(
            languages=self.unsupported_languages,
            split=self.split,
            unsupported_lang_mode=unsupported_lang_mode,
        )

        with pytest.raises(ValueError, match="not a valid Language"):
            StubSupportedLanguagesRankingTask(
                languages=self.faulty_lang_input,
                split=self.split,
                unsupported_lang_mode=unsupported_lang_mode,
            )

        with pytest.raises(ValueError, match="not a valid Language"):
            StubSupportedLanguagesRankingTask(
                languages=self.mixed_faulty_lang_input,
                split=self.split,
                unsupported_lang_mode=unsupported_lang_mode,
            )

        StubSupportedLanguagesRankingTask(
            languages=self.supported_languages,
            split=self.split,
            unsupported_lang_mode=unsupported_lang_mode,
        )


class TestMELORankingDatasetIds:
    """Test MELORanking.languages_to_dataset_ids filtering logic."""

    @pytest.mark.parametrize(
        "languages,expected_dataset_ids",
        [
            # Single language: only monolingual English dataset
            (
                [Language.EN],
                {"usa_q_en_c_en"},
            ),
            # Bulgarian only: only monolingual Bulgarian dataset
            (
                [Language.BG],
                {"bgr_q_bg_c_bg"},
            ),
            # Bulgarian + English: monolingual + cross-lingual datasets for both
            (
                [Language.BG, Language.EN],
                {"bgr_q_bg_c_bg", "bgr_q_bg_c_en", "usa_q_en_c_en"},
            ),
            # Czech + English: monolingual + cross-lingual datasets
            (
                [Language.CS, Language.EN],
                {"cze_q_cs_c_cs", "cze_q_cs_c_en", "usa_q_en_c_en"},
            ),
            # All languages needed for usa_q_en_c_de_en_es_fr_it_nl_pl_pt
            (
                [
                    Language.EN,
                    Language.DE,
                    Language.ES,
                    Language.FR,
                    Language.IT,
                    Language.NL,
                    Language.PL,
                    Language.PT,
                ],
                {
                    "aut_q_de_c_de",
                    "aut_q_de_c_en",
                    "bel_q_fr_c_fr",
                    "bel_q_fr_c_en",
                    "bel_q_nl_c_nl",
                    "bel_q_nl_c_en",
                    "deu_q_de_c_de",
                    "deu_q_de_c_en",
                    "esp_q_es_c_en",
                    "esp_q_es_c_es",
                    "fra_q_fr_c_en",
                    "fra_q_fr_c_fr",
                    "ita_q_it_c_en",
                    "ita_q_it_c_it",
                    "nld_q_nl_c_en",
                    "nld_q_nl_c_nl",
                    "pol_q_pl_c_en",
                    "pol_q_pl_c_pl",
                    "prt_q_pt_c_en",
                    "prt_q_pt_c_pt",
                    "usa_q_en_c_de_en_es_fr_it_nl_pl_pt",
                    "usa_q_en_c_en",
                },
            ),
            # Two languages without cross-lingual datasets between them
            (
                [Language.BG, Language.CS],
                {"bgr_q_bg_c_bg", "cze_q_cs_c_cs"},
            ),
        ],
    )
    def test_languages_to_dataset_ids(self, languages, expected_dataset_ids):
        """Test that dataset_ids matches expected for given language combinations."""
        task = MELORanking(split="test", languages=[lang.value for lang in languages])
        assert set(task.dataset_ids) == expected_dataset_ids


class TestMELOGetDatasetLanguages:
    """Test MELORanking.get_dataset_languages for all dataset IDs."""

    def test_all_dataset_ids_return_valid_dataset_languages(self):
        """get_dataset_languages returns a valid DatasetLanguages for every MELO dataset ID."""
        task = MELORanking(
            split="test",
            languages=[lang.value for lang in MELORanking.MELO_LANGUAGES],
        )
        for dataset_id in task.dataset_ids:
            result = task.get_dataset_languages(dataset_id)
            assert isinstance(result, DatasetLanguages), (
                f"Expected DatasetLanguages for '{dataset_id}', got {type(result)}"
            )
            assert len(result.input_languages) >= 1
            assert len(result.output_languages) >= 1

    @pytest.mark.parametrize(
        "dataset_id,expected_input,expected_output",
        [
            # Monolingual: query and corpus in the same language
            ("bgr_q_bg_c_bg", frozenset({Language.BG}), frozenset({Language.BG})),
            ("usa_q_en_c_en", frozenset({Language.EN}), frozenset({Language.EN})),
            # Cross-lingual: query in one language, corpus in another
            ("bgr_q_bg_c_en", frozenset({Language.BG}), frozenset({Language.EN})),
            ("deu_q_de_c_en", frozenset({Language.DE}), frozenset({Language.EN})),
            ("fra_q_fr_c_fr", frozenset({Language.FR}), frozenset({Language.FR})),
            # Multilingual corpus
            (
                "usa_q_en_c_de_en_es_fr_it_nl_pl_pt",
                frozenset({Language.EN}),
                frozenset(
                    {
                        Language.DE,
                        Language.EN,
                        Language.ES,
                        Language.FR,
                        Language.IT,
                        Language.NL,
                        Language.PL,
                        Language.PT,
                    }
                ),
            ),
        ],
    )
    def test_specific_dataset_languages(self, dataset_id, expected_input, expected_output):
        """get_dataset_languages returns correct input/output languages for specific datasets."""
        task = MELORanking(
            split="test",
            languages=[lang.value for lang in MELORanking.MELO_LANGUAGES],
        )
        result = task.get_dataset_languages(dataset_id)
        assert result.input_languages == expected_input, (
            f"Input mismatch for '{dataset_id}': {result.input_languages} != {expected_input}"
        )
        assert result.output_languages == expected_output, (
            f"Output mismatch for '{dataset_id}': {result.output_languages} != {expected_output}"
        )


class TestMELSGetDatasetLanguages:
    """Test MELSRanking.get_dataset_languages for all dataset IDs."""

    def test_all_dataset_ids_return_valid_dataset_languages(self):
        """get_dataset_languages returns a valid DatasetLanguages for every MELS dataset ID."""
        task = MELSRanking(
            split="test",
            languages=[lang.value for lang in MELSRanking.MELS_LANGUAGES],
        )
        for dataset_id in task.dataset_ids:
            result = task.get_dataset_languages(dataset_id)
            assert isinstance(result, DatasetLanguages), (
                f"Expected DatasetLanguages for '{dataset_id}', got {type(result)}"
            )
            assert len(result.input_languages) >= 1
            assert len(result.output_languages) >= 1

    @pytest.mark.parametrize(
        "dataset_id,expected_input,expected_output",
        [
            # Monolingual
            ("deu_q_de_c_de", frozenset({Language.DE}), frozenset({Language.DE})),
            ("bel_q_fr_c_fr", frozenset({Language.FR}), frozenset({Language.FR})),
            ("bel_q_nl_c_nl", frozenset({Language.NL}), frozenset({Language.NL})),
            ("swe_q_sv_c_sv", frozenset({Language.SV}), frozenset({Language.SV})),
            # Cross-lingual
            ("bel_q_fr_c_en", frozenset({Language.FR}), frozenset({Language.EN})),
            ("bel_q_nl_c_en", frozenset({Language.NL}), frozenset({Language.EN})),
            ("deu_q_de_c_en", frozenset({Language.DE}), frozenset({Language.EN})),
            ("swe_q_sv_c_en", frozenset({Language.SV}), frozenset({Language.EN})),
        ],
    )
    def test_specific_dataset_languages(self, dataset_id, expected_input, expected_output):
        """get_dataset_languages returns correct input/output languages for specific datasets."""
        task = MELSRanking(
            split="test",
            languages=[lang.value for lang in MELSRanking.MELS_LANGUAGES],
        )
        result = task.get_dataset_languages(dataset_id)
        assert result.input_languages == expected_input, (
            f"Input mismatch for '{dataset_id}': {result.input_languages} != {expected_input}"
        )
        assert result.output_languages == expected_output, (
            f"Output mismatch for '{dataset_id}': {result.output_languages} != {expected_output}"
        )
