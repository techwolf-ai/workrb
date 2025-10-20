"""
Test model-task compatibility for all combinations of model and task types.

This test suite validates that:
1. Classification tasks work with both BiEncoder and Classification models
2. Ranking tasks work with both BiEncoder and Classification models (when label space matches)
3. Proper error handling when models don't support tasks
4. Proper validation of label space mismatches
"""

import math

import pytest

from tests.test_utils import create_toy_task_class
from workrb.models import BiEncoderModel, RndESCOClassificationModel
from workrb.tasks import ESCOJob2SkillClassification, SkillMatch1kSkillSimilarityRanking
from workrb.tasks.abstract.base import Language
from workrb.tasks.abstract.classification_base import ClassificationDataset


def is_valid_metric(value: float) -> bool:
    """Check if a metric value is valid (in range or NaN for undefined metrics).

    Some metrics like ROC AUC can be NaN when only one class is present in the data,
    which is a valid state for small toy datasets.
    """
    return math.isnan(value) or (0 <= value <= 1)


class TestClassificationTaskWithBiEncoder:
    """Test classification tasks with BiEncoder models (ranking via similarity)."""

    def test_classification_task_with_biencoder_works(self):
        """BiEncoder should work on classification tasks by ranking texts against label space."""
        # Create toy classification task
        ToyJobSkill = create_toy_task_class(ESCOJob2SkillClassification)
        task = ToyJobSkill(split="val", languages=["en"])

        # Create BiEncoder model
        model = BiEncoderModel("all-MiniLM-L6-v2")

        # Should work - BiEncoder computes similarity between texts and label space
        results = task.evaluate(model, language=Language.EN)

        # Validate results
        assert "f1_macro" in results
        assert "roc_auc" in results
        assert 0 <= results["f1_macro"] <= 1
        assert is_valid_metric(results["roc_auc"])

    def test_classification_task_output_shape(self):
        """Verify BiEncoder returns correct shape for classification."""
        ToyJobSkill = create_toy_task_class(ESCOJob2SkillClassification)
        task = ToyJobSkill(split="val", languages=["en"])
        model = BiEncoderModel("all-MiniLM-L6-v2")

        # Get dataset
        dataset: ClassificationDataset = task.lang_datasets[Language.EN]

        # Compute predictions
        predictions = model.compute_classification(
            texts=dataset.texts,
            targets=dataset.label_space,
            input_type=task.input_type,
        )

        # Should return shape (n_texts, n_classes)
        assert predictions.shape[0] == len(dataset.texts)
        assert predictions.shape[1] == len(dataset.label_space)


class TestClassificationTaskWithClassificationModel:
    """Test classification tasks with classification models (proper classification head)."""

    def test_classification_task_with_classification_model_works(self):
        """Classification model should work on classification tasks when label space matches."""
        # Create toy classification task
        ToyJobSkill = create_toy_task_class(ESCOJob2SkillClassification)
        task = ToyJobSkill(split="val", languages=["en"])

        # Get the label space from the task
        dataset = task.lang_datasets[Language.EN]
        label_space = dataset.label_space

        # Create classification model with matching label space
        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=label_space,
        )

        # Should work - model has classification head with matching label space
        results = task.evaluate(model, language=Language.EN)

        # Validate results
        assert "f1_macro" in results
        assert "roc_auc" in results
        assert 0 <= results["f1_macro"] <= 1
        assert is_valid_metric(results["roc_auc"])

    def test_classification_task_label_space_size_mismatch_fails(self):
        """Classification model with wrong label space size should fail."""
        ToyJobSkill = create_toy_task_class(ESCOJob2SkillClassification)
        task = ToyJobSkill(split="val", languages=["en"])

        # Create model with different label space size
        wrong_label_space = ["Label1", "Label2", "Label3"]  # Only 3 labels
        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=wrong_label_space,
        )

        # Should fail with clear error about size mismatch
        with pytest.raises(ValueError, match="Model output size mismatch"):
            task.evaluate(model, language=Language.EN)

    def test_classification_task_label_space_order_mismatch_fails(self):
        """Classification model with wrong label order should fail."""
        ToyJobSkill = create_toy_task_class(ESCOJob2SkillClassification)
        task = ToyJobSkill(split="val", languages=["en"])

        # Get the label space and shuffle it
        dataset = task.lang_datasets[Language.EN]
        wrong_order_labels = list(reversed(dataset.label_space))

        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=wrong_order_labels,
        )

        # Should fail with clear error about order mismatch
        with pytest.raises(ValueError, match="label order doesn't match"):
            task.evaluate(model, language=Language.EN)


class TestRankingTaskWithBiEncoder:
    """Test ranking tasks with BiEncoder models (standard ranking behavior)."""

    def test_ranking_task_with_biencoder_works(self):
        """BiEncoder should work on ranking tasks (standard behavior)."""
        # Create toy ranking task
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
        task = ToySkillSim(split="val", languages=["en"])

        # Create BiEncoder model
        model = BiEncoderModel("all-MiniLM-L6-v2")

        # Should work - standard ranking behavior
        results = task.evaluate(model, language=Language.EN)

        # Validate results
        assert "map" in results
        assert "mrr" in results
        assert 0 <= results["map"] <= 1
        assert 0 <= results["mrr"] <= 1

    def test_ranking_task_output_shape(self):
        """Verify BiEncoder returns correct shape for ranking."""
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
        task = ToySkillSim(split="val", languages=["en"])
        model = BiEncoderModel("all-MiniLM-L6-v2")

        # Get dataset
        dataset = task.lang_datasets[Language.EN]

        # Compute predictions
        predictions = model.compute_rankings(
            queries=dataset.query_texts,
            targets=dataset.target_space,
            query_input_type=task.query_input_type,
            target_input_type=task.target_input_type,
        )

        # Should return shape (n_queries, n_targets)
        assert predictions.shape[0] == len(dataset.query_texts)
        assert predictions.shape[1] == len(dataset.target_space)


class TestRankingTaskWithClassificationModel:
    """Test ranking tasks with classification models (conditional support)."""

    def test_ranking_task_with_classification_model_matching_label_space_works(self):
        """Classification model should work on ranking when label space matches target space."""
        # Create toy ranking task
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
        task = ToySkillSim(split="val", languages=["en"])

        # Get the target space from the task
        dataset = task.lang_datasets[Language.EN]
        target_space = dataset.target_space

        # Create classification model with matching label space
        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=target_space,
        )

        # Should work - model's label space matches ranking target space
        results = task.evaluate(model, language=Language.EN)

        # Validate results
        assert "map" in results
        assert "mrr" in results
        assert 0 <= results["map"] <= 1
        assert 0 <= results["mrr"] <= 1

    def test_ranking_task_with_classification_model_size_mismatch_fails(self):
        """Classification model with wrong target space size should fail."""
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
        task = ToySkillSim(split="val", languages=["en"])

        # Create model with different label space size
        wrong_label_space = ["Skill1", "Skill2", "Skill3"]  # Wrong size
        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=wrong_label_space,
        )

        # Should fail with clear error about size mismatch
        with pytest.raises(ValueError, match="target space size mismatch"):
            task.evaluate(model, language=Language.EN)

    def test_ranking_task_with_classification_model_label_mismatch_fails(self):
        """Classification model with wrong labels should fail."""
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
        task = ToySkillSim(split="val", languages=["en"])

        # Get target space and create different labels with same size
        dataset = task.lang_datasets[Language.EN]
        wrong_labels = [f"WrongLabel_{i}" for i in range(len(dataset.target_space))]

        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=wrong_labels,
        )

        # Should fail with clear error about label mismatch
        with pytest.raises(ValueError, match="target labels don't match"):
            task.evaluate(model, language=Language.EN)

    def test_ranking_task_with_classification_model_order_mismatch_fails(self):
        """Classification model with wrong label order should fail."""
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
        task = ToySkillSim(split="val", languages=["en"])

        # Get target space and reverse order
        dataset = task.lang_datasets[Language.EN]
        wrong_order_labels = list(reversed(dataset.target_space))

        model = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=wrong_order_labels,
        )

        # Should fail with clear error about order mismatch
        with pytest.raises(ValueError, match="target label order doesn't match"):
            task.evaluate(model, language=Language.EN)


class TestModelTaskCompatibilitySummary:
    """Summary test that validates all combinations work as expected."""

    def test_all_model_task_combinations(self):
        """Test all four model-task combinations in one comprehensive test."""
        # Setup tasks
        ToyJobSkill = create_toy_task_class(ESCOJob2SkillClassification)
        ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)

        classification_task = ToyJobSkill(split="val", languages=["en"])
        ranking_task = ToySkillSim(split="val", languages=["en"])

        # Setup models
        biencoder_model = BiEncoderModel("all-MiniLM-L6-v2")

        # Get label spaces
        class_dataset = classification_task.lang_datasets[Language.EN]
        rank_dataset = ranking_task.lang_datasets[Language.EN]

        classification_model_for_class = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=class_dataset.label_space,
        )
        classification_model_for_rank = RndESCOClassificationModel(
            base_model_name="all-MiniLM-L6-v2",
            label_space=rank_dataset.target_space,
        )

        # Test all combinations
        results = {}

        # 1. Classification Task + BiEncoder (NEW)
        results["class_biencoder"] = classification_task.evaluate(
            biencoder_model, language=Language.EN
        )
        assert "f1_macro" in results["class_biencoder"]

        # 2. Classification Task + Classification Model (EXISTING)
        results["class_classification"] = classification_task.evaluate(
            classification_model_for_class, language=Language.EN
        )
        assert "f1_macro" in results["class_classification"]

        # 3. Ranking Task + BiEncoder (EXISTING)
        results["rank_biencoder"] = ranking_task.evaluate(biencoder_model, language=Language.EN)
        assert "map" in results["rank_biencoder"]

        # 4. Ranking Task + Classification Model (CONDITIONAL)
        results["rank_classification"] = ranking_task.evaluate(
            classification_model_for_rank, language=Language.EN
        )
        assert "map" in results["rank_classification"]

        print("\n" + "=" * 70)
        print("Model-Task Compatibility Test Summary")
        print("=" * 70)
        print("\n✅ All 4 model-task combinations work correctly:")
        print("  1. Classification Task + BiEncoder")
        print("  2. Classification Task + Classification Model")
        print("  3. Ranking Task + BiEncoder")
        print("  4. Ranking Task + Classification Model (matching label space)")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Allow running as standalone script
    pytest.main([__file__, "-v"])
