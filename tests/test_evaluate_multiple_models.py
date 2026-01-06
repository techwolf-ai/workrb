"""
Tests for evaluate_multiple_models function.

This test suite validates the evaluate_multiple_models function to ensure:
1. Multiple models are evaluated correctly with separate output folders
2. Input validation works (duplicate names, missing template placeholder)
3. Error handling works correctly
4. Results are properly structured and returned
"""

import time
from unittest.mock import patch

import pytest
import torch

from tests.test_utils import create_toy_task_class
from workrb.models.base import ModelInterface
from workrb.results import (
    BenchmarkMetadata,
    BenchmarkResults,
    MetricsResult,
    TaskResultMetadata,
    TaskResults,
)
from workrb.run import evaluate_multiple_models
from workrb.tasks import SkillMatch1kSkillSimilarityRanking
from workrb.tasks.abstract.base import DatasetSplit, Language
from workrb.types import ModelInputType


class ToyModel(ModelInterface):
    """Simple toy model for testing - no actual model loading required."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Toy model for testing: {self._name}"

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """Mock implementation - never called in these tests."""
        return torch.zeros(len(queries), len(targets))

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Mock implementation - never called in these tests."""
        return torch.zeros(len(texts), len(targets))

    @property
    def classification_label_space(self) -> list[str] | None:
        return None


def create_model_with_name(name: str):
    """Create a toy model with a custom name."""
    return ToyModel(name)


def create_mock_results(model_name: str, task_name: str) -> BenchmarkResults:
    """Create a mock BenchmarkResults object."""
    return BenchmarkResults(
        task_results={
            task_name: TaskResults(
                metadata=TaskResultMetadata(
                    task_group="skillsim",
                    task_type="ranking",
                    label_type="single_label",
                    description="Find similar skills using the SkillMatch-1K dataset",
                    split="val",
                ),
                language_results={
                    "en": MetricsResult(
                        evaluation_time=1.0,
                        metrics_dict={"map": 0.5},
                    )
                },
            )
        },
        metadata=BenchmarkMetadata(
            model_name=model_name,
            total_evaluation_time=1.0,
            timestamp=time.time(),
            num_tasks=1,
            languages=["en"],
            resumed_from_checkpoint=False,
        ),
        key_metrics_by_task_group={"skillsim": ["map"]},
    )


def test_evaluate_multiple_models_basic():
    """Test basic functionality of evaluate_multiple_models."""
    # Create real models with different names
    model1 = create_model_with_name("model1")
    model2 = create_model_with_name("model2")
    models = [model1, model2]

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])
    task_name = task.name

    # Mock the evaluate function
    with patch("workrb.run.evaluate") as mock_evaluate:
        # Set up return values for each model
        mock_evaluate.side_effect = [
            create_mock_results("model1", task_name),
            create_mock_results("model2", task_name),
        ]

        # Run evaluate_multiple_models
        results = evaluate_multiple_models(
            models=models,
            tasks=[task],
            output_folder_template="results/{model_name}",
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "model1" in results
        assert "model2" in results

        # Verify evaluate was called twice, once for each model
        assert mock_evaluate.call_count == 2

        # Verify evaluate was called with correct arguments
        calls = mock_evaluate.call_args_list
        assert calls[0][1]["model"] == model1
        assert calls[0][1]["tasks"] == [task]
        assert calls[0][1]["output_folder"] == "results/model1"

        assert calls[1][1]["model"] == model2
        assert calls[1][1]["tasks"] == [task]
        assert calls[1][1]["output_folder"] == "results/model2"

        # Verify results content
        assert results["model1"].metadata.model_name == "model1"
        assert results["model2"].metadata.model_name == "model2"


def test_evaluate_multiple_models_with_additional_kwargs():
    """Test evaluate_multiple_models passes additional kwargs to evaluate."""
    model = create_model_with_name("test_model")

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])
    task_name = task.name

    with patch("workrb.run.evaluate") as mock_evaluate:
        mock_evaluate.return_value = create_mock_results("test_model", task_name)

        results = evaluate_multiple_models(
            models=[model],
            tasks=[task],
            output_folder_template="results/{model_name}",
            description="Test benchmark",
            force_restart=True,
            metrics={task_name: ["map", "ndcg"]},
        )

        # Verify additional kwargs were passed
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["description"] == "Test benchmark"
        assert call_kwargs["force_restart"] is True
        assert call_kwargs["metrics"] == {task_name: ["map", "ndcg"]}


def test_evaluate_multiple_models_duplicate_names():
    """Test that evaluate_multiple_models raises error for duplicate model names."""
    model1 = create_model_with_name("duplicate_name")
    model2 = create_model_with_name("duplicate_name")

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])

    with pytest.raises(AssertionError, match="All models must have unique names"):
        evaluate_multiple_models(
            models=[model1, model2],
            tasks=[task],
            output_folder_template="results/{model_name}",
        )


def test_evaluate_multiple_models_missing_template_placeholder():
    """Test that evaluate_multiple_models raises error if template lacks {model_name}."""
    model = create_model_with_name("test_model")

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])

    with pytest.raises(AssertionError, match="Output folder template must contain {model_name}"):
        evaluate_multiple_models(
            models=[model],
            tasks=[task],
            output_folder_template="results/no_placeholder",
        )


def test_evaluate_multiple_models_error_handling():
    """Test that evaluate_multiple_models properly handles and re-raises errors."""
    model1 = create_model_with_name("model1")
    model2 = create_model_with_name("model2")

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])
    task_name = task.name

    with patch("workrb.run.evaluate") as mock_evaluate:
        # First model succeeds, second fails
        mock_evaluate.side_effect = [
            create_mock_results("model1", task_name),
            ValueError("Model evaluation failed"),
        ]

        # Should raise the error
        with pytest.raises(ValueError, match="Model evaluation failed"):
            evaluate_multiple_models(
                models=[model1, model2],
                tasks=[task],
                output_folder_template="results/{model_name}",
            )

        # Verify first model was called
        assert mock_evaluate.call_count == 2


def test_evaluate_multiple_models_output_folder_overrides_kwargs():
    """Test that output_folder in run_kwargs is properly overridden per model."""
    model1 = create_model_with_name("model1")
    model2 = create_model_with_name("model2")

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])
    task_name = task.name

    with patch("workrb.run.evaluate") as mock_evaluate:
        mock_evaluate.side_effect = [
            create_mock_results("model1", task_name),
            create_mock_results("model2", task_name),
        ]

        # Pass output_folder in kwargs (should be overridden)
        results = evaluate_multiple_models(
            models=[model1, model2],
            tasks=[task],
            output_folder_template="results/{model_name}",
            output_folder="should_be_overridden",
        )

        # Verify each model got its own output folder, not the one from kwargs
        calls = mock_evaluate.call_args_list
        assert calls[0][1]["output_folder"] == "results/model1"
        assert calls[1][1]["output_folder"] == "results/model2"
        assert calls[0][1]["output_folder"] != "should_be_overridden"
        assert calls[1][1]["output_folder"] != "should_be_overridden"


def test_evaluate_multiple_models_single_model():
    """Test evaluate_multiple_models with a single model."""
    model = create_model_with_name("single_model")

    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])
    task_name = task.name

    with patch("workrb.run.evaluate") as mock_evaluate:
        mock_evaluate.return_value = create_mock_results("single_model", task_name)

        results = evaluate_multiple_models(
            models=[model],
            tasks=[task],
            output_folder_template="results/{model_name}",
        )

        assert len(results) == 1
        assert "single_model" in results
        assert mock_evaluate.call_count == 1


def test_evaluate_multiple_models_empty_models_list():
    """Test evaluate_multiple_models with empty models list raises AssertionError."""
    # Create toy task from real task
    ToyTask = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)
    task = ToyTask(split=DatasetSplit.VAL, languages=[Language.EN])

    with patch("workrb.run.evaluate") as mock_evaluate:
        with pytest.raises(AssertionError) as excinfo:
            evaluate_multiple_models(
                models=[],
                tasks=[task],
                output_folder_template="results/{model_name}",
            )
        assert "At least one model must be provided to evaluate multiple models." in str(
            excinfo.value
        )
        assert mock_evaluate.call_count == 0
