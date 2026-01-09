import pytest  # noqa: D100
import torch

from workrb.models.bi_encoder import ConTeXTMatchModel
from workrb.tasks import TechSkillExtractRanking
from workrb.tasks.abstract.base import DatasetSplit, Language
from workrb.types import ModelInputType


class TestConTeXTMatchModelLoading:
    """Test that ConTeXTMatchModel can be correctly loaded and initialized."""

    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = ConTeXTMatchModel()
        assert model is not None
        assert model.base_model_name == "TechWolf/ConTeXT-Skill-Extraction-base"
        assert model.temperature == 1.0
        assert model.model is not None
        # Model should be in eval mode (not training)
        assert not model.model.training

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        custom_model_name = "TechWolf/ConTeXT-Skill-Extraction-base"
        custom_temperature = 0.5
        model = ConTeXTMatchModel(model_name=custom_model_name, temperature=custom_temperature)
        assert model.base_model_name == custom_model_name
        assert model.temperature == custom_temperature

    def test_model_properties(self):
        """Test model name and description properties."""
        model = ConTeXTMatchModel()
        name = model.name
        description = model.description
        citation = model.citation

        assert isinstance(name, str)
        assert len(name) > 0
        assert "ConTeXT" in name or "Skill" in name

        assert isinstance(description, str)
        assert len(description) > 0

        assert citation is not None
        assert isinstance(citation, str)
        assert "contextmatch" in citation.lower() or "ConTeXT" in citation

    def test_model_classification_label_space(self):
        """Test that classification_label_space returns None."""
        model = ConTeXTMatchModel()
        assert model.classification_label_space is None


class TestConTeXTMatchModelUsage:
    """Test that ConTeXTMatchModel can be used for ranking and classification."""

    def test_compute_rankings_basic(self):
        """Test basic ranking computation."""
        model = ConTeXTMatchModel()
        queries = ["software engineer", "data scientist"]
        targets = ["Python programming", "machine learning", "statistics"]

        scores = model._compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Check output shape: (n_queries, n_targets)
        assert scores.shape == (len(queries), len(targets))
        assert isinstance(scores, torch.Tensor)

        # Scores should be finite
        assert torch.isfinite(scores).all()

    def test_compute_classification_basic(self):
        """Test basic classification computation."""
        model = ConTeXTMatchModel()
        texts = ["software engineer", "data scientist"]
        targets = ["Python programming", "machine learning", "statistics"]

        scores = model._compute_classification(
            texts=texts,
            targets=targets,
            input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Check output shape: (n_texts, n_targets)
        assert scores.shape == (len(texts), len(targets))
        assert isinstance(scores, torch.Tensor)

        # Scores should be finite
        assert torch.isfinite(scores).all()

    def test_compute_classification_default_target_type(self):
        """Test classification with default target_input_type."""
        model = ConTeXTMatchModel()
        texts = ["software engineer", "data scientist"]
        targets = ["Python programming", "machine learning"]

        scores = model._compute_classification(
            texts=texts,
            targets=targets,
            input_type=ModelInputType.JOB_TITLE,
        )

        assert scores.shape == (len(texts), len(targets))
        assert torch.isfinite(scores).all()


@pytest.mark.model_performance
class TestConTeXTMatchModelTechSkillExtraction:
    """Test ConTeXTMatchModel performance on TECH skill extraction test set."""

    def test_tech_skill_extraction_benchmark_metrics(self):
        """
        Test that ConTeXTMatchModel achieves results close to paper-reported metrics.

        Paper reported on TECH skill extraction test set:
        - Mean Reciprocal Rank (MRR): 0.632
        - R-Precision@1 (RP@1): 50.99%
        - R-Precision@5 (RP@5): 63.98%
        - R-Precision@10 (RP@10): 73.99%
        """
        # Initialize model and task
        model = ConTeXTMatchModel()
        task = TechSkillExtractRanking(split=DatasetSplit.TEST, languages=[Language.EN])

        # Evaluate model on the task with the metrics from the paper
        metrics = ["mrr", "rp@1", "rp@5", "rp@10"]
        results = task.evaluate(model=model, metrics=metrics, language=Language.EN)

        # Paper-reported values (RP metrics are percentages, convert to decimals)
        expected_mrr = 0.632
        expected_rp1 = 50.99 / 100.0  # Convert percentage to decimal
        expected_rp5 = 63.98 / 100.0
        expected_rp10 = 73.99 / 100.0

        # Allow a little tolerance for floating point precision
        mrr_tolerance = 0.05
        rp_tolerance = 0.05

        # Check MRR
        actual_mrr = results["mrr"]
        assert actual_mrr == pytest.approx(expected_mrr, abs=mrr_tolerance), (
            f"MRR: expected {expected_mrr:.3f}, got {actual_mrr:.3f}"
        )

        # Check RP@1
        actual_rp1 = results["rp@1"]
        assert actual_rp1 == pytest.approx(expected_rp1, abs=rp_tolerance), (
            f"RP@1: expected {expected_rp1:.3f}, got {actual_rp1:.3f}"
        )

        # Check RP@5
        actual_rp5 = results["rp@5"]
        assert actual_rp5 == pytest.approx(expected_rp5, abs=rp_tolerance), (
            f"RP@5: expected {expected_rp5:.3f}, got {actual_rp5:.3f}"
        )

        # Check RP@10
        actual_rp10 = results["rp@10"]
        assert actual_rp10 == pytest.approx(expected_rp10, abs=rp_tolerance), (
            f"RP@10: expected {expected_rp10:.3f}, got {actual_rp10:.3f}"
        )
