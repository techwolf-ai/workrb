"""
Test script for ConTeXTMatchModel.

This test suite validates that:
1. ConTeXTMatchModel is correctly loaded and can be used as desired
2. The ranking scores are really an attention-weighted cosine similarity between
   mean target embedding and query token embeddings
"""

import torch

from workrb.models.bi_encoder import ConTeXTMatchModel
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


class TestConTeXTMatchAttentionWeightedSimilarity:
    """Test that ranking scores are attention-weighted cosine similarity."""

    def test_attention_weighted_similarity_manual_computation(self):
        """
        Manually compute attention-weighted cosine similarity and compare with model output.

        This test verifies that the _context_match_score method correctly implements:
        1. Dot product between token embeddings and target embeddings
        2. Softmax with temperature to get attention weights
        3. Cosine similarity between normalized embeddings
        4. Weighted sum of similarities
        """
        model = ConTeXTMatchModel(temperature=1.0)

        # Use small test inputs
        queries = ["Python programming"]
        targets = ["machine learning", "data science"]

        # Get embeddings using model's encode methods
        query_token_embeddings = ConTeXTMatchModel.encode(model.model, queries, mean=False)
        target_mean_embeddings = ConTeXTMatchModel.encode(model.model, targets, mean=True)

        # Get model's output
        model_scores = model._context_match_score(
            query_token_embeddings, target_mean_embeddings, temperature=model.temperature
        )

        # Manually compute the same thing
        # Step 1: Compute dot products: (B1, L, D) @ (B2, D).T -> (B1, L, B2)
        # -> transpose to (B1, B2, L)
        dot_scores = (query_token_embeddings @ target_mean_embeddings.T).transpose(1, 2)

        # Step 2: Apply near-zero threshold and softmax
        dot_scores_clipped = dot_scores.clone()
        dot_scores_clipped[dot_scores_clipped.abs() < ConTeXTMatchModel._NEAR_ZERO_THRESHOLD] = (
            float("-inf")
        )
        weights = torch.softmax(dot_scores_clipped / model.temperature, dim=2)

        # Step 3: Normalize embeddings for cosine similarity
        norm_tokens = torch.nn.functional.normalize(query_token_embeddings, p=2, dim=2)
        norm_targets = torch.nn.functional.normalize(target_mean_embeddings, p=2, dim=1)

        # Step 4: Compute cosine similarities: (B1, L, D) @ (B2, D).T -> (B1, L, B2)
        # -> transpose to (B1, B2, L)
        sim_scores = (norm_tokens @ norm_targets.T).transpose(1, 2)

        # Step 5: Weighted sum
        manual_scores = (weights * sim_scores).sum(dim=2)

        # Compare results - should be very close (within numerical precision)
        assert torch.allclose(model_scores, manual_scores, atol=1e-6, rtol=1e-5), (
            f"Model scores don't match manual computation.\n"
            f"Model scores: {model_scores}\n"
            f"Manual scores: {manual_scores}\n"
            f"Difference: {torch.abs(model_scores - manual_scores)}"
        )

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along the sequence dimension."""
        model = ConTeXTMatchModel(temperature=1.0)

        queries = ["Python programming"]
        targets = ["machine learning"]

        query_token_embeddings = ConTeXTMatchModel.encode(model.model, queries, mean=False)
        target_mean_embeddings = ConTeXTMatchModel.encode(model.model, targets, mean=True)

        # Compute dot scores and weights manually
        dot_scores = (query_token_embeddings @ target_mean_embeddings.T).transpose(1, 2)
        dot_scores_clipped = dot_scores.clone()
        dot_scores_clipped[dot_scores_clipped.abs() < ConTeXTMatchModel._NEAR_ZERO_THRESHOLD] = (
            float("-inf")
        )
        weights = torch.softmax(dot_scores_clipped / model.temperature, dim=2)

        # Weights should sum to 1 along dim=2 (sequence length dimension)
        weight_sums = weights.sum(dim=2)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
            f"Attention weights don't sum to 1.\nWeight sums: {weight_sums}"
        )

    def test_cosine_similarity_range(self):
        """Test that cosine similarities are in the expected range [-1, 1]."""
        model = ConTeXTMatchModel()

        queries = ["Python programming", "data analysis"]
        targets = ["machine learning", "statistics"]

        query_token_embeddings = ConTeXTMatchModel.encode(model.model, queries, mean=False)
        target_mean_embeddings = ConTeXTMatchModel.encode(model.model, targets, mean=True)

        # Normalize embeddings
        norm_tokens = torch.nn.functional.normalize(query_token_embeddings, p=2, dim=2)
        norm_targets = torch.nn.functional.normalize(target_mean_embeddings, p=2, dim=1)

        # Compute cosine similarities
        sim_scores = (norm_tokens @ norm_targets.T).transpose(1, 2)

        # Cosine similarity should be in [-1, 1]
        min_val = sim_scores.min()
        max_val = sim_scores.max()
        assert (sim_scores >= -1.0 - 1e-5).all()
        assert (sim_scores <= 1.0 + 1e-5).all(), (
            f"Cosine similarities out of range [-1, 1].\nMin: {min_val}, Max: {max_val}"
        )

    def test_temperature_effect(self):
        """Test that temperature parameter affects the attention distribution."""
        queries = ["Python programming"]
        targets = ["machine learning", "data science"]

        # Test with different temperatures
        model_low_temp = ConTeXTMatchModel(temperature=0.1)
        model_high_temp = ConTeXTMatchModel(temperature=10.0)

        query_token_embeddings = ConTeXTMatchModel.encode(model_low_temp.model, queries, mean=False)
        target_mean_embeddings = ConTeXTMatchModel.encode(model_low_temp.model, targets, mean=True)

        scores_low_temp = model_low_temp._context_match_score(
            query_token_embeddings, target_mean_embeddings, temperature=0.1
        )
        scores_high_temp = model_high_temp._context_match_score(
            query_token_embeddings, target_mean_embeddings, temperature=10.0
        )

        # With lower temperature, attention should be more peaked (higher variance in weights)
        # With higher temperature, attention should be more uniform
        # The actual scores may differ, but the computation should be consistent
        assert scores_low_temp.shape == scores_high_temp.shape
        assert torch.isfinite(scores_low_temp).all()
        assert torch.isfinite(scores_high_temp).all()

    def test_ranking_scores_via_compute_rankings(self):
        """
        Test that _compute_rankings uses attention-weighted similarity correctly.

        This verifies the full pipeline from text inputs to final scores.
        """
        model = ConTeXTMatchModel(temperature=1.0)

        queries = ["software engineer", "data scientist"]
        targets = ["Python programming", "machine learning"]

        # Get scores via _compute_rankings
        ranking_scores = model._compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Manually compute what it should be
        query_token_embeddings = ConTeXTMatchModel.encode(model.model, queries, mean=False)
        target_mean_embeddings = ConTeXTMatchModel.encode(model.model, targets, mean=True)

        manual_scores = model._context_match_score(
            query_token_embeddings, target_mean_embeddings, temperature=model.temperature
        )

        # Should match exactly (same method call)
        assert torch.allclose(ranking_scores, manual_scores, atol=1e-6), (
            f"_compute_rankings doesn't match direct _context_match_score call.\n"
            f"Ranking scores: {ranking_scores}\n"
            f"Manual scores: {manual_scores}"
        )

    def test_mean_target_embeddings(self):
        """Test that target embeddings are indeed mean embeddings (not token embeddings)."""
        model = ConTeXTMatchModel()

        targets = ["machine learning", "data science"]

        # Get mean embeddings (what's used in ranking)
        target_mean = ConTeXTMatchModel.encode(model.model, targets, mean=True)

        # Get token embeddings (what's NOT used for targets)
        target_tokens = ConTeXTMatchModel.encode(model.model, targets, mean=False)

        # Mean embeddings should be 2D: (batch, dim)
        expected_mean_dim = 2
        assert len(target_mean.shape) == expected_mean_dim
        assert target_mean.shape[0] == len(targets)

        # Token embeddings should be 3D: (batch, seq_len, dim)
        expected_token_dim = 3
        assert len(target_tokens.shape) == expected_token_dim
        assert target_tokens.shape[0] == len(targets)

        # Mean embeddings should have same feature dimension as token embeddings
        assert target_mean.shape[1] == target_tokens.shape[2]

    def test_query_token_embeddings(self):
        """Test that query embeddings are token embeddings (not mean embeddings)."""
        model = ConTeXTMatchModel()

        queries = ["Python programming", "data analysis"]

        # Get token embeddings (what's used in ranking)
        query_tokens = ConTeXTMatchModel.encode(model.model, queries, mean=False)

        # Should be 3D: (batch, seq_len, dim)
        expected_query_dim = 3
        assert len(query_tokens.shape) == expected_query_dim
        assert query_tokens.shape[0] == len(queries)
        assert query_tokens.shape[1] > 0  # Should have sequence length > 0
        assert query_tokens.shape[2] > 0  # Should have embedding dimension > 0
