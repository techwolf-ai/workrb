"""Unit tests for lexical baseline models."""

import torch

from workrb.models.lexical_baselines import (
    BM25Model,
    EditDistanceModel,
    RandomRankingModel,
    TfIdfModel,
)
from workrb.types import ModelInputType


class TestBM25Model:
    """Test BM25Model initialization and functionality."""

    def test_model_initialization_default(self):
        """Test BM25Model initialization with default parameters."""
        model = BM25Model()
        assert model is not None
        assert model.lowercase is True

    def test_model_initialization_custom_params(self):
        """Test BM25Model initialization with custom parameters."""
        model = BM25Model(lowercase=False)
        assert model.lowercase is False

    def test_model_properties(self):
        """Test BM25Model name and description properties."""
        model = BM25Model()
        assert model.name == "BM25-lower"
        assert isinstance(model.description, str)
        assert len(model.description) > 0
        assert model.classification_label_space is None

    def test_model_name_cased(self):
        """Test BM25Model name with lowercase=False."""
        model = BM25Model(lowercase=False)
        assert model.name == "BM25-cased"

    def test_compute_rankings_basic(self):
        """Test basic BM25 ranking computation."""
        model = BM25Model()
        queries = ["python developer", "data scientist"]
        targets = ["python programming", "machine learning", "java developer"]

        scores = model._compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Check output shape
        assert scores.shape == (len(queries), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert scores.dtype == torch.float32

        # Scores should be finite and non-negative (BM25 scores are >= 0)
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all()

    def test_compute_rankings_lowercase_sensitivity(self):
        """Test that lowercase parameter affects preprocessing."""
        model_lower = BM25Model(lowercase=True)
        model_no_lower = BM25Model(lowercase=False)

        # Test the preprocessing method directly
        text = "Python Developer"

        assert model_lower._preprocess(text) == "python developer"
        assert model_no_lower._preprocess(text) == "Python Developer"

    def test_compute_classification(self):
        """Test BM25 classification computation."""
        model = BM25Model()
        texts = ["python developer", "data scientist"]
        targets = ["python", "machine learning", "statistics"]

        scores = model._compute_classification(
            texts=texts,
            targets=targets,
            input_type=ModelInputType.JOB_TITLE,
        )

        assert scores.shape == (len(texts), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert torch.isfinite(scores).all()


class TestTfIdfModel:
    """Test TfIdfModel initialization and functionality."""

    def test_model_initialization_default(self):
        """Test TfIdfModel initialization with default parameters."""
        model = TfIdfModel()
        assert model is not None
        assert model.lowercase is True
        assert model.tokenization == "word"

    def test_model_initialization_word_tokenization(self):
        """Test TfIdfModel initialization with word tokenization."""
        model = TfIdfModel(lowercase=True, tokenization="word")
        assert model.lowercase is True
        assert model.tokenization == "word"

    def test_model_initialization_char_tokenization(self):
        """Test TfIdfModel initialization with char tokenization."""
        model = TfIdfModel(lowercase=False, tokenization="char")
        assert model.lowercase is False
        assert model.tokenization == "char"

    def test_model_initialization_invalid_tokenization(self):
        """Test that invalid tokenization raises ValueError."""
        try:
            TfIdfModel(tokenization="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid tokenization" in str(e)
            assert "Must be 'word' or 'char'" in str(e)

    def test_model_properties_word(self):
        """Test TfIdfModel properties with word tokenization."""
        model = TfIdfModel(tokenization="word")
        assert model.name == "TfIdf-word-lower"
        assert "word-level" in model.description
        assert model.classification_label_space is None

    def test_model_properties_char(self):
        """Test TfIdfModel properties with char tokenization."""
        model = TfIdfModel(tokenization="char")
        assert model.name == "TfIdf-char-lower"
        assert "character n-gram" in model.description
        assert model.classification_label_space is None

    def test_model_name_cased(self):
        """Test TfIdfModel name with lowercase=False."""
        model_word = TfIdfModel(lowercase=False, tokenization="word")
        assert model_word.name == "TfIdf-word-cased"

        model_char = TfIdfModel(lowercase=False, tokenization="char")
        assert model_char.name == "TfIdf-char-cased"

    def test_compute_rankings_word_tokenization(self):
        """Test TF-IDF ranking with word tokenization."""
        model = TfIdfModel(tokenization="word")
        queries = ["python developer", "data scientist"]
        targets = ["python programming", "machine learning", "java developer"]

        scores = model._compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Check output shape
        assert scores.shape == (len(queries), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert scores.dtype == torch.float32

        # Cosine similarity scores should be in [-1, 1], but typically [0, 1] for TF-IDF
        assert torch.isfinite(scores).all()
        assert (scores >= -1).all() and (scores <= 1).all()

    def test_compute_rankings_char_tokenization(self):
        """Test TF-IDF ranking with character n-gram tokenization."""
        model = TfIdfModel(tokenization="char")
        queries = ["python", "java"]
        targets = ["python", "pithon", "java"]  # pithon is similar to python

        scores = model._compute_rankings(queries, targets)

        # Check output shape
        assert scores.shape == (len(queries), len(targets))
        assert isinstance(scores, torch.Tensor)

        # Character n-grams should give high similarity for "python" vs "pithon"
        # query[0]="python" should have higher score with target[0]="python" than target[2]="java"
        assert scores[0, 0].item() > scores[0, 2].item()
        # query[0]="python" should have higher score with target[1]="pithon" than target[2]="java"
        assert scores[0, 1].item() > scores[0, 2].item()

    def test_compute_rankings_word_vs_char(self):
        """Test that word and char tokenization produce different results."""
        queries = ["software engineer"]
        targets = ["software development", "sofware engineering"]  # typo in second

        # Word tokenization
        model_word = TfIdfModel(tokenization="word")
        scores_word = model_word._compute_rankings(queries, targets)

        # Character tokenization
        model_char = TfIdfModel(tokenization="char")
        scores_char = model_char._compute_rankings(queries, targets)

        # Character n-grams should handle the typo better
        # So the relative scores should be different
        assert not torch.allclose(scores_word, scores_char, atol=0.01)

    def test_compute_classification(self):
        """Test TF-IDF classification computation."""
        model = TfIdfModel()
        texts = ["python developer", "data scientist"]
        targets = ["python", "machine learning", "statistics"]

        scores = model._compute_classification(
            texts=texts,
            targets=targets,
            input_type=ModelInputType.JOB_TITLE,
        )

        assert scores.shape == (len(texts), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert torch.isfinite(scores).all()


class TestEditDistanceModel:
    """Test EditDistanceModel initialization and functionality."""

    def test_model_initialization_default(self):
        """Test EditDistanceModel initialization with default parameters."""
        model = EditDistanceModel()
        assert model is not None
        assert model.lowercase is True

    def test_model_initialization_custom_params(self):
        """Test EditDistanceModel initialization with custom parameters."""
        model = EditDistanceModel(lowercase=False)
        assert model.lowercase is False

    def test_model_properties(self):
        """Test EditDistanceModel name and description properties."""
        model = EditDistanceModel()
        assert model.name == "EditDistance-lower"
        assert isinstance(model.description, str)
        assert "Levenshtein" in model.description
        assert model.classification_label_space is None

    def test_model_name_cased(self):
        """Test EditDistanceModel name with lowercase=False."""
        model = EditDistanceModel(lowercase=False)
        assert model.name == "EditDistance-cased"

    def test_compute_rankings_basic(self):
        """Test basic edit distance ranking computation."""
        model = EditDistanceModel()
        queries = ["python developer", "data scientist"]
        targets = ["python developer", "python developper", "java engineer"]

        scores = model._compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Check output shape
        assert scores.shape == (len(queries), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert scores.dtype == torch.float32

        # Levenshtein ratio scores are in [0, 100]
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all() and (scores <= 100).all()

        # Exact match should have score 100
        assert scores[0, 0].item() == 100.0

        # Similar strings should have high scores
        assert scores[0, 1].item() > 80  # "python developer" vs "python developper"

    def test_compute_rankings_exact_matches(self):
        """Test that exact matches get score of 100."""
        model = EditDistanceModel()
        queries = ["test", "example"]
        targets = ["test", "example", "different"]

        scores = model._compute_rankings(queries, targets)

        # Exact matches should be 100
        assert scores[0, 0].item() == 100.0  # "test" vs "test"
        assert scores[1, 1].item() == 100.0  # "example" vs "example"

    def test_compute_rankings_lowercase_sensitivity(self):
        """Test that lowercase parameter affects edit distance scores."""
        queries = ["Python"]
        targets = ["python", "PYTHON"]

        # With lowercase=True (default)
        model_lower = EditDistanceModel(lowercase=True)
        scores_lower = model_lower._compute_rankings(queries, targets)

        # Both should be exact matches after lowercasing
        assert scores_lower[0, 0].item() == 100.0
        assert scores_lower[0, 1].item() == 100.0

        # With lowercase=False
        model_no_lower = EditDistanceModel(lowercase=False)
        scores_no_lower = model_no_lower._compute_rankings(queries, targets)

        # Should not be exact matches without lowercasing
        assert scores_no_lower[0, 0].item() < 100.0
        assert scores_no_lower[0, 1].item() < 100.0

    def test_compute_classification(self):
        """Test edit distance classification computation."""
        model = EditDistanceModel()
        texts = ["python", "java"]
        targets = ["python", "javascript", "ruby"]

        scores = model._compute_classification(
            texts=texts,
            targets=targets,
            input_type=ModelInputType.SKILL_NAME,
        )

        assert scores.shape == (len(texts), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all() and (scores <= 100).all()


class TestRandomRankingModel:
    """Test RandomRankingModel initialization and functionality."""

    def test_model_initialization_default(self):
        """Test RandomRankingModel initialization with default parameters."""
        model = RandomRankingModel()
        assert model is not None
        assert model.seed is None

    def test_model_initialization_with_seed(self):
        """Test RandomRankingModel initialization with seed."""
        model = RandomRankingModel(seed=42)
        assert model.seed == 42

    def test_model_properties(self):
        """Test RandomRankingModel name and description properties."""
        model = RandomRankingModel()
        assert model.name == "RandomRanking"
        assert isinstance(model.description, str)
        assert "random" in model.description.lower() or "Random" in model.description
        assert model.classification_label_space is None

    def test_compute_rankings_basic(self):
        """Test basic random ranking computation."""
        model = RandomRankingModel(seed=42)
        queries = ["python developer", "data scientist"]
        targets = ["python programming", "machine learning", "java developer"]

        scores = model._compute_rankings(
            queries=queries,
            targets=targets,
            query_input_type=ModelInputType.JOB_TITLE,
            target_input_type=ModelInputType.SKILL_NAME,
        )

        # Check output shape
        assert scores.shape == (len(queries), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert scores.dtype == torch.float32

        # Random scores should be in [0, 1]
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_compute_rankings_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        queries = ["test query"]
        targets = ["target1", "target2", "target3"]

        # First run with seed
        model1 = RandomRankingModel(seed=12345)
        scores1 = model1._compute_rankings(queries, targets)

        # Second run with same seed
        model2 = RandomRankingModel(seed=12345)
        scores2 = model2._compute_rankings(queries, targets)

        # Should produce identical results
        assert torch.allclose(scores1, scores2)

    def test_compute_rankings_different_without_seed(self):
        """Test that different seeds produce different results."""
        queries = ["test query"]
        targets = ["target1", "target2", "target3"]

        # Run with different seeds
        model1 = RandomRankingModel(seed=1)
        scores1 = model1._compute_rankings(queries, targets)

        model2 = RandomRankingModel(seed=2)
        scores2 = model2._compute_rankings(queries, targets)

        # Should produce different results
        assert not torch.allclose(scores1, scores2)

    def test_compute_classification(self):
        """Test random classification computation."""
        model = RandomRankingModel(seed=42)
        texts = ["python developer", "data scientist"]
        targets = ["python", "machine learning", "statistics"]

        scores = model._compute_classification(
            texts=texts,
            targets=targets,
            input_type=ModelInputType.JOB_TITLE,
        )

        assert scores.shape == (len(texts), len(targets))
        assert isinstance(scores, torch.Tensor)
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all() and (scores <= 1).all()
