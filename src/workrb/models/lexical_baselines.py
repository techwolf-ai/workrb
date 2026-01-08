"""Lexical baseline models for ranking tasks in WorkRB.

These models provide fast, CPU-based baselines for ranking tasks. They are useful for
establishing performance bounds and enabling rapid iteration without GPU dependencies.
"""

import random
import unicodedata

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from workrb.models.base import ModelInterface
from workrb.registry import register_model
from workrb.types import ModelInputType


@register_model()
class BM25Model(ModelInterface):
    """BM25 Okapi probabilistic ranking baseline.

    Parameters
    ----------
    lowercase : bool, default=True
        Whether to lowercase texts before computing scores.

    Example
    -------
        >>> model = BM25Model()
        >>> queries = ["python developer", "data scientist"]
        >>> targets = ["python programming", "machine learning", "software engineer"]
        >>> scores = model.compute_rankings(queries, targets, None, None)
        >>> scores.shape
        torch.Size([2, 3])
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    @property
    def name(self) -> str:
        """Return the model name."""
        return "BM25"

    @property
    def description(self) -> str:
        """Return the model description."""
        return "BM25 Okapi probabilistic ranking baseline"

    @property
    def classification_label_space(self) -> list[str] | None:
        """Return None as this model has no fixed label space."""
        return None

    def _preprocess(self, text: str) -> str:
        """Preprocess text by normalizing Unicode and optionally lowercasing."""
        text = unicodedata.normalize("NFKD", text)
        if self.lowercase:
            return text.lower()
        return text

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute BM25 ranking scores.

        Parameters
        ----------
        queries : list[str]
            List of query texts
        targets : list[str]
            List of target texts (corpus)
        query_input_type : ModelInputType | None
            Type of query input (ignored by this model)
        target_input_type : ModelInputType | None
            Type of target input (ignored by this model)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_queries, n_targets) with BM25 scores
        """
        # Preprocess and tokenize corpus
        tokenized_corpus = [self._preprocess(target).split() for target in targets]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        # Compute scores for each query
        scores = []
        for query in queries:
            preprocessed_query = self._preprocess(query)
            tokenized_query = preprocessed_query.split()
            query_scores = bm25.get_scores(tokenized_query)
            scores.append(query_scores)

        scores_array = np.array(scores)
        return torch.tensor(scores_array, dtype=torch.float32)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores by ranking texts against target labels.

        Parameters
        ----------
        texts : list[str]
            List of input texts to classify
        targets : list[str]
            List of target class labels (as text)
        input_type : ModelInputType
            Type of input
        target_input_type : ModelInputType | None
            Type of target. If None, uses input_type.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_texts, n_classes) with BM25 scores
        """
        if target_input_type is None:
            target_input_type = input_type

        return self._compute_rankings(texts, targets, input_type, target_input_type)


@register_model()
class TfIdfModel(ModelInterface):
    """TF-IDF baseline with configurable tokenization.

    Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization followed by
    cosine similarity. Supports both word-level and character n-gram tokenization.

    Parameters
    ----------
    lowercase : bool, default=True
        Whether to lowercase texts before computing scores.
    tokenization : str, default="word"
        Tokenization strategy. Options:
        - "word": Word-level tokenization (default)
        - "char": Character n-gram tokenization (1-3 grams)
    """

    def __init__(self, lowercase: bool = True, tokenization: str = "word"):
        if tokenization not in ["word", "char"]:
            raise ValueError(f"Invalid tokenization: {tokenization}. Must be 'word' or 'char'.")

        self.lowercase = lowercase
        self.tokenization = tokenization

    @property
    def name(self) -> str:
        """Return the model name."""
        return f"TfIdf-{self.tokenization}"

    @property
    def description(self) -> str:
        """Return the model description."""
        if self.tokenization == "word":
            return "TF-IDF baseline with word-level tokenization"
        return "TF-IDF baseline with character n-gram tokenization (1-3)"

    @property
    def classification_label_space(self) -> list[str] | None:
        """Return None as this model has no fixed label space."""
        return None

    def _preprocess(self, text: str) -> str:
        """Preprocess text by normalizing Unicode and optionally lowercasing."""
        text = unicodedata.normalize("NFKD", text)
        if self.lowercase:
            return text.lower()
        return text

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute TF-IDF ranking scores.

        Parameters
        ----------
        queries : list[str]
            List of query texts
        targets : list[str]
            List of target texts (corpus)
        query_input_type : ModelInputType | None
            Type of query input (ignored by this model)
        target_input_type : ModelInputType | None
            Type of target input (ignored by this model)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_queries, n_targets) with cosine similarity scores
        """
        # Preprocess corpus
        processed_corpus = [self._preprocess(target) for target in targets]

        # Configure vectorizer based on tokenization strategy
        if self.tokenization == "char":
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        else:
            vectorizer = TfidfVectorizer()

        # Fit vectorizer on corpus
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)

        # Compute scores for each query
        scores = []
        for query in queries:
            preprocessed_query = self._preprocess(query)
            query_vector = vectorizer.transform([preprocessed_query])
            query_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
            scores.append(query_scores)

        scores_array = np.array(scores)
        return torch.tensor(scores_array, dtype=torch.float32)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores by ranking texts against target labels.

        Parameters
        ----------
        texts : list[str]
            List of input texts to classify
        targets : list[str]
            List of target class labels (as text)
        input_type : ModelInputType
            Type of input
        target_input_type : ModelInputType | None
            Type of target. If None, uses input_type.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_texts, n_classes) with cosine similarity scores
        """
        if target_input_type is None:
            target_input_type = input_type

        return self._compute_rankings(texts, targets, input_type, target_input_type)


@register_model()
class EditDistanceModel(ModelInterface):
    """Edit distance (Levenshtein ratio) baseline.

    Computes the Levenshtein ratio between query and target strings. The ratio is
    normalized to [0, 100] where 100 indicates identical strings. This model is
    effective for near-exact matches and normalization tasks.

    Parameters
    ----------
    lowercase : bool, default=True
        Whether to lowercase texts before computing scores.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    @property
    def name(self) -> str:
        """Return the model name."""
        return "EditDistance"

    @property
    def description(self) -> str:
        """Return the model description."""
        return "Levenshtein ratio baseline for string similarity"

    @property
    def classification_label_space(self) -> list[str] | None:
        """Return None as this model has no fixed label space."""
        return None

    def _preprocess(self, text: str) -> str:
        """Preprocess text by normalizing Unicode and optionally lowercasing."""
        text = unicodedata.normalize("NFKD", text)
        if self.lowercase:
            return text.lower()
        return text

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute edit distance ranking scores.

        Parameters
        ----------
        queries : list[str]
            List of query texts
        targets : list[str]
            List of target texts
        query_input_type : ModelInputType | None
            Type of query input (ignored by this model)
        target_input_type : ModelInputType | None
            Type of target input (ignored by this model)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_queries, n_targets) with Levenshtein ratio scores [0-100]
        """
        scores = []
        for query in queries:
            query_preprocessed = self._preprocess(query)
            query_scores = []
            for target in targets:
                target_preprocessed = self._preprocess(target)
                score = fuzz.ratio(query_preprocessed, target_preprocessed)
                query_scores.append(score)
            scores.append(query_scores)

        return torch.tensor(scores, dtype=torch.float32)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores by ranking texts against target labels.

        Parameters
        ----------
        texts : list[str]
            List of input texts to classify
        targets : list[str]
            List of target class labels (as text)
        input_type : ModelInputType
            Type of input
        target_input_type : ModelInputType | None
            Type of target. If None, uses input_type.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_texts, n_classes) with Levenshtein ratio scores
        """
        if target_input_type is None:
            target_input_type = input_type

        return self._compute_rankings(texts, targets, input_type, target_input_type)


@register_model()
class RandomRankingModel(ModelInterface):
    """Random ranking baseline for sanity checking.

    Generates random scores between 0 and 1 for all query-target pairs. This serves
    as a control baseline to verify that evaluation metrics and pipelines are working
    correctly. Any reasonable model should significantly outperform random scoring.

    Parameters
    ----------
    seed : int | None, default=None
        Random seed for reproducibility. If None, results will vary between runs.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "RandomRanking"

    @property
    def description(self) -> str:
        """Return the model description."""
        return "Random ranking baseline for sanity checking"

    @property
    def classification_label_space(self) -> list[str] | None:
        """Return None as this model has no fixed label space."""
        return None

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute random ranking scores.

        Parameters
        ----------
        queries : list[str]
            List of query texts
        targets : list[str]
            List of target texts
        query_input_type : ModelInputType | None
            Type of query input (ignored by this model)
        target_input_type : ModelInputType | None
            Type of target input (ignored by this model)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_queries, n_targets) with random scores [0-1]
        """
        scores = []
        for _ in queries:
            query_scores = [random.random() for _ in targets]
            scores.append(query_scores)

        return torch.tensor(scores, dtype=torch.float32)

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute random classification scores.

        Parameters
        ----------
        texts : list[str]
            List of input texts to classify
        targets : list[str]
            List of target class labels (as text)
        input_type : ModelInputType
            Type of input
        target_input_type : ModelInputType | None
            Type of target. If None, uses input_type.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_texts, n_classes) with random scores [0-1]
        """
        if target_input_type is None:
            target_input_type = input_type

        return self._compute_rankings(texts, targets, input_type, target_input_type)
