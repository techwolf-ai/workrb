"""Classification model wrapper for WorkRB with proper label space tracking.

This module provides an example of how to create a classification model that
properly tracks its label space, similar to how timm stores ImageNet class names.
"""

import logging

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from workrb.data.esco import ESCO
from workrb.models.base import ModelInterface
from workrb.registry import register_model
from workrb.tasks.abstract.base import Language
from workrb.types import ModelInputType

logger = logging.getLogger(__name__)


@register_model()
class RndESCOClassificationModel(ModelInterface):
    """
    Multi-label classification model with random prediction head for ESCO, to serve as a baseline.

    This model wraps a sentence transformer encoder with a random classification head.
    Critically, it tracks the exact label space and order used during training,
    ensuring predictions align correctly with task labels during evaluation.

    Example usage:
        >>> from workrb.types import ModelInputType
        >>> texts = ["Job title 1", "Job title 2", "Job title 3"]
        >>> model = RndESCOClassificationModel()
        >>> label_space = model.classification_label_space
        >>> predictions = model.compute_classification(texts, label_space, ModelInputType.JOB_TITLE)
    """

    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        label_space: list[str] | None = None,
    ):
        """
        Initialize classification model with label space tracking.

        Args:
            base_model_name: Name of the base sentence transformer model
            label_space: Ordered list of label names (e.g., skill names).
                        This MUST match the task's label space exactly.
            hidden_dim: Hidden dimension for classification head
            dropout: Dropout rate for classification head
        """
        if label_space is None:
            label_space = ESCO(version="1.2.0", language=Language.EN).get_skills_vocabulary()

        self._label_space = label_space
        self.base_model_name = base_model_name

        # Load base encoder
        self.encoder = SentenceTransformer(base_model_name)
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        assert embedding_dim is not None, (
            f"Embedding dimension is not available for {base_model_name}"
        )

        # Random classification head
        self.rnd_classifier = nn.Sequential(
            nn.Linear(embedding_dim, len(label_space)),
        )

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnd_classifier.to(self.device)
        self.encoder.to(self.device)

        self.encoder.eval()
        self.rnd_classifier.eval()

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"Classifier-{self.base_model_name.split('/')[-1]}"

    @property
    def description(self) -> str:
        return (
            "Random baseline for multi-label classification with random prediction head for ESCO."
        )

    @property
    def classification_label_space(self) -> list[str] | None:
        """
        Ordered list of label names corresponding to classification outputs.

        Returns the exact label ordering used during training. This is critical
        for ensuring model outputs align with task labels during evaluation.
        """
        return self._label_space

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """
        Compute classification scores for texts.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (as text)
            input_type: Type of input (e.g., JOB_TITLE, SKILL_NAME)
            target_input_type: Type of target (unused for this model, kept for interface compatibility)

        Returns
        -------
            Tensor of shape (n_texts, n_classes) where column i corresponds
            to self._label_space[i]. Values are logits (not probabilities).
        """
        if self.rnd_classifier is None:
            raise ValueError(
                "Model does not have a classification head. "
                "Initialize with label_space to create classifier."
            )

        # Validate that provided targets match model's internal label space
        if targets != self._label_space:
            if len(targets) != len(self._label_space):
                raise ValueError(
                    f"Target space size mismatch: provided {len(targets)} targets "
                    f"but model has {len(self._label_space)} classes."
                )
            if set(targets) != set(self._label_space):
                raise ValueError(
                    f"Target labels don't match model's label space. "
                    f"Model expects: {self._label_space[:5]}... "
                    f"but got: {targets[:5]}..."
                )
            # Same labels, different order - need to handle this
            logger.warning(
                "Target label order doesn't match model's internal label space. "
                "This may cause incorrect predictions. Consider reordering or retraining."
            )

        # Encode texts
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings = embeddings.to(self.device)

        # Apply classification head
        outputs = self.rnd_classifier(embeddings)

        return outputs

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """
        Compute ranking scores using classification scores.

        This model is designed for classification, but can still compute
        similarity-based rankings when the target space matches the model's label space.

        Args:
            queries: List of query texts
            targets: List of target texts (must match model's label space)
            query_input_type: Type of query input
            target_input_type: Type of target input

        Returns
        -------
            Tensor of shape (n_queries, n_targets) with classification scores
        """
        # Validate that targets match the model's label space
        if targets != self._label_space:
            if len(targets) != len(self._label_space):
                raise ValueError(
                    "Cannot use classification model for ranking: target space size mismatch. "
                    f"Model has {len(self._label_space)} classes but task has {len(targets)} targets."
                )
            if set(targets) != set(self._label_space):
                raise ValueError(
                    "Cannot use classification model for ranking: target labels don't match "
                    "model's label space. Model was trained on different labels than the ranking task requires."
                )
            # Same labels, different order
            raise ValueError(
                "Cannot use classification model for ranking: target label order doesn't match. "
                "Model expects labels in specific order but ranking task provides different order."
            )

        # Compute classification scores and use for ranking
        # Since targets match label space, classification scores are valid ranking scores
        return self.compute_classification(
            texts=queries,
            targets=targets,
            input_type=query_input_type,
            target_input_type=target_input_type,
        )
