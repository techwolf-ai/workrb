"""Model interface for WorkRB."""

from abc import ABC, abstractmethod

import torch

from workrb.types import ModelInputType


class ModelInterface(ABC):
    """Abstract base class for all models in WorkRB."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the model."""

    @property
    def citation(self) -> str | None:
        """Return the citation of the model."""
        return None

    @abstractmethod
    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """Compute ranking scores between queries and targets."""

    def compute_rankings(self, *args, **kwargs) -> torch.Tensor:
        """Compute ranking scores between queries and targets."""
        with torch.no_grad():
            return self._compute_rankings(*args, **kwargs)

    @abstractmethod
    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores for texts.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (as text)
            input_type: Type of input (e.g., JOB_TITLE, SKILL_NAME)
            target_input_type: Type of target (e.g., SKILL_NAME). If None, uses input_type.

        Returns
        -------
            Tensor of shape (n_texts, n_classes) with class logits/probabilities
        """

    def compute_classification(self, *args, **kwargs) -> torch.Tensor:
        """Compute classification scores for texts."""
        with torch.no_grad():
            return self._compute_classification(*args, **kwargs)

    @property
    @abstractmethod
    def classification_label_space(self) -> list[str] | None:
        """
        Ordered list of label names corresponding to classification outputs.

        This defines the mapping between model output indices and label names.
        The order must match exactly how the model was trained.

        Returns
        -------
            List of label names where index i corresponds to output neuron i,
            or None if model doesn't have a classification head.

        Example
        -------
            ["Java", "Machine Learning", "Python", ...]
            where model output[:,0] = probability for "Java", etc.

        Note
        ----
            This is critical for ensuring model outputs align with task labels.
        """
