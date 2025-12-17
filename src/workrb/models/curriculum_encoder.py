"""CurriculumMatch model wrapper for WorkRB."""

import torch
from sentence_transformers import SentenceTransformer

from workrb.models.base import ModelInterface
from workrb.registry import register_model
from workrb.types import ModelInputType


@register_model()
class CurriculumMatchModel(ModelInterface):
    """Curriculum_encoder model using sentence-transformers.

    This model is a fine-tuned MPNet-v2 trained using curriculum-based
    contrastive learning for retrieving ESCO skill labels from job description
    fragments. It maps job description sentences and skill labels to a shared
    embedding space, enabling similarity-based skill extraction.
    """

    def __init__(
        self,
        model_name: str = "Aleksandruz/skillmatch-mpnet-curriculum-retriever",
        **kwargs,
    ):
        self.base_model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.base_model_name.split("/")[-1]

    @property
    def description(self) -> str:
        """Return the model description."""
        return (
            "CurriculumMatch bi-encoder from Aleksandruz: "
            "https://huggingface.co/Aleksandruz/skillmatch-mpnet-curriculum-retriever"
        )

    @torch.no_grad()
    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute ranking scores using cosine similarity.

        Uses mean pooling to get sentence embeddings, then computes
        cosine similarity between query and target embeddings.
        """
        # Encode queries and targets (SentenceTransformer handles mean pooling)
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        target_embeddings = self.model.encode(targets, convert_to_tensor=True)

        # Normalize for cosine similarity
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(query_norm, target_norm.t())
        return similarity_matrix

    @torch.no_grad()
    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores by ranking texts against target labels.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (as text)
            input_type: Type of input (e.g., JOB_TITLE)
            target_input_type: Type of target (e.g., SKILL_NAME). If None, uses input_type.

        Returns
        -------
            Tensor of shape (n_texts, n_classes) with similarity scores
        """
        if target_input_type is None:
            target_input_type = input_type

        return self._compute_rankings(
            queries=texts,
            targets=targets,
            query_input_type=input_type,
            target_input_type=target_input_type,
        )

    @property
    def classification_label_space(self) -> list[str] | None:
        """CurriculumMatch model do not have classification heads."""
        return None

    @property
    def citation(self) -> str | None:
        """CurriculumMatch model citations."""
        return """
@inproceedings{bielinski2025retrieval,
  articleno    = {5},
  author       = {Bielinski, Aleksander and Brazier, David},
  booktitle    = {Proceedings of the 5th Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2025), in conjunction with the 19th ACM Conference on Recommender Systems},
  editor       = {Bogers, Toine and Bied, Guillaume and Decorte, Jean-Joris and Johnson, Chris and Kaya, Mesut},
  issn         = {1613-0073},
  language     = {eng},
  location     = {Prague, Czech Republic},
  pages        = {10},
  publisher    = {CEUR},
  title        = {From Retrieval to Ranking: A Two-Stage Neural Framework for Automated Skill Extraction},
  url          = {https://ceur-ws.org/Vol-4046/RecSysHR2025-paper_5.pdf},
  volume       = {4046},
  year         = {2025}}
"""
