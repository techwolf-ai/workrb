"""
Custom Model Example - Creating a Custom Model.

This example demonstrates how to create a custom model that can be used
with the WorkRB framework. Custom models should inherit from workrb.models.ModelInterface
and implement the required abstract methods.
"""

import torch
from sentence_transformers import SentenceTransformer

from workrb.models.base import ModelInterface
from workrb.registry import register_model
from workrb.types import ModelInputType


@register_model()
class MyCustomModel(ModelInterface):
    """
    Example custom model for demonstrating the extensibility of WorkRB.

    This model shows how to:
    1. Inherit from ModelInterface
    2. Implement required abstract methods
    3. Add custom parameters (e.g., custom_skip_query_norm)
    4. Work with both ranking and classification tasks
    """

    def __init__(
        self, base_model_name: str = "all-MiniLM-L6-v2", custom_skip_query_norm: bool = False
    ):
        """
        Initialize the custom model.

        Args:
            base_model_name: Name of the sentence transformer model to use
            custom_skip_query_norm: If True, skip normalization of query embeddings
        """
        self.base_model_name = base_model_name
        self.custom_skip_query_norm = custom_skip_query_norm

        # Load the base encoder
        self.encoder = SentenceTransformer(base_model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(device)
        self.encoder.eval()

    def name(self) -> str:
        """Return the unique name of this model."""
        return f"MyCustomModel-{self.base_model_name.split('/')[-1]}"

    def description(self) -> str:
        """Return the description of this model."""
        return "A custom model that demonstrates WorkRB extensibility"

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """
        Compute ranking scores between queries and targets.

        This implementation demonstrates custom normalization behavior where
        query embeddings can optionally skip normalization based on a flag.

        Args:
            queries: List of query texts
            targets: List of target texts
            query_input_type: Type of query input (e.g., JOB_TITLE)
            target_input_type: Type of target input (e.g., SKILL_NAME)

        Returns
        -------
            Tensor of shape (n_queries, n_targets) with similarity scores
        """
        # Encode queries and targets
        query_embeddings = self.encoder.encode(queries, convert_to_tensor=True)
        target_embeddings = self.encoder.encode(targets, convert_to_tensor=True)

        # Custom behavior: optionally skip query normalization
        query_norm = (
            torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            if not self.custom_skip_query_norm
            else query_embeddings
        )
        target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix (n_queries x n_targets)
        similarity_matrix = torch.mm(query_norm, target_norm.t())

        return similarity_matrix

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """
        Compute classification scores for texts.

        For models without a dedicated classification head, we can reuse
        the ranking mechanism to compute similarity between texts and class labels.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (as text)
            input_type: Type of input (e.g., JOB_TITLE)
            target_input_type: Type of target (e.g., SKILL_NAME)

        Returns
        -------
            Tensor of shape (n_texts, n_classes) with similarity scores
        """
        if target_input_type is None:
            target_input_type = input_type

        # Reuse ranking logic for classification
        return self._compute_rankings(
            queries=texts,
            targets=targets,
            query_input_type=input_type,
            target_input_type=target_input_type,
        )

    @property
    def classification_label_space(self) -> list[str] | None:
        """
        Return the classification label space.

        Returns None because this model doesn't have a fixed classification head.
        The label space is determined dynamically by the task.
        """
        return None


if __name__ == "__main__":
    # Example usage
    print("🚀 Custom Model Example")
    print("=" * 50)

    # 1. Create custom model with custom configuration
    model = MyCustomModel(base_model_name="all-MiniLM-L6-v2", custom_skip_query_norm=False)
    print(f"Model name: {model.name}")

    # 2. Create a simple task to test the model
    tasks = [
        workrb.tasks.ESCOJob2SkillRanking(split="val", languages=["en"]),
    ]

    # 3. Run the benchmark
    print("\nRunning benchmark with custom model...")
    results = workrb.evaluate(
        model,
        tasks,
        output_folder="results/custom_model_demo",
        description="Demonstration of custom model with configurable query normalization",
        force_restart=True,
    )

    # 5. Display results
    print("\nResults:")
    print(results)
