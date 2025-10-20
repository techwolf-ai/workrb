"""
Custom Task Example - Creating a Custom Ranking Task

This example demonstrates how to create a custom ranking task that can be used
with the WorkRB framework. Custom tasks should inherit from workrb.tasks.RankingTask
and implement the required abstract methods.
"""

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class MyCustomRankingTask(workrb.tasks.RankingTask):
    """
    Example custom ranking task for demonstrating the extensibility of WorkRB.

    This task shows how to:
    1. Inherit from RankingTask
    2. Implement required abstract methods
    3. Provide custom data loading logic
    4. Use custom metrics if needed
    """

    def __init__(self, **kwargs):
        """
        Initialize the custom ranking task.

        Args:
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return the name of this task."""
        return "MyCustomRankingTask"

    @property
    def description(self) -> str:
        """Return a description of this task."""
        return "A custom ranking task that demonstrates WorkRB extensibility"

    @property
    def query_input_type(self) -> ModelInputType:
        """Return the input type for queries."""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Return the input type for targets."""
        return ModelInputType.JOB_TITLE

    @property
    def default_metrics(self) -> list[str]:
        """Override default metrics if needed."""
        return ["map", "mrr", "recall@5", "recall@10"]

    @property
    def task_group(self) -> RankingTaskGroup:
        """Task group is custom."""
        return RankingTaskGroup.JOB2SKILL

    @property
    def label_type(self) -> LabelType:
        """Label type is multi label."""
        return LabelType.MULTI_LABEL

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages are English."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages are English."""
        return [Language.EN]

    def load_monolingual_data(self, language: Language, split: DatasetSplit) -> RankingDataset:
        """
        Load data for evaluation.

        This method must return a RankingDataset.

        Args:
            language: Language code (e.g., "en", "de", "fr")
            split: Data split ("test", "validation", "train")

        Returns
        -------
            RankingDataset with queries, targets, and labels for ranking evaluation
        """
        # Example data - in a real implementation, you would load this from files/databases
        queries = [
            "Machine learning engineer position",
            "Data scientist role",
            "Software developer job",
        ]

        targets = [
            "Python programming",
            "Machine learning",
            "Data analysis",
            "Software engineering",
            "Statistics",
            "Deep learning",
            "Web development",
            "Database management",
        ]

        # Labels: indices of relevant targets for each query
        labels = [
            [1, 3, 6],  # ML engineer: ML, software eng, web dev
            [1, 2, 4, 5],  # Data scientist: ML, data analysis, stats, deep learning
            [0, 3, 6, 7],  # Software dev: Python, software eng, web dev, database
        ]

        return RankingDataset(
            query_texts=queries,
            target_indices=labels,
            target_space=targets,
            language=language,
        )

    # Note: The evaluate() method is inherited from RankingTask and doesn't need
    # to be overridden unless you want custom evaluation logic


if __name__ == "__main__":
    # Example usage
    print("🚀 Custom Task Example")
    print("=" * 50)

    # 1. Create a model
    model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")

    # 2. Create custom tasks
    tasks = [MyCustomRankingTask(languages=["en"], split="test")]

    # 3. Run the benchmark
    results = workrb.evaluate(
        model,
        tasks,
        output_folder="results/custom_task_demo",
        description="Demonstration of custom ranking tasks",
        force_restart=True,
    )
