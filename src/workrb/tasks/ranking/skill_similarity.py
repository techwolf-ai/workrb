"""Skill similarity ranking task using SkillMatch dataset."""

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class SkillMatch1kSkillSimilarityRanking(RankingTask):
    """
    Skill similarity ranking task using SkillMatch-1K dataset.

    Predict similar skills from the SkillMatch dataset.
    """

    @property
    def name(self) -> str:
        """Skill similarity task name."""
        return "Skill Similarity"

    @property
    def description(self) -> str:
        """Skill similarity task description."""
        return "Find similar skills using the SkillMatch-1K dataset"

    @property
    def task_group(self) -> RankingTaskGroup:
        """Skill similarity task group."""
        return RankingTaskGroup.SKILLSIM

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages - English only."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages - English only."""
        return [Language.EN]

    @property
    def split_test_fraction(self) -> float:
        """Fraction of data to use for test split."""
        return 0.9

    @property
    def label_type(self) -> LabelType:
        """Label type is single label."""
        return LabelType.SINGLE_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for skills."""
        return ModelInputType.SKILL_NAME

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for skills."""
        return ModelInputType.SKILL_NAME

    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """
        Load skill similarity data from SkillMatch dataset.

        Uses only the 1k related pairs from the SkillMatch dataset, but
        uses all skills from the SkillMatch dataset for the vocabulary.
        """
        if language != Language.EN:
            raise ValueError("The validation set of this task is only available in English.")

        # Load data from external SkillMatch dataset
        dataset = load_dataset("TechWolf/SkillMatch-1K", split="test")
        assert isinstance(dataset, Dataset)

        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # Gather all unique skills (before split)
        skill_vocab = sorted(set(df["skill_a"].tolist() + df["skill_b"].tolist()))
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        # Only keep related pairs
        df = df[df["related"] == 1]

        all_queries = df["skill_a"].tolist()
        all_labels = [[skill2label[skill]] for skill in df["skill_b"].tolist()]

        # Split into validation and test sets
        query_val, query_test, label_val, label_test = train_test_split(
            all_queries,
            all_labels,
            test_size=self.split_test_fraction,
            random_state=self.split_seed,
        )

        if split == DatasetSplit.VAL:
            selected_queries = query_val
            selected_labels = label_val
        elif split == DatasetSplit.TEST:
            selected_queries = query_test
            selected_labels = label_test

        return RankingDataset(selected_queries, selected_labels, skill_vocab, language=language)

    @property
    def citation(self) -> str:
        """Skill similarity task citation."""
        return """
@misc{decorte2024skillmatchevaluatingselfsupervisedlearning,
      title={SkillMatch: Evaluating Self-supervised Learning of Skill Relatedness},
      author={Jens-Joris Decorte and Jeroen Van Hautte and Thomas Demeester and Chris Develder},
      year={2024},
      eprint={2410.05006},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05006},
}
"""
