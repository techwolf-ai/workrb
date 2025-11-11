"""ESCO Skill Normalization ranking task."""

from enum import Enum

from sklearn.model_selection import train_test_split

from workrb.data.esco import ESCO
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


class SplitMode(str, Enum):
    """Split mode for val/test data splitting."""

    RND_SPLIT = "rnd_split"
    CLASS_UNIFORM_SPLIT = "class_uniform_split"


@register_task()
class ESCOSkillNormRanking(RankingTask):
    """
    ESCO Skill Normalization ranking task.

    Predict a canonical ESCO skill name (target) from an alternative ESCO skill name (query).

    Supports two split modes:
    - rnd_split: Random split based on test/validation fraction ratio (default)
    - class_uniform_split: Takes 1 alternative from each class with multiple alternatives
      for validation, rest for test. Disregards the val/test fraction ratio.
    """

    def __init__(
        self, esco_version: str = "1.2.0", split_mode: str = SplitMode.CLASS_UNIFORM_SPLIT, **kwargs
    ):
        """Initialize ESCO Skill Normalization task.

        Args:
            esco_version: Version of ESCO to use (default: "1.2.0")
            split_mode: Split mode for val/test split. Either "rnd_split" or
                       "class_uniform_split" (default: "rnd_split")
            **kwargs: Additional arguments passed to RankingTask
        """
        self.esco_version = esco_version
        try:
            self.split_mode = SplitMode(split_mode)
        except ValueError as e:
            raise ValueError(
                f"Invalid split_mode: '{split_mode}'. Supported modes: {list(SplitMode)}"
            ) from e
        self._adhoc_split_test_fraction: float | None = None
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """ESCO Skill Normalization task name."""
        return "ESCO Skill Normalization"

    @property
    def description(self) -> str:
        """ESCO Skill Normalization task description."""
        return "Normalize alternative skill names to canonical ESCO skill names"

    @property
    def task_group(self) -> RankingTaskGroup:
        """Job Normalization task group."""
        return RankingTaskGroup.SKILL_NORMALIZATION

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages are all ESCO languages."""
        return list(ESCO.SUPPORTED_ESCO_LANGUAGES)

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages are all ESCO languages."""
        return list(ESCO.SUPPORTED_ESCO_LANGUAGES)

    @property
    def split_test_fraction(self) -> float:
        """Default fraction of data to use for test split."""
        if self.split_mode == SplitMode.CLASS_UNIFORM_SPLIT:
            assert self._adhoc_split_test_fraction is not None, "Adhoc split test fraction not set"
            return self._adhoc_split_test_fraction

        return 0.8

    @property
    def label_type(self) -> LabelType:
        """Label type is multi label. There can be identical alternatives for different skills in ESCO."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for skill alternatives."""
        return ModelInputType.SKILL_NAME

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for canonical skill names."""
        return ModelInputType.SKILL_NAME

    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """Load skill normalization data from ESCO."""
        target_esco = ESCO(version=self.esco_version, language=Language(language))

        # Full vocab, even those without alternatives
        skill_vocab = target_esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        # Get alternative and preferred labels for skills
        skill2alts: dict[str, list[str]] = target_esco.get_skills_with_alternatives()
        alt2skills = self._get_alt2skills(skill2alts)

        if self.split_mode == SplitMode.RND_SPLIT:
            selected_queries, selected_labels = self._rnd_split(alt2skills, skill2label, split)

        if self.split_mode == SplitMode.CLASS_UNIFORM_SPLIT:
            selected_queries, selected_labels = self._class_uniform_split(
                alt2skills, skill2label, split
            )

        return RankingDataset(selected_queries, selected_labels, skill_vocab, language=language)

    def _rnd_split(
        self, alt2skills: dict[str, list[str]], skill2label: dict[str, int], split: DatasetSplit
    ) -> tuple[list[str], list[list[int]]]:
        """Random split implementation (current default behavior)."""
        all_queries: list[str] = list(alt2skills.keys())
        all_labels: list[list[int]] = [
            [skill2label[skill] for skill in skills] for skills in alt2skills.values()
        ]

        query_val, query_test, label_val, label_test = train_test_split(
            all_queries,
            all_labels,
            test_size=self.split_test_fraction,
            random_state=self.split_seed,
        )

        if split == DatasetSplit.VAL:
            return query_val, label_val
        if split == DatasetSplit.TEST:
            return query_test, label_test
        raise ValueError(f"Unsupported split: {split}")

    def _class_uniform_split(
        self, alt2skills: dict[str, list[str]], skill2label: dict[str, int], split: DatasetSplit
    ) -> tuple[list[str], list[list[int]]]:
        """Class uniform split: 1 alternative per skill-class for validation, rest for test."""
        # Split strategy
        val_queries, val_labels = [], []
        test_queries, test_labels = [], []

        # Keep count for all target skills, if no alt-count yet add to validation set else add to test set
        skill2alt_count = dict.fromkeys(skill2label.keys(), 0)

        # First is for
        for alt, skills in alt2skills.items():
            if len(skills) == 0:
                continue
            labels = [skill2label[skill] for skill in skills]

            if any(skill2alt_count[skill] == 0 for skill in skills):
                val_queries.append(alt)
                val_labels.append(labels)
            else:
                test_queries.append(alt)
                test_labels.append(labels)

            for skill in skills:
                skill2alt_count[skill] += 1

        if split == DatasetSplit.VAL:
            return val_queries, val_labels
        if split == DatasetSplit.TEST:
            return test_queries, test_labels
        raise ValueError(f"Unsupported split: {split}")

    def _get_alt2skills(self, skill2alts: dict[str, list[str]]) -> dict[str, list[str]]:
        """Get alt2skill mapping. Duplicate alt queries their skill labels get merged."""
        alt2skill: dict[str, list[str]] = {}
        for skill, alts in skill2alts.items():
            for alt in alts:
                if alt not in alt2skill:
                    alt2skill[alt] = [skill]
                elif skill not in alt2skill[alt]:
                    alt2skill[alt].append(skill)
        return alt2skill

    @property
    def citation(self) -> str:
        """Skill normalization task citation."""
        return """UWE-PLACEHOLDER"""
