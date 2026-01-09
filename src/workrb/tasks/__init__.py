"""
Task module with hierarchical structure.
"""

# Core task classes
from .abstract import ClassificationTask, LabelType, Task
from .abstract.ranking_base import RankingDataset, RankingTask

# Task implementations
from .classification.job2skill import ESCOJob2SkillClassification
from .ranking.job2skill import ESCOJob2SkillRanking
from .ranking.job_similarity import JobTitleSimilarityRanking
from .ranking.jobnorm import JobBERTJobNormRanking
from .ranking.skill2job import ESCOSkill2JobRanking
from .ranking.skill_extraction import (
    HouseSkillExtractRanking,
    SkillSkapeExtractRanking,
    TechSkillExtractRanking,
)
from .ranking.skill_similarity import SkillMatch1kSkillSimilarityRanking
from .ranking.skillnorm import ESCOSkillNormRanking

__all__ = [
    # Abstract classes
    "Task",
    "LabelType",
    "RankingTask",
    "ClassificationTask",
    "RankingDataset",
    # Classification tasks
    "ESCOJob2SkillClassification",
    # Ranking tasks
    "ESCOJob2SkillRanking",
    "ESCOSkill2JobRanking",
    "ESCOSkillNormRanking",
    "JobBERTJobNormRanking",
    "JobTitleSimilarityRanking",
    "HouseSkillExtractRanking",
    "TechSkillExtractRanking",
    "SkillSkapeExtractRanking",
    "SkillMatch1kSkillSimilarityRanking",
]
