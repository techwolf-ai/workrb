"""
Task module with hierarchical structure.
"""

# Core task classes
from .abstract import ClassificationTask, LabelType, Task
from .abstract.ranking_base import DuplicateStrategy, RankingDataset, RankingTask

# Task implementations
from .classification.job2skill import ESCOJob2SkillClassification
from .ranking.freelancer_project_matching import (
    ProjectCandidateRanking,
    SearchQueryCandidateRanking,
)
from .ranking.job2skill import ESCOJob2SkillRanking
from .ranking.job_similarity import JobTitleSimilarityRanking
from .ranking.jobnorm import JobBERTJobNormRanking
from .ranking.jobnorm_onet import JobBERTONetJobNormRanking
from .ranking.melo import MELORanking
from .ranking.mels import MELSRanking
from .ranking.skill2job import ESCOSkill2JobRanking
from .ranking.skill_extraction import (
    HouseGradedSkillExtractRanking,
    HouseSkillExtractRanking,
    SkillSkapeExtractRanking,
    SkillSkapeGradedSkillExtractRanking,
    SkillXLSkillExtractRanking,
    TechGradedSkillExtractRanking,
    TechSkillExtractRanking,
    TechWolfGradedSkillExtractRanking,
    TechWolfSkillExtractRanking,
)
from .ranking.skill_similarity import SkillMatch1kSkillSimilarityRanking
from .ranking.skillnorm import ESCOGradedSkillNormRanking, ESCOSkillNormRanking

__all__ = [
    # Abstract classes
    "Task",
    "LabelType",
    "RankingTask",
    "ClassificationTask",
    "RankingDataset",
    "DuplicateStrategy",
    # Classification tasks
    "ESCOJob2SkillClassification",
    # Ranking tasks
    "ESCOGradedSkillNormRanking",
    "ESCOJob2SkillRanking",
    "ESCOSkill2JobRanking",
    "ESCOSkillNormRanking",
    "JobBERTJobNormRanking",
    "JobTitleSimilarityRanking",
    "MELORanking",
    "MELSRanking",
    "HouseGradedSkillExtractRanking",
    "HouseSkillExtractRanking",
    "TechGradedSkillExtractRanking",
    "TechSkillExtractRanking",
    "TechWolfGradedSkillExtractRanking",
    "TechWolfSkillExtractRanking",
    "SkillSkapeExtractRanking",
    "SkillSkapeGradedSkillExtractRanking",
    "SkillXLSkillExtractRanking",
    "SkillMatch1kSkillSimilarityRanking",
    "ProjectCandidateRanking",
    "SearchQueryCandidateRanking",
    "JobBERTONetJobNormRanking",
]
