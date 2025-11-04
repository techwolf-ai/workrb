"""
Ranking task implementations using ESCO data directly.

This module contains improved implementations of ranking tasks that:
- Use ESCO library directly instead of preprocessed files
- Support multilingual evaluation across ESCO versions
- Minimize external dependencies
"""

from workbench.tasks.ranking.job2skill import ESCOJob2SkillRanking
from workbench.tasks.ranking.jobnorm import JobBERTJobNormRanking
from workbench.tasks.ranking.skill2job import ESCOSkill2JobRanking
from workbench.tasks.ranking.skill_extraction import (
    HouseSkillExtractRanking,
    TechSkillExtractRanking,
)
from workbench.tasks.ranking.skill_similarity import SkillMatch1kSkillSimilarityRanking
from workbench.tasks.ranking.skillnorm import ESCOSkillNormRanking

__all__ = [
    "ESCOJob2SkillRanking",
    "ESCOSkill2JobRanking",
    "ESCOSkillNormRanking",
    "HouseSkillExtractRanking",
    "JobBERTJobNormRanking",
    "SkillMatch1kSkillSimilarityRanking",
    "TechSkillExtractRanking",
]
