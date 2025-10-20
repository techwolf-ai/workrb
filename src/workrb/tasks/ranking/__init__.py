"""
Ranking task implementations using ESCO data directly.

This module contains improved implementations of ranking tasks that:
- Use ESCO library directly instead of preprocessed files
- Support multilingual evaluation across ESCO versions
- Minimize external dependencies
"""

from workrb.tasks.ranking.job2skill import ESCOJob2SkillRanking
from workrb.tasks.ranking.jobnorm import JobBERTJobNormRanking
from workrb.tasks.ranking.skill2job import ESCOSkill2JobRanking
from workrb.tasks.ranking.skill_extraction import (
    HouseSkillExtractRanking,
    TechSkillExtractRanking,
)
from workrb.tasks.ranking.skill_similarity import SkillMatch1kSkillSimilarityRanking
from workrb.tasks.ranking.skillnorm import ESCOSkillNormRanking

__all__ = [
    "ESCOJob2SkillRanking",
    "ESCOSkill2JobRanking",
    "ESCOSkillNormRanking",
    "HouseSkillExtractRanking",
    "JobBERTJobNormRanking",
    "SkillMatch1kSkillSimilarityRanking",
    "TechSkillExtractRanking",
]
