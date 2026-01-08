"""
Model interfaces and implementations for WorkRB.
"""

from workrb.models.base import ModelInterface
from workrb.models.bi_encoder import BiEncoderModel, ConTeXTMatchModel, JobBERTModel
from workrb.models.classification_model import RndESCOClassificationModel
from workrb.models.curriculum_encoder import CurriculumMatchModel
from workrb.models.lexical_baselines import (
    BM25Model,
    EditDistanceModel,
    RandomRankingModel,
    TfIdfModel,
)

__all__ = [
    "BM25Model",
    "BiEncoderModel",
    "ConTeXTMatchModel",
    "CurriculumMatchModel",
    "EditDistanceModel",
    "JobBERTModel",
    "ModelInterface",
    "RandomRankingModel",
    "RndESCOClassificationModel",
    "TfIdfModel",
]
