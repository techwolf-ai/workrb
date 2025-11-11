"""Shared types and enums used across WorkRB."""

from enum import Enum


class Language(str, Enum):
    """Language enum."""

    EN = "en"
    DE = "de"
    FR = "fr"
    IT = "it"
    ES = "es"
    BG = "bg"
    CS = "cs"
    DA = "da"
    ET = "et"
    EL = "el"
    GA = "ga"
    HR = "hr"
    LV = "lv"
    LT = "lt"
    HU = "hu"
    MT = "mt"
    NL = "nl"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    SK = "sk"
    SL = "sl"
    FI = "fi"
    SV = "sv"
    IS = "is"
    NO = "no"
    AR = "ar"
    UK = "uk"


class LabelType(str, Enum):
    """Label type enum."""

    SINGLE_LABEL = "single_label"
    """Multi-class or binary classification."""

    MULTI_LABEL = "multi_label"
    """Multi-label classification."""


class DatasetSplit(str, Enum):
    """Dataset split enum."""

    VAL = "val"
    TEST = "test"


class ModelInputType(str, Enum):
    """
    Type describing the input modality for a model.

    Can be used for type-specific routing in model inference.
    """

    JOB_TITLE = "job_title"
    """A job title."""

    SKILL_NAME = "skill_name"
    """A skill name."""

    SKILL_SENTENCE = "skill_sentence"
    """
    Sentence describing or containing a skill.
    For example, a skill description or job ad sentence.
    """
