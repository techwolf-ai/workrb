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
    JA = "ja"
    KO = "ko"
    ZH = "zh"
    CROSS = "cross_lingual"


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

    PROJECT_BRIEF_STRING = "brief_string"
    """
    Full natural-language description of a freelance or consulting opportunity.

    Context:
        Used as the primary input to candidate-matching pipelines.
        Represents a concrete project (not a permanent job position).

    Content:
        Typically includes business context, scope, required skills,
        expectations, and optional soft requirements.

    Example:
        Lead Dev for Greenfield B2B Material Platform (Next.js/GraphQL)

        We are Architech Innovations, a scale-up in the AEC tech space. ...
        We are building a greenfield MVP from the ground up. ...
        We are looking for a developer who have a proven track record ...
        skills: Strong decision-making, ...
    """

    SEARCH_QUERY_STRING = "search_query_string"
    """
    Short keyword-based query used to retrieve candidate profiles.

    Context:
        Used when a full project brief is not available, or as a
        lightweight input to narrow down candidates before deeper matching.

    Content:
        Minimal, high-signal keywords describing a role, skillset,
        or professional focus.

    Example:
        IT financial controller
    """

    CANDIDATE_PROFILE_STRING = "candidate_profile_string"
    """
    Structured natural-language summary of a candidate’s professional background.

    Context:
        Retrieved from the candidate database and evaluated for relevance
        against PROJECT_BRIEF_TEXT or SEARCH_QUERY_TEXT.

    Content:
        Include title, bio, skills, experience history.

    Example:
        Expert Shopify Developer

        Bio:
        I build and optimize Shopify Plus environments, ...

        Skills:
        Shopify Plus,Headless Commerce, ...

        Experiences:
        Lead Shopify Developer
        I led the development of a headless e-commerce site ...
        Shopify Plus, ...

        ...
    """
