"""Shared types and enums used across WorkRB."""

from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple


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


class DatasetLanguages(NamedTuple):
    """Languages associated with a dataset for metric aggregation.

    Describes the input and output languages of a dataset. Used by
    ``_aggregate_per_language`` to group results by language.

    Examples
    --------
    Monolingual (e.g. English-only):
        ``DatasetLanguages(input_languages=frozenset({Language.EN}),
                          output_languages=frozenset({Language.EN}))``

    Cross-lingual (e.g. English queries, German targets):
        ``DatasetLanguages(input_languages=frozenset({Language.EN}),
                          output_languages=frozenset({Language.DE}))``

    Multilingual (e.g. queries in multiple languages, targets in one):
        ``DatasetLanguages(input_languages=frozenset({Language.EN, Language.FR}),
                          output_languages=frozenset({Language.DE}))``
    """

    input_languages: frozenset[Language]
    output_languages: frozenset[Language]


class LanguageAggregationMode(str, Enum):
    """Mode for grouping datasets by language during metric aggregation.

    Controls how ``_aggregate_per_language`` determines the grouping language
    for each dataset result.
    """

    MONOLINGUAL_ONLY = "monolingual_only"
    """Only aggregate monolingual datasets (singleton input == singleton output).

    Cross-lingual or multilingual datasets are skipped.
    """

    CROSSLINGUAL_GROUP_INPUT_LANGUAGES = "crosslingual_group_input_languages"
    """Group by the input language (requires singleton input_languages).

    Datasets with multiple input languages are skipped.
    """

    CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES = "crosslingual_group_output_languages"
    """Group by the output language (requires singleton output_languages).

    Datasets with multiple output languages are skipped.
    """


def get_language_grouping_key(
    input_languages: Sequence[str],
    output_languages: Sequence[str],
    mode: LanguageAggregationMode,
) -> str | None:
    """Determine the grouping language for a dataset given its languages.

    Returns ``None`` when the dataset is incompatible with the requested
    mode, so that the caller can skip it.

    Parameters
    ----------
    input_languages : Sequence[str]
        Input language codes for the dataset (e.g. query languages).
    output_languages : Sequence[str]
        Output language codes for the dataset (e.g. target languages).
    mode : LanguageAggregationMode
        The aggregation mode controlling how the language key is derived.

    Returns
    -------
    str or None
        Language code to group by, or ``None`` if the dataset is
        incompatible with the mode.
    """
    if mode == LanguageAggregationMode.MONOLINGUAL_ONLY:
        if (
            len(input_languages) != 1
            or len(output_languages) != 1
            or input_languages[0] != output_languages[0]
        ):
            return None
        return input_languages[0]

    if mode == LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES:
        if len(input_languages) != 1:
            return None
        return input_languages[0]

    if mode == LanguageAggregationMode.CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES:
        if len(output_languages) != 1:
            return None
        return output_languages[0]

    return None


class ExecutionMode(str, Enum):
    """Controls whether ``evaluate()`` skips datasets incompatible with the language aggregation.

    When a ``LanguageAggregationMode`` is specified, ``LAZY`` (the default) avoids
    running datasets that would be discarded during aggregation, saving compute.
    ``ALL`` evaluates every dataset regardless.
    """

    LAZY = "lazy"
    """Skip datasets incompatible with the chosen aggregation mode (default)."""

    ALL = "all"
    """Evaluate all datasets regardless of aggregation compatibility."""


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
