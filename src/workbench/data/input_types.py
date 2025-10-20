"""Types describing the input modality for a model."""

from enum import StrEnum


class ModelInputType(StrEnum):
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
