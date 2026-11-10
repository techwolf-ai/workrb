"""Tests for runtime registration of Language and ModelInputType values."""

import pickle

import pytest

from workrb.tasks.abstract.base import BaseTaskGroup, LabelType
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask
from workrb.types import DatasetLanguages, Language, ModelInputType


def test_unregistered_value_raises():
    """Unknown codes keep raising, so typos are still caught."""
    with pytest.raises(ValueError, match="not a valid Language"):
        Language("deu")
    with pytest.raises(ValueError, match="not a valid Language"):
        Language(["en"])


def test_register_makes_value_resolvable():
    """A registered value behaves like a built-in member."""
    sw = Language.register("sw")

    assert Language("sw") is sw
    assert Language.register("sw") is sw
    assert isinstance(sw, Language)
    assert (sw.value, sw.name, sw == "sw") == ("sw", "SW", True)
    assert pickle.loads(pickle.dumps(sw)) is sw


def test_register_existing_value_returns_builtin_member():
    """Registering a shipped value is a no-op."""
    assert Language.register("en") is Language.EN
    assert ModelInputType.register("job_title") is ModelInputType.JOB_TITLE


def test_registries_are_isolated_per_enum():
    """Registering on one enum does not leak into another."""
    ModelInputType.register("course_title")

    assert ModelInputType("course_title").value == "course_title"
    with pytest.raises(ValueError, match="not a valid Language"):
        Language("course_title")


@pytest.mark.parametrize("value", ["EN", "Yo", "", None])
def test_register_rejects_non_lowercase_values(value):
    """Case variants are rejected to keep result keys unique."""
    with pytest.raises(ValueError, match="lowercase"):
        Language.register(value)


def test_task_accepts_registered_values():
    """A task can use a registered language, input type and its own task group."""
    yo = Language.register("yo")
    course = ModelInputType.register("course_name")

    class CourseGroup(BaseTaskGroup):
        COURSES = "rank_courses"

    class YorubaTask(RankingTask):
        name = "YorubaTask"
        description = "Toy task in a registered language"
        query_input_type = course
        target_input_type = course
        task_group = CourseGroup.COURSES
        label_type = LabelType.SINGLE_LABEL
        supported_query_languages = (yo,)
        supported_target_languages = (yo,)

        def load_dataset(self, dataset_id, split):
            return RankingDataset(
                query_texts=["a"],
                target_indices=[[0]],
                target_space=["a", "b"],
                dataset_id=dataset_id,
            )

    task = YorubaTask(languages=["yo"], split="test")

    assert task.dataset_ids == ["yo"]
    assert task.get_dataset_languages("yo") == DatasetLanguages(frozenset({yo}), frozenset({yo}))
    assert task.get_task_config()["task_group"] == "rank_courses"
