"""Test task registry system functionality."""

import pytest

import workrb
from workrb.registry import (
    TaskRegistry,
    create_task_from_config,
    register_task,
)
from workrb.tasks.abstract import Task
from workrb.tasks.abstract.base import LabelType, Language, TaskType
from workrb.tasks.abstract.ranking_base import RankingTaskGroup


class BaseTestTask(Task):
    """Base test task class that implements all abstract methods."""

    def __init__(self, languages, split="test", **kwargs):
        super().__init__(languages=languages, split=split)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str:
        return f"Test task: {self.__class__.__name__}"

    @property
    def task_group(self):
        return RankingTaskGroup.JOB_NORMALIZATION

    @property
    def task_type(self):
        return TaskType.RANKING

    @property
    def label_type(self):
        return LabelType.SINGLE_LABEL

    @property
    def supported_query_languages(self):
        return [Language.EN, Language.DE]

    @property
    def supported_target_languages(self):
        return [Language.EN, Language.DE]

    @property
    def default_metrics(self):
        return ["accuracy"]

    def load_monolingual_data(self, language, split):
        return {"test": "data", "language": str(language), "split": str(split)}

    def evaluate(self, model, metrics=None, language="en"):
        return {"accuracy": 0.95, "test_metric": 1.0}


class TestTaskRegistry:
    """Test cases for the TaskRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        TaskRegistry._registry.clear()

    def test_register_task_decorator_basic(self):
        """Test basic task registration with decorator."""

        @register_task()
        class TestTask(BaseTestTask):
            pass

        # Task should be automatically registered
        assert "TestTask" in TaskRegistry._registry
        assert TaskRegistry._registry["TestTask"] == TestTask

    def test_register_task_decorator_custom_name(self):
        """Test task registration with custom name."""

        @register_task("CustomName")
        class AnotherTestTask(BaseTestTask):
            pass

        # Task should be registered with custom name
        assert "CustomName" in TaskRegistry._registry
        assert "AnotherTestTask" not in TaskRegistry._registry
        assert TaskRegistry._registry["CustomName"] == AnotherTestTask

    def test_register_non_task_class_fails(self):
        """Test that registering non-Task class raises ValueError."""
        with pytest.raises(ValueError, match="must inherit from Task"):

            @register_task()
            class NotATask:
                pass

    def test_registry_get_existing_task(self):
        """Test getting an existing task from registry."""

        @register_task()
        class GetTestTask(BaseTestTask):
            pass

        retrieved_class = TaskRegistry.get("GetTestTask")
        assert retrieved_class == GetTestTask

    def test_registry_get_nonexistent_task_fails(self):
        """Test that getting nonexistent task raises ValueError."""
        with pytest.raises(ValueError, match="Task 'NonexistentTask' not found"):
            TaskRegistry.get("NonexistentTask")

    def test_registry_create_task(self):
        """Test creating task instance from registry."""

        @register_task()
        class CreateTestTask(BaseTestTask):
            pass

        # Create task instance
        task = TaskRegistry.create("CreateTestTask", languages=["en"], custom_param="test_value")

        assert isinstance(task, CreateTestTask)
        assert task.languages == [Language.EN]
        assert task.custom_param == "test_value"
        assert task.name == "CreateTestTask"

    def test_registry_list_available(self):
        """Test listing available tasks."""

        @register_task()
        class ListTestTask1(BaseTestTask):
            pass

        @register_task()
        class ListTestTask2(BaseTestTask):
            pass

        available = TaskRegistry.list_available()

        assert "ListTestTask1" in available
        assert "ListTestTask2" in available
        assert len(available) >= 2

        # Check module path format
        assert available["ListTestTask1"].endswith("ListTestTask1")
        assert available["ListTestTask2"].endswith("ListTestTask2")

    def test_register_duplicate_name_raises_error(self):
        """Test that registering duplicate task names raises ValueError."""

        @register_task("DuplicateName")
        class FirstTask(BaseTestTask):
            pass

        # Attempting to register another class with the same name should fail
        with pytest.raises(ValueError, match="already registered"):

            @register_task("DuplicateName")
            class SecondTask(BaseTestTask):
                pass

    def test_register_same_class_twice_succeeds(self):
        """Test that re-registering the same class is allowed (e.g., during re-import)."""

        @register_task("ReimportTest")
        class ReimportTask(BaseTestTask):
            pass

        # Re-registering the same exact class should not raise an error
        # This simulates module re-imports
        TaskRegistry.register("ReimportTest", ReimportTask)

        assert TaskRegistry.get("ReimportTest") == ReimportTask


class TestTaskRegistryConfigIntegration:
    """Test config-based task creation."""

    def setup_method(self):
        """Clear registry before each test."""
        TaskRegistry._registry.clear()

    def test_create_task_from_config_basic(self):
        """Test creating task from configuration dictionary."""

        @register_task()
        class ConfigTestTask(BaseTestTask):
            pass

        config = {
            "class": "ConfigTestTask",
            "languages": ["en", "de"],
            "config_param": "config_value",
        }

        task = create_task_from_config(config)

        assert isinstance(task, ConfigTestTask)
        assert task.languages == [Language.EN, Language.DE]
        assert task.config_param == "config_value"
        assert task.name == "ConfigTestTask"

    def test_create_task_from_config_missing_class_fails(self):
        """Test that config without 'class' field fails."""
        config = {"languages": ["en"], "some_param": "value"}

        with pytest.raises(KeyError):
            create_task_from_config(config)

    def test_create_task_from_config_missing_languages_fails(self):
        """Test that config without 'languages' field fails."""

        @register_task()
        class LanguageTestTask(BaseTestTask):
            pass

        config = {"class": "LanguageTestTask", "some_param": "value"}

        with pytest.raises(KeyError):
            create_task_from_config(config)

    def test_create_task_from_config_unknown_task_fails(self):
        """Test that config with unknown task class fails."""
        config = {"class": "UnknownTaskClass", "languages": ["en"]}

        with pytest.raises(ValueError, match="Task 'UnknownTaskClass' not found"):
            create_task_from_config(config)


class TestWorkRBRegistryIntegration:
    """Test WorkRB integration with task registry."""

    def setup_method(self):
        """Clear registry before each test."""
        TaskRegistry._registry.clear()

    def test_workrb_list_available_tasks(self):
        """Test WorkRB.list_available_tasks() method."""

        @register_task()
        class WorkRBTestTask(BaseTestTask):
            pass

        available_tasks = workrb.list_available_tasks()

        assert "WorkRBTestTask" in available_tasks
        assert isinstance(available_tasks, dict)
        assert available_tasks["WorkRBTestTask"].endswith("WorkRBTestTask")

    def _get_task_names(self, available_tasks: dict[str, str]) -> list[str]:
        task_names = []
        for task_name, task_class_path in available_tasks.items():
            task_class = TaskRegistry.get(task_name)
            task_instance = task_class(languages=["en"], split="test")
            task_names.append(task_instance.name)
        return task_names

    def test_registered_tasks_have_unique_names(self):
        """Test that registered tasks have unique names."""
        available_tasks = workrb.list_available_tasks()
        assert len(available_tasks) == len(set(available_tasks.keys())), (
            "Duplicate registered tasks"
        )

        # Now check if the imported Task instances have unique names
        task_names = self._get_task_names(available_tasks)
        assert len(task_names) == len(set(task_names)), (
            f"Duplicate task names found: {[name for name in task_names if task_names.count(name) > 1]}"
        )

    def test_registered_tasks_with_duplicate_names(self):
        """Test that tasks with duplicate .name property are detected (not registry keys).

        This test checks for duplicate task.name properties, not registry keys.
        Registry keys must be unique (enforced by ValueError in register()),
        but task.name could theoretically collide if different registry keys are used.
        """

        # Different registry keys but same .name property
        @register_task("Task1")  # Different registry key
        class WorkRBTestTask1(BaseTestTask):
            @property
            def name(self) -> str:
                return "WorkRBTestTask"  # Same .name property

        @register_task("Task2")  # Different registry key
        class WorkRBTestTask2(BaseTestTask):
            @property
            def name(self) -> str:
                return "WorkRBTestTask"  # Same .name property

        available_tasks = workrb.list_available_tasks()
        task_names = self._get_task_names(available_tasks)
        assert len(task_names) - len(set(task_names)) == 1, (
            f"Should have found one duplicate task name 'WorkRBTestTask', got {len(task_names) - len(set(task_names))}"
        )


class TestRealTaskRegistration:
    """Test that real tasks can be imported (integration test)."""

    def test_real_task_imports_work(self):
        """Test that real task classes can be imported without errors."""
        # This tests that the decorator doesn't break the imports
        from workrb.tasks.classification.job2skill import ESCOJob2SkillClassification
        from workrb.tasks.ranking.job2skill import ESCOJob2SkillRanking
        from workrb.tasks.ranking.skill2job import ESCOSkill2JobRanking

        # Verify classes exist and have the decorator applied
        assert ESCOJob2SkillRanking is not None
        assert ESCOSkill2JobRanking is not None
        assert ESCOJob2SkillClassification is not None

        # Verify they inherit from Task
        from workrb.tasks.abstract import Task

        assert issubclass(ESCOJob2SkillRanking, Task)
        assert issubclass(ESCOSkill2JobRanking, Task)
        assert issubclass(ESCOJob2SkillClassification, Task)
