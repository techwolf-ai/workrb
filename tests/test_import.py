"""Test basic import functionality."""

import importlib

from workrb import tasks
from workrb.config import BenchmarkConfig
from workrb.registry import TaskRegistry
from workrb.tasks import (
    ESCOJob2SkillClassification,
    ESCOJob2SkillRanking,
    ESCOSkill2JobRanking,
    ESCOSkillNormRanking,
    HouseSkillExtractRanking,
    JobBERTJobNormRanking,
    RankingDataset,
    RankingTask,
    SkillMatch1kSkillSimilarityRanking,
    Task,
    TechSkillExtractRanking,
    ranking,
)


def test_basic_imports():
    """Test that we can import core components."""
    assert isinstance(BenchmarkConfig.__name__, str)

    print("✓ All core imports successful")


def test_task_abstract_imports():
    """Test that we can import all task classes."""
    assert isinstance(RankingDataset.__name__, str)
    assert isinstance(RankingTask.__name__, str)
    assert isinstance(Task.__name__, str)

    print("✓ Successfully imported Task, RankingTask, RankingDataset")


def test_task_classification_imports():
    """Test that we can import all task classes."""
    assert isinstance(ESCOJob2SkillClassification.__name__, str)
    print("✓ Successfully imported JobSkillClassification")


def test_task_ranking_imports():
    """Test that we can import all task classes."""
    assert isinstance(ESCOJob2SkillRanking.__name__, str)
    assert isinstance(ESCOSkill2JobRanking.__name__, str)
    assert isinstance(ESCOSkillNormRanking.__name__, str)
    assert isinstance(HouseSkillExtractRanking.__name__, str)
    assert isinstance(JobBERTJobNormRanking.__name__, str)
    assert isinstance(SkillMatch1kSkillSimilarityRanking.__name__, str)
    assert isinstance(TechSkillExtractRanking.__name__, str)

    print("✓ Successfully imported ranking task classes")


def test_task_package_exports():
    """Test that all registered tasks are imported into workrb.tasks."""
    TaskRegistry.auto_discover()
    registered = TaskRegistry.list_available().values()

    missing = []
    for module_path in registered:
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        task_class = getattr(module, class_name)
        if not hasattr(tasks, task_class.__name__):
            missing.append(task_class.__name__)

    assert not missing, f"Missing task imports in workrb.tasks: {missing}"


def test_ranking_package_exports():
    """Test that all registered ranking tasks are imported into workrb.tasks.ranking."""
    TaskRegistry.auto_discover()
    registered = TaskRegistry.list_available().values()

    missing = []
    for module_path in registered:
        module_name, class_name = module_path.rsplit(".", 1)
        if not module_name.startswith("workrb.tasks.ranking."):
            continue
        module = importlib.import_module(module_name)
        task_class = getattr(module, class_name)
        if not hasattr(ranking, task_class.__name__):
            missing.append(task_class.__name__)

    assert not missing, f"Missing task imports in workrb.tasks.ranking: {missing}"


if __name__ == "__main__":
    """Run tests directly for quick verification."""
    test_basic_imports()
    test_task_abstract_imports()
    test_task_classification_imports()
    test_task_ranking_imports()
    print("✓ All import tests passed!")
