import workrb
from workrb.tasks.abstract.base import Language


def test_freelancer_project_ranking_task_loads():
    """Test that task loads without errors"""
    task = workrb.tasks.ProjectCandidateRanking(split="test", languages=[Language.EN.value])
    dataset = task.datasets["en"]

    assert len(dataset.query_texts) > 0
    assert len(dataset.target_space) > 0
    assert len(dataset.target_indices) == len(dataset.query_texts)

    task = workrb.tasks.SearchQueryCandidateRanking(split="test", languages=[Language.EN.value])
    dataset = task.datasets["en"]

    assert len(dataset.query_texts) > 0
    assert len(dataset.target_space) > 0
    assert len(dataset.target_indices) == len(dataset.query_texts)
