import workrb
from workrb.tasks.abstract.base import Language


def test_freelancer_project_ranking_task_loads():
    """Test that task loads without errors"""
    dataset_name = Language.EN.value

    task = workrb.tasks.ProjectCandidateRanking(split="test", languages=[dataset_name])
    dataset = task.datasets[dataset_name]

    assert len(dataset.query_texts) > 0
    assert len(dataset.target_space) > 0
    assert len(dataset.target_indices) == len(dataset.query_texts)

    task = workrb.tasks.SearchQueryCandidateRanking(split="test", languages=[dataset_name])
    dataset = task.datasets[dataset_name]

    assert len(dataset.query_texts) > 0
    assert len(dataset.target_space) > 0
    assert len(dataset.target_indices) == len(dataset.query_texts)
