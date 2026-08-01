import workrb
from workrb.tasks.abstract.base import Language


def test_my_custom_task_loads():
    """Test that task loads without errors"""
    task = workrb.tasks.JobBERTONetJobNormRanking(split="val", languages=["en"])
    dataset_id = Language.EN.value
    dataset = task.datasets[dataset_id]

    assert len(dataset.query_texts) > 0
    assert len(dataset.target_space) > 0
    assert len(dataset.target_indices) == len(dataset.query_texts)
