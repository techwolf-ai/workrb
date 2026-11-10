"""Test the JobBERT job normalization task on the O*NET taxonomy."""

from workrb.tasks import JobBERTONetJobNormRanking
from workrb.tasks.abstract.base import Language


def test_loads_original_onet_version():
    """The validation split loads with the full O*NET 30.3 target space."""
    task = JobBERTONetJobNormRanking(split="val", languages=["en"])
    dataset = task.datasets[Language.EN.value]

    assert len(dataset.query_texts) == 13615
    assert len(dataset.target_space) == 1016
    assert len(dataset.target_indices) == len(dataset.query_texts)


def test_remaps_labels_to_other_onet_version():
    """Labels are remapped by O*NET-SOC code when another version is requested."""
    task = JobBERTONetJobNormRanking(split="val", languages=["en"], onet_version="25.0")
    dataset = task.datasets[Language.EN.value]

    # O*NET 25.0 uses the older O*NET-SOC 2010 taxonomy
    assert len(dataset.target_space) == 1110
    assert 0 < len(dataset.query_texts) < 13615
    assert all(
        0 <= index < len(dataset.target_space)
        for indices in dataset.target_indices
        for index in indices
    )
