"""Tests for ranking artifact persistence in evaluate(save_rankings=...)."""

import json
import shutil
from pathlib import Path

import pytest
import torch

import workrb
from workrb.models.base import ModelInterface
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


class TinyRankingTask(RankingTask):
    """Minimal ranking task used to test ranking artifact persistence."""

    @property
    def name(self) -> str:
        return "Tiny Ranking Task"

    @property
    def description(self) -> str:
        return "Tiny in-memory ranking task for tests."

    @property
    def supported_query_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def task_group(self) -> RankingTaskGroup:
        return RankingTaskGroup.SKILL_EXTRACTION

    @property
    def label_type(self) -> LabelType:
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        return ModelInputType.SKILL_SENTENCE

    @property
    def target_input_type(self) -> ModelInputType:
        return ModelInputType.SKILL_NAME

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        return RankingDataset(
            query_texts=["query one", "query two"],
            target_indices=[[0], [1]],
            target_space=["target_a", "target_b", "target_c"],
            dataset_id=dataset_id,
        )


class TinyDeterministicModel(ModelInterface):
    """Deterministic model with fixed ranking scores for tests."""

    @property
    def name(self) -> str:
        return "tiny-deterministic-model"

    @property
    def description(self) -> str:
        return "Tiny deterministic model for ranking artifact tests."

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        # 2 queries x 3 targets
        return torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        return torch.zeros((len(texts), len(targets)))

    @property
    def classification_label_space(self) -> list[str] | None:
        return None


def test_evaluate_saves_rankings_artifact_when_enabled():
    """evaluate(save_rankings=True) writes one JSON artifact per ranking dataset with full per-target scores."""
    output_folder = Path("tmp/rankings_artifact_test_enabled")
    if output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)

    model = TinyDeterministicModel()
    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]

    _ = workrb.evaluate(
        model=model,
        tasks=tasks,
        output_folder=str(output_folder),
        force_restart=True,
        save_rankings=True,
    )

    rankings_dir = output_folder / "rankings" / model.name
    ranking_files = list(rankings_dir.glob("*.json"))
    assert len(ranking_files) == 1

    with open(ranking_files[0]) as f:
        payload = json.load(f)

    assert set(payload.keys()) == {model.name}
    task_payload = payload[model.name]
    assert set(task_payload.keys()) == {"Tiny Ranking Task"}
    dataset_payload = task_payload["Tiny Ranking Task"]
    assert set(dataset_payload.keys()) == {"en"}

    leaf = dataset_payload["en"]
    assert leaf["num_queries"] == 2
    assert leaf["num_targets"] == 3

    scores = leaf["scores"]
    assert set(scores.keys()) == {"query one", "query two"}
    assert scores["query one"]["target_a"] == pytest.approx(0.1)
    assert scores["query one"]["target_b"] == pytest.approx(0.2)
    assert scores["query one"]["target_c"] == pytest.approx(0.3)
    assert scores["query two"]["target_a"] == pytest.approx(0.4)
    assert scores["query two"]["target_b"] == pytest.approx(0.5)
    assert scores["query two"]["target_c"] == pytest.approx(0.6)


def test_evaluate_does_not_save_rankings_artifact_by_default():
    """evaluate() without save_rankings does not create a rankings/ directory."""
    output_folder = Path("tmp/rankings_artifact_test_disabled")
    if output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)

    model = TinyDeterministicModel()
    tasks = [TinyRankingTask(split=DatasetSplit.TEST, languages=[Language.EN])]

    _ = workrb.evaluate(
        model=model,
        tasks=tasks,
        output_folder=str(output_folder),
        force_restart=True,
    )

    rankings_dir = output_folder / "rankings"
    assert not rankings_dir.exists()
