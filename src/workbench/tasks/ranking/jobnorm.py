"""Job Normalization Ranking Task."""

import pandas as pd

from workbench.data.esco import ESCO
from workbench.data.input_types import ModelInputType
from workbench.data.utils import get_data_path
from workbench.registry import register_task
from workbench.tasks.abstract.base import DatasetSplit, LabelType, Language
from workbench.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup


@register_task()
class JobBERTJobNormRanking(RankingTask):
    """Job Normalization Ranking Task."""

    local_data_path = get_data_path("jobbert_v1")
    raw_val_csv = "jobbert_v1_titles.csv"
    raw_test_csv = "jobbert_v1_titles.test.csv"
    orig_esco_version = "1.0.5"

    def __init__(self, esco_version: str = "1.0.5", **kwargs):
        self.esco_version = esco_version

        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Job Normalization task name."""
        return "Job Normalization"

    @property
    def description(self) -> str:
        """Job Normalization task description."""
        return "Normalize job descriptions to canonical ESCO occupation titles"

    @property
    def task_group(self) -> RankingTaskGroup:
        """Job Normalization task group."""
        return RankingTaskGroup.JOB_NORMALIZATION

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages are always English."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages are all ESCO languages."""
        return list(ESCO.SUPPORTED_ESCO_LANGUAGES)

    @property
    def label_type(self) -> LabelType:
        """Label type is single label."""
        return LabelType.SINGLE_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for job titles/descriptions."""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for ESCO occupations."""
        return ModelInputType.JOB_TITLE

    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """Load job normalization data."""
        csv_filename = self.raw_val_csv if split == DatasetSplit.VAL else self.raw_test_csv
        data_path = self.local_data_path / csv_filename
        df = pd.read_csv(data_path)

        # ESCO used in the original JobBERT dataset
        orig_esco = ESCO(version=self.orig_esco_version, language=Language.EN)

        # Map based on URI if not original version
        if self.esco_version == self.orig_esco_version and language == Language.EN:
            job_vocab = orig_esco.get_occupations_vocabulary()

        else:
            target_esco = ESCO(version=self.esco_version, language=language)

            # Original job URIs
            orig_job_uris = orig_esco.get_occupations_uris()
            df["uri"] = df["label"].apply(lambda x: orig_job_uris.get(x))

            # Target job titles
            target_job_uris = target_esco.get_occupations_uris()
            target_uris_to_job = {v: k for k, v in target_job_uris.items()}
            df["target_label"] = df["uri"].apply(lambda x: target_uris_to_job.get(x, pd.NA))

            # Filter NANs (without match to URI)
            df.dropna(inplace=True)
            df.drop(columns=["uri", "label"], inplace=True)
            df.rename(columns={"target_label": "label"}, inplace=True)

            # Keep the full target job vocabulary (even those without match)
            job_vocab = target_esco.get_occupations_vocabulary()

        # Final dataset post filtering
        job2label = {job: i for i, job in enumerate(job_vocab)}
        label_indices = [[job2label[title]] for title in df["label"].tolist()]
        query_texts = df["text"].tolist()

        return RankingDataset(
            query_texts=query_texts,
            target_indices=label_indices,
            target_space=job_vocab,
            language=language,
        )
