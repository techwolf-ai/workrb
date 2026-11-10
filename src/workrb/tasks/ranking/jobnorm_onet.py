"""Job Normalization Ranking Task with O*NET occupations as target space."""

import logging

import pandas as pd
from datasets import DatasetDict, load_dataset

from workrb.data.onet import ONET
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType

logger = logging.getLogger(__name__)


@register_task()
class JobBERTONetJobNormRanking(RankingTask):
    """Job Normalization Ranking Task on the O*NET taxonomy.

    Queries are the vacancy job titles of the JobBERT evaluation dataset. Targets
    are all occupation titles of the configured O*NET version. The ground-truth
    label of a query is the O*NET occupation obtained by mapping its original
    ESCO occupation through the official ESCO-to-O*NET crosswalk.
    """

    orig_onet_version = "30.3"

    # Pinned, as the dataset is hosted on a personal account
    dataset_revision = "437285e6aa24b1c74f34347fadd7382914f5d2c4"

    def __init__(self, onet_version: str = "30.3", **kwargs):
        self.onet_version = onet_version

        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Job Normalization (O*NET) task name."""
        return "Job Normalization (O*NET)"

    @property
    def description(self) -> str:
        """Job Normalization (O*NET) task description."""
        return "Normalize job titles to canonical O*NET occupation titles"

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
        """Supported target languages are always English."""
        return [Language.EN]

    @property
    def label_type(self) -> LabelType:
        """Label type is single label."""
        return LabelType.SINGLE_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for job titles."""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for O*NET occupations."""
        return ModelInputType.JOB_TITLE

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load job normalization data for a specific split and dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            RankingDataset object
        """
        ds = load_dataset(
            "workforcedeveloper/JobBERT-ONET-evaluation-dataset",
            revision=self.dataset_revision,
        )
        assert isinstance(ds, DatasetDict)

        split_map = {split.VAL: "valid", split.TEST: "test"}
        ds_split = ds[split_map[split]]
        df = ds_split.to_pandas()
        assert isinstance(df, pd.DataFrame)

        label_col = "onet_job_title"
        text_col = "vacancy_job_title"
        assert {label_col, text_col} <= set(df.columns), (
            f"Unexpected columns in dataset: {df.columns}, expected {label_col, text_col}"
        )
        df = df[[text_col, label_col]]

        # O*NET used to build the dataset
        orig_onet = ONET(version=self.orig_onet_version)

        if self.onet_version == self.orig_onet_version:
            job_vocab = orig_onet.get_occupations_vocabulary()

        # Map based on O*NET-SOC code if not original version
        else:
            target_onet = ONET(version=self.onet_version)

            orig_job_codes = orig_onet.get_occupations_codes()
            target_codes_to_job = {v: k for k, v in target_onet.get_occupations_codes().items()}
            df[label_col] = df[label_col].apply(
                lambda x: target_codes_to_job.get(orig_job_codes.get(x), pd.NA)
            )

            # Filter NANs (without match to O*NET-SOC code)
            df = df.dropna()

            # Keep the full target job vocabulary (even those without match)
            job_vocab = target_onet.get_occupations_vocabulary()

        # Final dataset post filtering
        job2label = {job: i for i, job in enumerate(job_vocab)}
        label_indices = [[job2label[title]] for title in df[label_col].tolist()]
        query_texts = df[text_col].tolist()

        return RankingDataset(
            query_texts=query_texts,
            target_indices=label_indices,
            target_space=job_vocab,
            dataset_id=dataset_id,
        )

    @property
    def citation(self) -> str:
        """No peer reviewed dataset!"""
        logger.warning(
            "The dataset of task '%s' is not peer reviewed, so there is no publication to cite.",
            self.name,
        )
        return """"""
