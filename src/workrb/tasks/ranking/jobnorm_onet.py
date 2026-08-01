import csv
from pathlib import Path

import appdirs
import requests
from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class JobBERTONetJobNormRanking(RankingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache dir pattern mirrors ESCO (src/workrb/data/esco.py:130-131),
        # but uses "workrb" as the app name.
        cache_dir = appdirs.user_cache_dir("workrb")
        self.base_path = Path(cache_dir) / "onet"
        self.onet_file_path = self.base_path / "onet_30_3_occupations.txt"

    @property
    def name(self) -> str:
        """Job Normalization task name."""
        return "ONET Job Title Normalization"

    @property
    def description(self) -> str:
        """Job Normalization task description."""
        return "Normalize job titles to canonical O*NET occupation groups"

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
        """Query input type for job titles"""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for O*NET occupations."""
        return ModelInputType.JOB_TITLE

    def _download_onet_data(self) -> bool:
        """Downloads O*NET version 30.3 Occupations Data from O*NET website

        Returns
        -------
            bool
        """
        url = "https://www.onetcenter.org/dl_files/database/db_30_3_text/Occupation%20Data.txt"
        r = requests.get(url)
        if r.status_code == 200:
            self.base_path.mkdir(parents=True, exist_ok=True)
            with open(self.onet_file_path, "wb") as file:
                file.write(r.content)
                return True
        else:
            return False

    def _check_onet_data_exists(self) -> bool:
        """Checks whether the O*NET data file already exists in the cache dir.

        Returns
        -------
            bool
        """
        return self.onet_file_path.is_file()

    def _load_onet_target_space(self) -> dict:
        """Loads downloaded TSV file from O*NET and forms target space

        Returns
        -------
            target_space: dict[str, int]
        """
        target_space = {}
        with open(self.onet_file_path) as f:
            csvf = csv.DictReader(f, delimiter="\t")
            for index, row in enumerate(csvf):
                target_space[row["Title"]] = index
        return target_space

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load job normalization data for a specific split and dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            RankingDataset object
        """
        # Download file from O*NET if not in cache
        if not self._check_onet_data_exists():
            print("Downloading O*NET v 30.3 Occupations Data")
            self._download_onet_data()

        # Create target space for metrics
        target_space = self._load_onet_target_space()

        query_texts, target_indices = [], []

        ds = load_dataset("workforcedeveloper/JobBERT-ONET-evaluation-dataset")
        split_map = {split.VAL: "valid", split.TEST: "test"}
        ds_split = ds[split_map[split]]
        df = ds_split.to_pandas()
        for index, row in df.iterrows():
            if row["onet_job_title"] in target_space:
                query_texts.append(row["vacancy_job_title"])
                target_index = target_space[row["onet_job_title"]]
                target_indices.append([target_index])

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=target_space,
            dataset_id=dataset_id,
        )
