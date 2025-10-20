"""Job Normalization Ranking Task."""

import pandas as pd
from datasets import DatasetDict, load_dataset

from workrb.data.esco import ESCO
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class JobBERTJobNormRanking(RankingTask):
    """Job Normalization Ranking Task."""

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
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("TechWolf/JobBERT-evaluation-dataset")
        assert isinstance(ds, DatasetDict)

        split_map = {split.VAL: "validation", split.TEST: "test"}
        ds_split = ds[split_map[split]]
        df = ds_split.to_pandas()
        assert isinstance(df, pd.DataFrame)

        uri_col = "esco_URI"
        label_col = "esco_job_title"
        text_col = "vacancy_job_title"
        assert set(df.columns) == {uri_col, label_col, text_col}, (
            f"Unexpected columns in dataset: {df.columns}, expected {uri_col, label_col, text_col}"
        )

        # ESCO used in the original JobBERT dataset
        orig_esco = ESCO(version=self.orig_esco_version, language=Language.EN)

        # Map based on URI if not original version
        if self.esco_version == self.orig_esco_version and language == Language.EN:
            job_vocab = orig_esco.get_occupations_vocabulary()

        else:
            target_esco = ESCO(version=self.esco_version, language=language)

            # Original job URIs
            orig_job_uris = orig_esco.get_occupations_uris()
            df[uri_col] = df[label_col].apply(lambda x: orig_job_uris.get(x))

            # Target job titles
            target_job_uris = target_esco.get_occupations_uris()
            target_uris_to_job = {v: k for k, v in target_job_uris.items()}
            df["target_label"] = df[uri_col].apply(lambda x: target_uris_to_job.get(x, pd.NA))

            # Filter NANs (without match to URI)
            df.dropna(inplace=True)
            df.drop(columns=[uri_col, label_col], inplace=True)
            df.rename(columns={"target_label": label_col}, inplace=True)

            # Keep the full target job vocabulary (even those without match)
            job_vocab = target_esco.get_occupations_vocabulary()

        # Final dataset post filtering
        job2label = {job: i for i, job in enumerate(job_vocab)}
        label_indices = [[job2label[title]] for title in df[label_col].tolist()]
        query_texts = df[text_col].tolist()

        return RankingDataset(
            query_texts=query_texts,
            target_indices=label_indices,
            target_space=job_vocab,
            language=language,
        )

    @property
    def citation(self) -> str:
        """JobBERT paper."""
        return """
@inproceedings{jobbert_2021,
    author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Demeester, Thomas and Develder, Chris}},
    booktitle    = {{FEAST, ECML-PKDD 2021 Workshop, Proceedings}},
    language     = {{eng}},
    location     = {{Online}},
    pages        = {{9}},
    title        = {{JobBERT : understanding job titles through skills}},
    url          = {{https://feast-ecmlpkdd.github.io/papers/FEAST2021_paper_6.pdf}},
    year         = {{2021}},
}
"""
