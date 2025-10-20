"""Job-Skill Multi-Label Classification benchmark task."""

import logging

import pandas as pd
from datasets import DatasetDict, load_dataset

from workrb.data.esco import ESCO
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, Language
from workrb.tasks.abstract.classification_base import (
    ClassificationDataset,
    ClassificationTaskGroup,
    MultiLabelClassificationTask,
)
from workrb.types import ModelInputType

logger = logging.getLogger(__name__)


@register_task()
class ESCOJob2SkillClassification(MultiLabelClassificationTask):
    """
    Job-skill multi-label classification task.

    This task evaluates a model's ability to predict relevant skills for a job
    (multi-label classification). It uses ESCO occupation-skill relations.
    """

    # Static validation dataset location (within repo data folder)
    original_esco_version = "1.2.0"

    def __init__(self, esco_version: str = "1.2.0", **kwargs):
        """
        Initialize Job-Skill Classification task.

        Args:
            esco_version: ESCO version to use for loading skills and relations
            **kwargs: Arguments passed to parent ClassificationTask (languages, split, etc.)
        """
        self.esco_version = esco_version
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Job-Skill Classification"

    @property
    def description(self) -> str:
        return "Multi-label classification of relevant skills for jobs"

    @property
    def task_group(self) -> ClassificationTaskGroup:
        return ClassificationTaskGroup.JOB2SKILL

    @property
    def supported_query_languages(self) -> list[Language]:
        """Queries (job titles) can be in English for val; all ESCO languages for test."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Target skills vocabulary spans all ESCO languages."""
        return list(ESCO.SUPPORTED_ESCO_LANGUAGES)

    @property
    def input_type(self) -> ModelInputType:
        """Input is job titles."""
        return ModelInputType.JOB_TITLE

    def get_output_space_size(self, language: Language) -> int:
        """Number of output classes (skills) for this classification task."""
        ds: ClassificationDataset = self.lang_datasets[language]
        return len(ds.label_space)

    def load_monolingual_data(
        self, split: DatasetSplit, language: Language
    ) -> ClassificationDataset:
        """
        Load job-skill classification data for specified language and split.

        Args:
            split: Data split (VAL or TEST)
            language: Language code

        Returns
        -------
            ClassificationDataset with job titles and multi-label skill assignments
        """
        if split == DatasetSplit.VAL:
            return self._load_val(language)
        if split == DatasetSplit.TEST:
            return self._load_test(language)
        raise ValueError(f"Split '{split}' not supported. Use VAL or TEST")

    def _load_test(self, language: Language) -> ClassificationDataset:
        """Load test data from ESCO occupation-skill relations."""
        target_esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = target_esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        # Build occupation -> skills mapping from ESCO relations
        occupation_to_skills: dict[str, list[str]] = {}

        for occupation_uri, skill_uri, _ in target_esco.get_occupation_skill_relations():
            occupation = target_esco.uri_to_preferred_label(
                occupation_uri, entity_type="occupation"
            )
            skill = target_esco.uri_to_preferred_label(skill_uri, entity_type="skill")
            if occupation and skill:
                if occupation not in occupation_to_skills:
                    occupation_to_skills[occupation] = []
                occupation_to_skills[occupation].append(skill)

        texts = list(occupation_to_skills.keys())
        labels = [
            [skill2label[skill] for skill in occupation_to_skills[q] if skill in skill2label]
            for q in texts
        ]

        return ClassificationDataset(
            texts=texts,
            labels=labels,
            label_space=skill_vocab,
            language=language,
        )

    def _load_val(self, language: Language) -> ClassificationDataset:
        """
        Load validation set based on vacancies with job titles.

        Static validation set only available in English.
        """
        if language != Language.EN:
            raise ValueError(
                "Validation set for Job-Skill Classification is only available in English."
            )

        target_esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = target_esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        logger.debug("Loading validation dataset from TechWolf/vacancy-job-to-skill")
        ds = load_dataset("TechWolf/vacancy-job-to-skill")
        assert isinstance(ds, DatasetDict)
        df = ds["validation"].to_pandas()
        assert isinstance(df, pd.DataFrame)

        title_col_name = "vacancy_job_title"
        skills_col_name = "tagged_esco_skills"
        assert set(df.columns) == {title_col_name, skills_col_name}, (
            f"Unexpected columns in dataset: {df.columns}, expected {title_col_name, skills_col_name}"
        )

        # Normalize possible list vs string columns
        queries_col = df[title_col_name].apply(lambda x: [x] if isinstance(x, str) else x).tolist()
        skills_col = df[skills_col_name].apply(lambda x: [x] if isinstance(x, str) else x).tolist()

        # Map skills from original EN v1.2.0 to target version/language via URIs
        original_esco = ESCO(version=self.original_esco_version, language=Language.EN)
        original_skill_uris = original_esco.get_skills_uris()  # EN label -> URI
        target_skill_uris = target_esco.get_skills_uris()  # target label -> URI
        target_uri_to_label = {v: k for k, v in target_skill_uris.items()}  # URI -> target label

        converted_skills: list[list[str]] = []
        for skill_list in skills_col:
            mapped_labels: list[str] = []
            for skill_label_en in skill_list:
                assert skill_label_en in original_skill_uris, (
                    f"Skill label {skill_label_en} not found in original ESCO skill URIs"
                )
                uri = original_skill_uris[skill_label_en]
                if uri in target_uri_to_label:
                    mapped_labels.append(target_uri_to_label[uri])
            converted_skills.append(mapped_labels)

        # Convert to classification format (one text per occupation title)
        texts: list[str] = []
        labels: list[list[int]] = []
        for job_titles, skill_list in zip(queries_col, converted_skills, strict=True):
            for job_title in job_titles:
                texts.append(job_title)
                indices = [skill2label[s] for s in skill_list if s in skill2label]
                labels.append(indices)

        return ClassificationDataset(
            texts=texts,
            labels=labels,
            label_space=skill_vocab,
            language=language,
        )
