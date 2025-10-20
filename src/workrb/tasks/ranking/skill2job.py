"""Skill to Job Ranking Task."""

import pandas as pd
from datasets import DatasetDict, load_dataset

from workrb.data.esco import ESCO
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class ESCOSkill2JobRanking(RankingTask):
    """Skill to Job Ranking Task."""

    def __init__(self, esco_version: str = "1.2.0", **kwargs):
        self.esco_version = esco_version
        self.original_esco_version = "1.2.0"

        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Skill to Job task name."""
        return "Skill to Job"

    @property
    def description(self) -> str:
        """Skill to Job task description."""
        return "Rank ESCO occupations relevant for a given skill"

    @property
    def task_group(self) -> RankingTaskGroup:
        """Skill to Job task group."""
        return RankingTaskGroup.SKILL2JOB

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages are all ESCO languages (val is EN-only)."""
        return list(ESCO.SUPPORTED_ESCO_LANGUAGES)

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages are all ESCO languages."""
        return [Language.EN]

    @property
    def label_type(self) -> LabelType:
        """Label type is multi label (a skill maps to multiple jobs)."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for skills."""
        return ModelInputType.SKILL_NAME

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for jobs."""
        return ModelInputType.JOB_TITLE

    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """
        Load skill-to-job data for a specific split and language.

        Validation set is static and only available in English.
        Test set is generated from ESCO relations for the selected version and language.
        """
        if split == DatasetSplit.TEST:
            return self._load_test(language=language)

        if split == DatasetSplit.VAL:
            return self._load_val(language=language)

        raise ValueError(f"Invalid split: {split}")

    def _load_test(self, language: Language) -> RankingDataset:
        """Load test data for a specific version and language."""
        target_esco = ESCO(version=self.esco_version, language=language)

        # Target space is the occupations vocabulary
        job_vocab = target_esco.get_occupations_vocabulary()
        job2label = {job: i for i, job in enumerate(job_vocab)}

        # Build mapping Skill -> [Occupations] from ESCO relations
        relations = target_esco.get_occupation_skill_relations()

        skill_to_jobs: dict[str, list[str]] = {}
        for occupation_uri, skill_uri, _ in relations:
            occupation = target_esco.uri_to_preferred_label(
                occupation_uri, entity_type="occupation"
            )
            skill = target_esco.uri_to_preferred_label(skill_uri, entity_type="skill")
            if occupation and skill:
                if skill not in skill_to_jobs:
                    skill_to_jobs[skill] = []
                skill_to_jobs[skill].append(occupation)

        query_texts: list[str] = []
        target_indices: list[list[int]] = []
        for skill, jobs in skill_to_jobs.items():
            label_indices = [job2label[j] for j in jobs if j in job2label]
            if label_indices:
                query_texts.append(skill)
                target_indices.append(label_indices)

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=job_vocab,
            language=language,
        )

    def _load_val(self, language: Language) -> RankingDataset:
        """
        Use vacancies with job titles where descriptions yield ESCO skills.

        Validation set is static and only available in English.
        """
        if language != Language.EN:
            raise ValueError("Validation set for Skill-to-Job is only available in English.")

        target_esco = ESCO(version=self.esco_version, language=language)

        # Load validation split from Hugging Face dataset and invert mapping to Skill -> Jobs
        ds = load_dataset("TechWolf/vacancy-job-to-skill")
        assert isinstance(ds, DatasetDict)
        df = ds["validation"].to_pandas()
        assert isinstance(df, pd.DataFrame)

        title_col_name = "vacancy_job_title"
        skills_col_name = "tagged_esco_skills"
        assert set(df.columns) == {title_col_name, skills_col_name}

        # Target space is all the job titles in the validation set
        job_vocab = sorted(df[title_col_name].unique().tolist())
        job2label = {job: i for i, job in enumerate(job_vocab)}

        jobs_per_row = df[title_col_name].tolist()
        skills_per_row = df[skills_col_name].tolist()

        # Convert original EN skill labels to target version/language via URIs
        orig_esco = ESCO(version=self.original_esco_version, language=Language.EN)
        orig_skill_uris = orig_esco.get_skills_uris()  # label -> uri (original)
        target_skill_uris = target_esco.get_skills_uris()  # label -> uri (target)
        target_uri_to_skill = {v: k for k, v in target_skill_uris.items()}

        # For each row, convert skills using URI mapping
        converted_skill_rows: list[list[str]] = []
        for skill_list in skills_per_row:
            converted = [
                target_uri_to_skill[orig_skill_uris[s]]
                for s in skill_list
                if s in orig_skill_uris and orig_skill_uris[s] in target_uri_to_skill
            ]
            converted_skill_rows.append(converted)

        # Build Skill -> set(Jobs) mapping from rows
        skill_to_jobs_map: dict[str, set[str]] = {}
        for job_title, skill_list in zip(jobs_per_row, converted_skill_rows, strict=True):
            for skill in skill_list:
                if skill not in skill_to_jobs_map:
                    skill_to_jobs_map[skill] = set()
                if isinstance(job_title, str):
                    skill_to_jobs_map[skill].add(job_title)

        # Convert mapping to ranking format using job2label
        query_texts: list[str] = []
        target_indices: list[list[int]] = []
        for skill, jobs in skill_to_jobs_map.items():
            indices = [job2label[j] for j in jobs if j in job2label]
            if indices:
                query_texts.append(skill)
                target_indices.append(indices)

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=job_vocab,
            language=language,
        )

    @property
    def citation(self) -> str:
        """Skill to Job task citation."""
        return """UWE-PLACEHOLDER"""
