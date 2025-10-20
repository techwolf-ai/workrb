"""Job-to-Skills Ranking Task."""

import pandas as pd

from workbench.data.esco import ESCO
from workbench.data.input_types import ModelInputType
from workbench.data.utils import get_data_path
from workbench.registry import register_task
from workbench.tasks.abstract.base import DatasetSplit, LabelType, Language
from workbench.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup


@register_task()
class ESCOJob2SkillRanking(RankingTask):
    """Job-to-Skills Ranking Task."""

    # Static validation dataset location (within repo data folder)
    local_data_path = get_data_path("techwolf_vacancies")
    raw_val_parquet = "job_skill_val.parquet"
    original_esco_version = "1.2.0"

    def __init__(self, esco_version: str = "1.2.0", **kwargs):
        self.esco_version = esco_version

        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Task name."""
        return "Job to Skills"

    @property
    def description(self) -> str:
        """Task description."""
        return "Recommend relevant ESCO skills for occupations"

    @property
    def task_group(self) -> RankingTaskGroup:
        """Task group identifier."""
        return RankingTaskGroup.JOB2SKILL

    @property
    def supported_query_languages(self) -> list[Language]:
        """Queries (occupation titles) can be in any ESCO language for test; val is EN-only."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Target skills vocabulary spans all ESCO languages."""
        return list(ESCO.SUPPORTED_ESCO_LANGUAGES)

    @property
    def label_type(self) -> LabelType:
        """Multi-label ranking of skills for each job."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for jobs."""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for skills."""
        return ModelInputType.SKILL_NAME

    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """
        Load job-to-skills data for a specific split and language.

        Static validation set only available in English.
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

        query_texts = list(occupation_to_skills.keys())
        target_indices = [
            [skill2label[skill] for skill in occupation_to_skills[q] if skill in skill2label]
            for q in query_texts
        ]

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=skill_vocab,
            language=language,
        )

    def _load_val(self, language: Language) -> RankingDataset:
        """
        Validation set based on vacancies with job titles, where description is used to extract ESCO skills.

        Static validation set only available in English.
        """
        if language != Language.EN:
            raise ValueError("Validation set for Job-to-Skills is only available in English.")

        target_esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = target_esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        data_path = self.local_data_path / self.raw_val_parquet
        df = pd.read_parquet(data_path)

        # Normalize possible list vs string columns
        queries_col = df["title"].apply(lambda x: [x] if isinstance(x, str) else x).tolist()
        skills_col = df["skill_name"].apply(lambda x: [x] if isinstance(x, str) else x).tolist()

        # Map skills from original EN v1.2.0 to target version/language via URIs
        original_esco = ESCO(version=self.original_esco_version, language=Language.EN)
        original_skill_uris = original_esco.get_skills_uris()  # EN label -> URI
        target_skill_uris = target_esco.get_skills_uris()  # target label -> URI
        target_uri_to_label = {v: k for k, v in target_skill_uris.items()}  # URI -> target label

        converted_skills: list[list[str]] = []
        for skill_list in skills_col:
            mapped_labels: list[str] = []
            for skill_label_en in skill_list:
                if skill_label_en in original_skill_uris:
                    uri = original_skill_uris[skill_label_en]
                    if uri in target_uri_to_label:
                        mapped_labels.append(target_uri_to_label[uri])
            converted_skills.append(mapped_labels)

        # Convert to ranking format (one query per occupation title)
        query_texts: list[str] = []
        target_indices: list[list[int]] = []
        for job_titles, skill_list in zip(queries_col, converted_skills, strict=True):
            for job_title in job_titles:
                query_texts.append(job_title)
                indices = [skill2label[s] for s in skill_list if s in skill2label]
                target_indices.append(indices)

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=skill_vocab,
            language=language,
        )
