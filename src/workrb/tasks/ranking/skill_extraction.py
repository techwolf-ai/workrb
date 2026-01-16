"""Skill Extraction Ranking Tasks."""

import pandas as pd
from datasets import Dataset, load_dataset

from workrb.data.esco import ESCO
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import (
    RankingDataset,
    RankingTask,
    RankingTaskGroup,
)
from workrb.types import ModelInputType


class BaseESCOSkillExtractRanking(RankingTask):
    """Base ESCO Skill Extraction Ranking Task."""

    def __init__(
        self,
        hf_name: str,
        esco_version: str = "1.1.0",
        orig_esco_version: str = "1.1.0",
        **kwargs,
    ):
        """
        Initialize ESCO Skill Extraction Ranking Task.

        Args:
            hf_name: Name of the Hugging Face dataset
            esco_version: Target version of ESCO to use
            orig_esco_version: Version of ESCO used for tagging the original data
            **kwargs: Additional arguments for base class
        """
        self.esco_version = esco_version
        self.hf_name = hf_name
        self.orig_esco_version = orig_esco_version
        super().__init__(**kwargs)

    @property
    def task_group(self) -> RankingTaskGroup:
        """Skill extraction house task group."""
        return RankingTaskGroup.SKILL_EXTRACTION

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
        """Label type is multi-label."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for skill extraction sentences."""
        return ModelInputType.SKILL_SENTENCE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for ESCO skills."""
        return ModelInputType.SKILL_NAME

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load skill extraction data for a specific split and dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            RankingDataset object
        """
        language = Language(dataset_id)
        # Load data
        split_names = {DatasetSplit.TEST: "test", DatasetSplit.VAL: "validation"}
        dataset = load_dataset(self.hf_name, split=split_names[split])
        assert isinstance(dataset, Dataset)
        df = dataset.to_pandas()
        assert isinstance(df, pd.DataFrame)

        # If ESCO version is not 1.1.0 and / or language is not en, we need to translate the skills
        if self.esco_version != self.orig_esco_version or language != Language.EN:
            original_esco = ESCO(version=self.orig_esco_version, language=Language.EN)
            original_skill_uris = original_esco.get_skills_uris()
            original_uris_to_skill = {v: k for k, v in original_skill_uris.items()}

            target_esco = ESCO(version=self.esco_version, language=language)
            target_skill_uris = target_esco.get_skills_uris()
            target_uris_to_skill = {v: k for k, v in target_skill_uris.items()}

            original_skill_to_target_skill = {}
            for uri, orig_skill in original_uris_to_skill.items():
                if uri in target_uris_to_skill:
                    original_skill_to_target_skill[orig_skill] = target_uris_to_skill[uri]

            df["label"] = df["label"].apply(
                lambda orig_skill: original_skill_to_target_skill.get(orig_skill)
            )
            # Drop rows where label is None
            df = df[df["label"].notna()].reset_index(drop=True).copy()

        grouped_df = df.groupby("sentence")["label"].apply(list).reset_index()

        # Load ESCO skill vocabulary for target version/language
        esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = esco.get_skills_vocabulary()
        skill2label = {skill: i for i, skill in enumerate(skill_vocab)}

        # Filter skills that exist in vocabulary (Excludes "LABEL NOT PRESENT" and "UNDERSPECIFIED")
        filtered_queries = []
        filtered_labels = []
        for query, skill_list in zip(grouped_df["sentence"], grouped_df["label"], strict=True):
            filtered_skill_list = [skill for skill in skill_list if skill in skill2label]
            if len(filtered_skill_list) == 0:
                continue
            filtered_queries.append(query)
            filtered_labels.append([skill2label[skill] for skill in filtered_skill_list])

        return RankingDataset(
            query_texts=filtered_queries,
            target_indices=filtered_labels,
            target_space=skill_vocab,
            dataset_id=dataset_id,
        )


@register_task()
class HouseSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from House Dataset Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="TechWolf/skill-extraction-house", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction house task name."""
        return "Skill Extraction House"

    @property
    def description(self) -> str:
        """Skill extraction house task description."""
        return "Extract skills from general text descriptions in the HOUSE subset of CAREER."

    @property
    def citation(self) -> str:
        """Skill extraction house task citation."""
        return """@inproceedings{decorte2022design,
  articleno    = {{4}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas}},
  booktitle    = {{Proceedings of the 2nd Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2022)}},
  editor       = {{Kaya, Mesut and Bogers, Toine and Graus, David and Mesbah, Sepideh and Johnson, Chris and Gutiérrez, Francisco}},
  isbn         = {{9781450398565}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Seatle, USA}},
  pages        = {{7}},
  publisher    = {{CEUR}},
  title        = {{Design of negative sampling strategies for distantly supervised skill extraction}},
  url          = {{https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf}},
  volume       = {{3218}},
  year         = {{2022}},
}
"""


@register_task()
class TechSkillExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from Tech Dataset Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="TechWolf/skill-extraction-tech", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction tech task name."""
        return "Skill Extraction Tech"

    @property
    def description(self) -> str:
        """Skill extraction tech task description."""
        return "Extract skills from technical text descriptions in the TECH subset of CAREER."

    @property
    def citation(self) -> str:
        """Skill extraction tech task citation."""
        return """@inproceedings{decorte2022design,
  articleno    = {{4}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Deleu, Johannes and Develder, Chris and Demeester, Thomas}},
  booktitle    = {{Proceedings of the 2nd Workshop on Recommender Systems for Human Resources (RecSys-in-HR 2022)}},
  editor       = {{Kaya, Mesut and Bogers, Toine and Graus, David and Mesbah, Sepideh and Johnson, Chris and Gutiérrez, Francisco}},
  isbn         = {{9781450398565}},
  issn         = {{1613-0073}},
  language     = {{eng}},
  location     = {{Seatle, USA}},
  pages        = {{7}},
  publisher    = {{CEUR}},
  title        = {{Design of negative sampling strategies for distantly supervised skill extraction}},
  url          = {{https://ceur-ws.org/Vol-3218/RecSysHR2022-paper_4.pdf}},
  volume       = {{3218}},
  year         = {{2022}},
}
"""


@register_task()
class SkillSkapeExtractRanking(BaseESCOSkillExtractRanking):
    """Skill Extraction from SkillSkape Ranking Task."""

    orig_esco_version = "1.1.0"

    def __init__(self, esco_version: str = "1.1.0", **kwargs):
        self.esco_version = esco_version
        super().__init__(hf_name="jjzha/skillskape", **kwargs)

    @property
    def name(self) -> str:
        """Skill extraction SkillSkape task name."""
        return "Skill Extraction SkillSkape"

    @property
    def description(self) -> str:
        """Skill extraction from SkillSkape task description."""
        return "Extract skills from text descriptions."

    @property
    def citation(self) -> str:
        """Skill extraction SkillSkape task citation."""
        return """@inproceedings{magron-etal-2024-jobskape,
  title     = {{JobSkape: A Framework for Generating Synthetic Job Postings to Enhance Skill Matching}},
  author    = {{Magron, Antoine and Dai, Anna and Zhang, Mike and Montariol, Syrielle and Bosselut, Antoine}},
  editor    = {{Hruschka, Estevam and Lake, Thom and Otani, Naoki and Mitchell, Tom}},
  booktitle = {{Proceedings of the First Workshop on Natural Language Processing for Human Resources (NLP4HR 2024)}},
  month     = {{mar}},
  year      = {{2024}},
  address   = {{St. Julian's, Malta}},
  publisher = {{Association for Computational Linguistics}},
  url       = {{https://aclanthology.org/2024.nlp4hr-1.4/}},
  pages     = {{43--58}}
}
"""
