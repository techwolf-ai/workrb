"""Project brief Freelancer candidate ranking task using a synthetic dataset provided by Malt."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


class _BaseCandidateRanking(RankingTask, ABC):
    """
    Project brief Freelancer candidate ranking task based on Jouanneau et al. (2024) and Jouanneau et al. (2025).

    This task evaluates a model's ability to rank freelancer candidates based on their profile
    by semantic similarity (skill matching fit) to search query or project brief.
    Given a query job title, the model must rank corpus job titles such that semantically similar ones
    appear higher than non-similar ones.

    Notes
    -----
    HuggingFace Dataset: https://huggingface.co/datasets/MaltCompany/Freelancer-Project-Matching

    Languages: en, de, es, fr, nl.
    + cross_lingual to test language agnostic matching

    The dataset contains:
    - ``queries``: 200 search queries
    - ``briefs``: 200 project briefs (title + description)
    - ``profiles``: 4019 candidate profiles to rank each containing:
        - a headline
        - a language
        - a description
        - a list of skills and soft_skills
        - a list of experience:
            - with a title
            - a description
            - a list of skills
    - Skill matching fit ``brief_scores`` 200 * 4019 and ``query_score`` 200 * 4019

    Each query or brief and profile pair are scored. The score indicates the skill matching fit.
    A threshold is applied on the score to binarize the interactions into relevant and non-relevant pairs.
    """

    SUPPORTED_DATASET_LANGUAGES = [
        Language.DE,
        Language.EN,
        Language.ES,
        Language.FR,
        Language.NL,
        Language.CROSS,
    ]

    RELEVANCE_SCORE_THRESHOLD = 0.8

    @property
    def default_metrics(self) -> list[str]:
        return ["map", "rp@5", "rp@10", "mrr"]

    @property
    def task_group(self) -> RankingTaskGroup:
        """Job Title Similarity task group."""
        return RankingTaskGroup.JOBSIM

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages."""
        return self.SUPPORTED_DATASET_LANGUAGES

    @property
    def split_test_fraction(self) -> float:
        """Fraction of data to use for test split."""
        return 1.0

    @property
    def label_type(self) -> LabelType:
        """Multi-label ranking for project-candidate skill job fit."""
        return LabelType.MULTI_LABEL

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for profiles."""
        return ModelInputType.CANDIDATE_PROFILE_STRING

    @staticmethod
    def _candidate_profile_to_str(candidate: dict[str, Any]) -> str:
        experiences = "\n\n".join(
            f"{exp['title']}\n{exp['description']}\n{', '.join(exp['skills'])}"
            for exp in candidate["experiences"]
        )
        return (
            f"{candidate['headline']}\n\n"
            f"Bio:\n{candidate['description']}\n\n"
            f"Skills:\n{','.join(candidate['skills'] + candidate['soft_skills'])}\n\n"
            f"Experiences:\n{experiences}"
        )

    @staticmethod
    @abstractmethod
    def _input_to_str(query: dict[str, str]) -> str:
        pass

    def _load_and_format_data(
        self,
        split: DatasetSplit,
        language: Language,
        dataset_id: str,
        query_key: str,
        score_key: str,
        query_id_column: str,
    ) -> RankingDataset:
        if split != DatasetSplit.TEST:
            raise ValueError(f"Split '{split}' not supported. Use TEST")

        if language not in self.SUPPORTED_DATASET_LANGUAGES:
            raise ValueError(f"Language '{language}' not supported.")

        query_df = pd.DataFrame(
            load_dataset("MaltCompany/Freelancer-Project-Matching", query_key)["test"]
        )
        candidate_df = pd.DataFrame(
            load_dataset("MaltCompany/Freelancer-Project-Matching", "profiles")["test"]
        )
        score_df = pd.DataFrame(
            load_dataset("MaltCompany/Freelancer-Project-Matching", score_key)["test"]
        )

        if language != Language.CROSS:
            candidate_df = candidate_df[candidate_df["language"] == language.value]

        # create labels
        candidate_df = (
            candidate_df.reset_index(drop=True).reset_index(names="idx").sort_values("idx")
        )
        # add labels to scores
        score_df = candidate_df[["profile_id", "idx"]].merge(score_df, how="left", on="profile_id")

        # threshold score and keep relevancy labels
        score_df = (
            score_df.sort_values("idx")
            .groupby(by=query_id_column)
            .apply(
                lambda group: list(group[group["score"] >= self.RELEVANCE_SCORE_THRESHOLD]["idx"]),
                include_groups=False,
            )
            .reset_index(name="relevancy_labels")
        )

        # make sure inputs coincide and process documents' strings
        relevancy_labels = score_df.sort_values(query_id_column)["relevancy_labels"].tolist()
        queries = [
            self._input_to_str(q)
            for q in query_df.sort_values(query_id_column).to_dict(orient="records")
        ]
        corpus = [self._candidate_profile_to_str(p) for p in candidate_df.to_dict(orient="records")]

        return RankingDataset(queries, relevancy_labels, corpus, dataset_id=dataset_id)

    @property
    def citation(self) -> str:
        """Job Title Similarity task citation."""
        return """
@article{jouanneau2024skill,
    title={Skill matching at scale: freelancer-project alignment for efficient multilingual candidate retrieval},
    author={Jouanneau, Warren and Palyart, Marc and Jouffroy, Emma},
    booktitle = {Proceedings of the 4th Workshop on Recommender Systems for Human Resources
      (RecSys in {HR} 2024), in conjunction with the 18th {ACM} Conference on
      Recommender Systems},
    year={2024}
}
@article{jouanneau2025efficient,
  title={An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit},
  author={Jouanneau, Warren and Jouffroy, Emma and Palyart, Marc},
  booktitle = {Proceedings of the 5th Workshop on Recommender Systems for Human Resources
      (RecSys in {HR} 2025), in conjunction with the 19th {ACM} Conference on
      Recommender Systems},
  year={2025}
}
"""


@register_task()
class ProjectCandidateRanking(_BaseCandidateRanking):
    """
    Project brief Freelancer candidate ranking task based on Jouanneau et al. (2024) and Jouanneau et al. (2025).

    This task evaluates a model's ability to rank freelancer candidates (based on their profile) by semantic similarity
    (skill matching fit) to projects briefs. Given a query job title, the model must rank corpus job titles such
    that semantically similar ones appear higher than non-similar ones.

    Notes
    -----
    HuggingFace Dataset: https://huggingface.co/datasets/MaltCompany/Freelancer-Project-Matching

    Languages: en, de, es, fr, nl.
    + cross_lingual to test language agnostic matching

    The dataset contains:
        - ``briefs``: 200 project briefs (title + description)
        - ``profiles``: 4019 candidate profiles to rank each containing:
            - a headline
            - a language
            - a description
            - a list of skills and soft_skills
            - a list of experience:
                - with a title
                - a description
                - a list of skills
        - Skill matching fit ``brief_scores`` 200 * 4019

    Each brief and profile pair are scored. The score indicates the skill matching fit.
    A threshold is applied on the score to binarize the interactions into relevant and non-relevant pairs.
    """

    @property
    def name(self) -> str:
        """Project candidate matching task name."""
        return "Project-Candidate Matching"

    @property
    def description(self) -> str:
        """Project candidate matching task description."""
        return "Rank candidate's profile in a corpus based on their skill based job fit to query project briefs."

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for briefs."""
        return ModelInputType.PROJECT_BRIEF_STRING

    @staticmethod
    def _input_to_str(input_dict: dict[str, str]) -> str:
        return f"{input_dict['title']}\n\n{input_dict['description']}"

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load Job Title Similarity data from the HuggingFace dataset."""
        language = Language(dataset_id)
        return self._load_and_format_data(
            split, language, dataset_id, "briefs", "brief_scores", "brief_id"
        )


@register_task()
class SearchQueryCandidateRanking(_BaseCandidateRanking):
    """
    Search query Freelancer candidate ranking task based on Jouanneau et al. (2024) and Jouanneau et al. (2025).

    This task evaluates a model's ability to rank freelancer candidates (based on their profile) by semantic similarity
    (skill matching fit) to search query. Given a query job title, the model must rank corpus job titles such
    that semantically similar ones appear higher than non-similar ones.

    Notes
    -----
    HuggingFace Dataset: https://huggingface.co/datasets/MaltCompany/Freelancer-Project-Matching

    Languages: en, de, es, fr, nl.
    + cross_lingual to test language agnostic matching

    The dataset contains:
        - ``queries``: 200 search queries
        - ``profiles``: 4019 candidate profiles to rank each containing:
            - a headline
            - a language
            - a description
            - a list of skills and soft_skills
            - a list of experience:
                - with a title
                - a description
                - a list of skills
        - Skill matching fit ``query_score`` 200 * 4019

    Each query and profile pair are scored. The score indicates the skill matching fit.
    A threshold is applied on the score to binarize the interactions into relevant and non-relevant pairs.
    """

    @property
    def name(self) -> str:
        """Project candidate matching task name."""
        return "Query-Candidate Matching"

    @property
    def description(self) -> str:
        """Project candidate matching task description."""
        return "Rank candidate's profile in a corpus based on their skill based job fit to a search query."

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for briefs."""
        return ModelInputType.SEARCH_QUERY_STRING

    @staticmethod
    def _input_to_str(input_dict: dict[str, str]) -> str:
        return input_dict["value"]

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load Job Title Similarity data from the HuggingFace dataset."""
        language = Language(dataset_id)
        return self._load_and_format_data(
            split, language, dataset_id, "queries", "query_scores", "query_id"
        )
