"""Job Title Similarity ranking task using Zbib et al. (2022) and Deniz et al. (2024) datasets."""

from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class JobTitleSimilarityRanking(RankingTask):
    """
    Job Title Similarity ranking task based on Zbib et al. (2022) and Deniz et al. (2024).

    This task evaluates a model's ability to rank job titles by semantic similarity to a
    query job title. Given a query job title, the model must rank corpus job titles such
    that semantically similar ones appear higher than non-similar ones.

    Notes
    -----
    HuggingFace Dataset: https://huggingface.co/datasets/Avature/Job-Title-Similarity

    Languages: en, de, es, fr, it, ja, ko, nl, pl, pt, zh.

    Each language is a HuggingFace subset, and contains two splits:
    - ``queries``: ~105 query job titles with indices of relevant corpus elements
    - ``corpus``: ~2,500 job titles to rank

    Each query has binary relevance annotations indicating which corpus job titles are
    semantically similar (multi-label).

    Example (English):
    - Query: "Food Technologist"
    - This query has 9 relevant corpus titles out of 2,619: "Food Scientist",
      "Food Engineer", "Flavorist", among others.

    Difference between this task and Job Normalization:
    Job Normalization measures whether the model identifies the single canonical form of a
    job title. Job Title Similarity measures whether the model ranks semantically similar
    titles above dissimilar ones.
    """

    SUPPORTED_DATASET_LANGUAGES = [
        Language.DE,
        Language.EN,
        Language.ES,
        Language.FR,
        Language.IT,
        Language.JA,
        Language.KO,
        Language.NL,
        Language.PL,
        Language.PT,
        Language.ZH,
    ]

    @property
    def name(self) -> str:
        """Job Title Similarity task name."""
        return "Job Title Similarity"

    @property
    def description(self) -> str:
        """Job Title Similarity task description."""
        return "Rank job titles in a corpus based on their semantic similarity to query job titles."

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
        return self.SUPPORTED_DATASET_LANGUAGES

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
        """Multi-label ranking for semantically similar job titles."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for job titles."""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for job titles."""
        return ModelInputType.JOB_TITLE

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load Job Title Similarity data from the HuggingFace dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            RankingDataset object
        """
        language = Language(dataset_id)

        if split != DatasetSplit.TEST:
            raise ValueError(f"Split '{split}' not supported. Use TEST")

        if language not in self.SUPPORTED_DATASET_LANGUAGES:
            raise ValueError(f"Language '{language}' not supported.")

        ds = load_dataset("Avature/Job-Title-Similarity", language.value)

        queries = list(ds["queries"]["text"])
        relevancy_labels = list(ds["queries"]["labels"])
        corpus = list(ds["corpus"]["text"])

        return RankingDataset(queries, relevancy_labels, corpus, dataset_id=dataset_id)

    @property
    def citation(self) -> str:
        """Job Title Similarity task citation."""
        return """
@article{zbib2022Learning,
      title={{Learning Job Titles Similarity from Noisy Skill Labels}},
      author={Rabih Zbib and
              Lucas Alvarez Lacasa and
              Federico Retyk and
              Rus Poves and
              Juan Aizpuru and
              Hermenegildo Fabregat and
              Vaidotas Šimkus and
              Emilia García-Casademont},
      journal={{FEAST, ECML-PKDD 2022 Workshop}},
      year={{2022}},
      url="https://feast-ecmlpkdd.github.io/archive/2022/papers/FEAST2022_paper_4972.pdf"
}
@inproceedings{deniz2024Combined,
  title        = {Combined Unsupervised and Contrastive Learning for Multilingual Job Recommendations},
  author       = {Daniel Deniz and
                  Federico Retyk and
                  Laura García-Sardiña and
                  Hermenegildo Fabregat and
                  Luis Gasco and
                  Rabih Zbib},
  booktitle    = {Proceedings of the 4th Workshop on Recommender Systems for Human Resources
                  (RecSys in {HR} 2024), in conjunction with the 18th {ACM} Conference on
                  Recommender Systems},
  year         = {2024},
}
"""
