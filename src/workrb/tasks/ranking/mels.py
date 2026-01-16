"""Skill Normalization ranking task using datasets inspired in MELO (Retyk et al., 2024)."""

from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class MELSRanking(RankingTask):
    """
    MELS (Multilingual Entity Linking of Skills): Skills Normalization ranking task using datasets
    that were inspired in MELO (Retyk et al., 2024).

    MELS is a sibling dataset to MELO (Multilingual Entity Linking of Occupations). Both datasets
    were built using the same methodology and the same type of source data: crosswalks between
    national taxonomies and ESCO, published by official labor-related organizations from EU member
    states.

    The difference is the entity type:
     - MELO links occupation mentions (job titles) to ESCO Occupations
     - MELS links skill mentions to ESCO Skills
    """

    MELS_LANGUAGES = [
        Language.DE,
        Language.EN,
        Language.FR,
        Language.NL,
        Language.SV,
    ]

    LANGUAGE_TO_DATASETS = [
        "bel_q_fr_c_fr",
        "bel_q_fr_c_en",
        "bel_q_nl_c_nl",
        "bel_q_nl_c_en",
        "deu_q_de_c_de",
        "deu_q_de_c_en",
        "swe_q_sv_c_sv",
        "swe_q_sv_c_en",
    ]

    @property
    def name(self) -> str:
        """MELS task name."""
        return "MELS"

    @property
    def description(self) -> str:
        """MELS task description."""
        return "Skill Normalization ranking task into ESCO."

    @property
    def default_metrics(self) -> list[str]:
        return ["mrr", "hit@1", "hit@5", "hit@10"]

    @property
    def task_group(self) -> RankingTaskGroup:
        """Skill Normalization task group."""
        return RankingTaskGroup.SKILL_NORMALIZATION

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages."""
        return self.MELS_LANGUAGES

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages."""
        return self.MELS_LANGUAGES

    @property
    def split_test_fraction(self) -> float:
        """Fraction of data to use for test split."""
        return 1.0

    @property
    def label_type(self) -> LabelType:
        """Multi-label ranking for Skill Normalization."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for skill names."""
        return ModelInputType.SKILL_NAME

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for skill names."""
        return ModelInputType.SKILL_NAME

    def _parse_dataset_id(self, dataset_id: str) -> tuple[str, list[str]]:
        """Parse dataset_id into query language and corpus languages.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier in format: <country>_q_<query_lang>_c_<corpus_lang>[_<more>].

        Returns
        -------
        tuple[str, list[str]]
            Tuple of (query_language_code, list_of_corpus_language_codes).
        """
        parts = dataset_id.split("_")
        # Find the index of 'q' and 'c' markers
        q_idx = parts.index("q")
        c_idx = parts.index("c")
        # Query language is between 'q' and 'c'
        query_lang = "_".join(parts[q_idx + 1 : c_idx])
        # Corpus languages are everything after 'c'
        corpus_langs = parts[c_idx + 1 :]
        return query_lang, corpus_langs

    def languages_to_dataset_ids(self, languages: list[Language]) -> list[str]:
        """Filter datasets based on the target languages.

        Parameters
        ----------
        languages : list[Language]
            List of Language enums requested for evaluation.

        Returns
        -------
        list[str]
            List of dataset identifier strings.
        """
        lang_codes = {lang.value for lang in languages}

        result = []
        for dataset_id in self.LANGUAGE_TO_DATASETS:
            query_lang, corpus_langs = self._parse_dataset_id(dataset_id)
            # Check if all involved languages are in the target set
            all_langs = {query_lang} | set(corpus_langs)
            if all_langs <= lang_codes:
                result.append(dataset_id)

        return result

    def get_dataset_language(self, dataset_id: str) -> Language | None:
        """Return the language of a dataset if it's monolingual, None otherwise.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        Language or None
            Language enum if this is a monolingual dataset, None for cross-language datasets.
        """
        query_lang, corpus_langs = self._parse_dataset_id(dataset_id)

        # Not monolingual if multiple corpus languages or query != corpus
        if len(corpus_langs) != 1 or query_lang != corpus_langs[0]:
            return None

        return Language(query_lang)

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load MELS data from the HuggingFace dataset.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.
        split : DatasetSplit
            Dataset split to load.

        Returns
        -------
        RankingDataset
            RankingDataset object.
        """
        if split != DatasetSplit.TEST:
            raise ValueError(f"Split '{split}' not supported. Use TEST")

        if dataset_id not in self.dataset_ids:
            raise ValueError(f"Dataset '{dataset_id}' not supported.")

        ds = load_dataset("Avature/MELS-Benchmark", dataset_id)

        queries = list(ds["queries"]["text"])
        relevancy_labels = list(ds["queries"]["labels"])
        corpus = list(ds["corpus"]["text"])

        return RankingDataset(
            queries, relevancy_labels, corpus, dataset_id=dataset_id, allow_duplicate_targets=True
        )

    @property
    def citation(self) -> str:
        """MELS task citation."""
        return """
@inproceedings{retyk2024melo,
  title        = {{MELO: An Evaluation Benchmark for Multilingual Entity Linking of Occupations}},
  author       = {Federico Retyk and Luis Gasco and Casimiro Pio Carrino and Daniel Deniz and Rabih Zbib},
  booktitle    = {Proceedings of the 4th Workshop on Recommender Systems for Human Resources
                  (RecSys in {HR} 2024), in conjunction with the 18th {ACM} Conference on
                  Recommender Systems},
  year         = {2024},
  url          = {https://recsyshr.aau.dk/wp-content/uploads/2024/10/RecSysHR2024-paper_2.pdf},
}
"""
