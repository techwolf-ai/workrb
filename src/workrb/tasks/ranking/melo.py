"""Job Normalization ranking task using the MELO Benchmark (Retyk et al., 2024)."""

from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import DatasetLanguages, ModelInputType


@register_task()
class MELORanking(RankingTask):
    """
    MELO: Multilingual Entity Linking of Occupations (Retyk et al., 2024).

    **Scope.** Job Normalization ranking task using datasets from the MELO Benchmark.
    It evaluates entity linking of occupation mentions (job titles) to the ESCO
    Occupations taxonomy, posed as a ranking problem. It is built from
    high-quality crosswalks between national occupation taxonomies and ESCO, published
    by official labor-related organizations across EU member states and the USA. It
    includes 48 datasets spanning 23 countries and 21 languages.

    **Task structure.** Each element in a national taxonomy becomes a query (a job
    title). The corpus consists of occupation surface forms from ESCO concepts in the
    corpus language(s) — each concept may contribute multiple synonymous names. The
    goal is to rank the correct ESCO occupation(s) for each query. Relevance labels
    are binary and derived at the concept level from the crosswalk: all surface forms
    of a relevant concept are marked as relevant. Queries may map to one or more
    concepts (multi-label). Only a test split is available (no train/validation).

    **Dataset variants.** Most countries have two variants: a **monolingual** variant
    where both queries and corpus are in the same language (e.g., ``aut_q_de_c_de`` —
    Austrian queries in German, ESCO corpus in German), and a **cross-lingual** variant
    where queries are in the national language and the corpus is in English (e.g.,
    ``aut_q_de_c_en`` — Austrian queries in German, ESCO corpus in English). Belgium
    has four variants (two per official language: French and Dutch). The USA dataset
    includes a monolingual English variant and a multilingual corpus variant.

    **Naming convention.** Dataset IDs follow the pattern
    ``{country}_q_{query_lang}_c_{corpus_lang(s)}``, where ``{country}`` is the
    ISO 3166-1 alpha-3 country code (e.g., ``aut`` for Austria), ``q_{lang}`` is the
    query language as an ISO 639-1 code, and ``c_{lang(s)}`` is one or more corpus
    languages as ISO 639-1 codes.

    Examples
    --------
    In the ``ita_q_it_c_it`` dataset, Italian occupation names from the Italian
    national taxonomy (e.g., query: "Vigili del fuoco") must be matched against ESCO
    Occupations in Italian (e.g., corpus: "vigile del fuoco", "pompiere"). The
    ``ita_q_it_c_en`` cross-lingual variant uses the same Italian queries but an
    English ESCO corpus.

    Notes
    -----
    - Paper: Retyk et al. (2024), "MELO: An Evaluation Benchmark for Multilingual
      Entity Linking of Occupations"
    - HuggingFace: https://huggingface.co/datasets/Avature/MELO-Benchmark
    """

    MELO_LANGUAGES = [
        Language.BG,
        Language.CS,
        Language.DA,
        Language.DE,
        Language.EN,
        Language.ES,
        Language.ET,
        Language.FR,
        Language.HR,
        Language.HU,
        Language.IT,
        Language.LT,
        Language.LV,
        Language.NL,
        Language.NO,
        Language.PL,
        Language.PT,
        Language.RO,
        Language.SK,
        Language.SL,
        Language.SV,
    ]

    MELO_DATASET_IDS = [
        "aut_q_de_c_de",
        "aut_q_de_c_en",
        "bel_q_fr_c_fr",
        "bel_q_fr_c_en",
        "bel_q_nl_c_nl",
        "bel_q_nl_c_en",
        "bgr_q_bg_c_bg",
        "bgr_q_bg_c_en",
        "cze_q_cs_c_cs",
        "cze_q_cs_c_en",
        "deu_q_de_c_de",
        "deu_q_de_c_en",
        "dnk_q_da_c_da",
        "dnk_q_da_c_en",
        "esp_q_es_c_en",
        "esp_q_es_c_es",
        "est_q_et_c_en",
        "est_q_et_c_et",
        "fra_q_fr_c_en",
        "fra_q_fr_c_fr",
        "hrv_q_hr_c_en",
        "hrv_q_hr_c_hr",
        "hun_q_hu_c_en",
        "hun_q_hu_c_hu",
        "ita_q_it_c_en",
        "ita_q_it_c_it",
        "ltu_q_lt_c_en",
        "ltu_q_lt_c_lt",
        "lva_q_lv_c_en",
        "lva_q_lv_c_lv",
        "nld_q_nl_c_en",
        "nld_q_nl_c_nl",
        "nor_q_no_c_en",
        "nor_q_no_c_no",
        "pol_q_pl_c_en",
        "pol_q_pl_c_pl",
        "prt_q_pt_c_en",
        "prt_q_pt_c_pt",
        "rou_q_ro_c_en",
        "rou_q_ro_c_ro",
        "svk_q_sk_c_en",
        "svk_q_sk_c_sk",
        "svn_q_sl_c_en",
        "svn_q_sl_c_sl",
        "swe_q_sv_c_en",
        "swe_q_sv_c_sv",
        "usa_q_en_c_de_en_es_fr_it_nl_pl_pt",
        "usa_q_en_c_en",
    ]

    @property
    def name(self) -> str:
        """MELO task name."""
        return "MELO"

    @property
    def description(self) -> str:
        """MELO task description."""
        return "Job Normalization ranking task into ESCO."

    @property
    def default_metrics(self) -> list[str]:
        return ["mrr", "hit@1", "hit@5", "hit@10"]

    @property
    def task_group(self) -> RankingTaskGroup:
        """Job Normalization task group."""
        return RankingTaskGroup.JOB_NORMALIZATION

    @property
    def supported_query_languages(self) -> list[Language]:
        """Supported query languages."""
        return self.MELO_LANGUAGES

    @property
    def supported_target_languages(self) -> list[Language]:
        """Supported target languages."""
        return self.MELO_LANGUAGES

    @property
    def split_test_fraction(self) -> float:
        """Fraction of data to use for test split."""
        return 1.0

    @property
    def label_type(self) -> LabelType:
        """Multi-label ranking for Job Normalization."""
        return LabelType.MULTI_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        """Query input type for job titles."""
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for job titles."""
        return ModelInputType.JOB_TITLE

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
        for dataset_id in self.MELO_DATASET_IDS:
            query_lang, corpus_langs = self._parse_dataset_id(dataset_id)
            # Check if all involved languages are in the target set
            all_langs = {query_lang} | set(corpus_langs)
            if all_langs <= lang_codes:
                result.append(dataset_id)

        return result

    def get_dataset_languages(self, dataset_id: str) -> DatasetLanguages:
        """Map a dataset ID to its input/output languages.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier in format: ``<country>_q_<query_lang>_c_<corpus_lang>[_<more>]``.

        Returns
        -------
        DatasetLanguages
            Named tuple with ``input_languages`` (query) and ``output_languages`` (corpus).
        """
        query_lang, corpus_langs = self._parse_dataset_id(dataset_id)
        return DatasetLanguages(
            input_languages=frozenset({Language(query_lang)}),
            output_languages=frozenset(Language(lang) for lang in corpus_langs),
        )

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load MELO data from the HuggingFace dataset.

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

        ds = load_dataset("Avature/MELO-Benchmark", dataset_id)

        queries = list(ds["queries"]["text"])
        relevancy_labels = list(ds["queries"]["labels"])
        corpus = list(ds["corpus"]["text"])

        return RankingDataset(
            queries, relevancy_labels, corpus, dataset_id=dataset_id, allow_duplicate_targets=True
        )

    @property
    def citation(self) -> str:
        """MELO task citation."""
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
