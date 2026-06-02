"""Shared base for BEIR-layout graded-relevance ranking tasks.

Several tasks publish graded (0-4) relevance annotations on the Hugging Face
Hub following the BEIR convention (``queries``, ``corpus``, ``qrels`` configs).
They differ only in their task group and query input type (e.g. skill
extraction from sentences vs skill normalization from surface terms), so the
data-loading logic lives here and concrete tasks supply the task-specific
properties.
"""

import pandas as pd
from datasets import Dataset, load_dataset

from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask
from workrb.types import ModelInputType


class GradedBEIRRankingTask(RankingTask):
    """Base class for BEIR-layout graded ranking tasks.

    Reads the ``queries``, ``corpus`` and ``qrels`` configs published on the
    Hugging Face Hub via ``load_dataset``. The target_space is the corpus's
    ``title`` column (ESCO preferred labels, in corpus order); qrels are
    expected to contain only non-zero judgments (absent items are implicit
    grade 0).

    Which splits a task exposes depends on what the underlying dataset
    publishes: some release both a validation and a test split, others only
    one. Because several tasks share this loader, the supported splits are
    declared per subclass via :attr:`split_to_hf_split` (which maps each
    supported :class:`DatasetSplit` to the HF split name its
    ``queries``/``qrels`` configs live under) rather than hardcoded with an
    inline guard as in the single-split tasks (e.g. ``MELORanking``). The
    default exposes only the validation split, the common case for in-progress
    benchmark datasets whose test split is withheld.

    Concrete subclasses set ``hf_name`` via ``__init__`` and supply the
    task-specific ``task_group``, ``query_input_type``, ``name``,
    ``description`` and ``citation``.
    """

    def __init__(self, hf_name: str, **kwargs):
        """Initialize the task.

        Args:
            hf_name: Name of the Hugging Face dataset (BEIR layout).
            **kwargs: Additional arguments for the base class.
        """
        self.hf_name = hf_name
        super().__init__(**kwargs)

    @property
    def split_to_hf_split(self) -> dict[DatasetSplit, str]:
        """Map each supported split to the HF split name backing it.

        The corpus config is always loaded from the ``corpus`` split; this
        mapping only governs the ``queries`` and ``qrels`` configs. Override to
        expose more or fewer splits, e.g. ``{DatasetSplit.VAL: "validation",
        DatasetSplit.TEST: "test"}`` for a dataset that releases both.
        """
        return {DatasetSplit.VAL: "validation"}

    @property
    def supported_query_languages(self) -> list[Language]:
        """Annotations are released in English only at this stage."""
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        """The corpus titles are released in English only at this stage."""
        return [Language.EN]

    @property
    def label_type(self) -> LabelType:
        """Label type is multi-label."""
        return LabelType.MULTI_LABEL

    @property
    def target_input_type(self) -> ModelInputType:
        """Target input type for ESCO skills."""
        return ModelInputType.SKILL_NAME

    @property
    def default_metrics(self) -> list[str]:
        """Default metrics include nDCG to leverage the graded labels.

        ``ndcg`` without a cutoff scores the full ranking (k = |target_space|).
        """
        return ["ndcg", "ndcg@5", "ndcg@10", "map", "rp@10", "mrr"]

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load BEIR-style graded annotations and convert to a RankingDataset."""
        hf_split = self.split_to_hf_split.get(split)
        if hf_split is None:
            supported = ", ".join(sorted(s.value for s in self.split_to_hf_split))
            raise ValueError(
                f"Split '{split.value}' not supported for {type(self).__name__}: "
                f"only [{supported}] {'is' if len(self.split_to_hf_split) == 1 else 'are'} "
                f"annotated for this dataset."
            )

        queries_ds = load_dataset(self.hf_name, "queries", split=hf_split)
        corpus_ds = load_dataset(self.hf_name, "corpus", split="corpus")
        qrels_ds = load_dataset(self.hf_name, "qrels", split=hf_split)
        assert isinstance(queries_ds, Dataset)
        assert isinstance(corpus_ds, Dataset)
        assert isinstance(qrels_ds, Dataset)
        queries_df = queries_ds.to_pandas()
        corpus_df = corpus_ds.to_pandas()
        qrels_df = qrels_ds.to_pandas()
        assert isinstance(queries_df, pd.DataFrame)
        assert isinstance(corpus_df, pd.DataFrame)
        assert isinstance(qrels_df, pd.DataFrame)

        # target_space is the corpus titles in corpus order; URIs map by row index.
        target_space = corpus_df["title"].tolist()
        uri_to_idx = {uri: i for i, uri in enumerate(corpus_df["_id"])}

        qrels_df["target_idx"] = qrels_df["corpus-id"].map(uri_to_idx)
        # Every qrel should resolve; if any don't, surface the issue rather than silently dropping.
        unresolved = qrels_df["target_idx"].isna().sum()
        assert unresolved == 0, (
            f"{unresolved} qrel rows reference corpus-ids not present in the corpus config"
        )
        qrels_df["target_idx"] = qrels_df["target_idx"].astype(int)

        id_to_query = dict(zip(queries_df["_id"], queries_df["text"], strict=True))
        qrels_df["sentence"] = qrels_df["query-id"].map(id_to_query)

        grouped = qrels_df.groupby("sentence")
        filtered_queries: list[str] = []
        filtered_indices: list[list[int]] = []
        filtered_relevance: list[list[float]] = []
        for sentence, group in grouped:
            filtered_queries.append(str(sentence))
            filtered_indices.append(group["target_idx"].tolist())
            filtered_relevance.append([float(s) for s in group["score"].tolist()])

        return RankingDataset(
            query_texts=filtered_queries,
            target_indices=filtered_indices,
            target_space=target_space,
            dataset_id=dataset_id,
            target_relevance=filtered_relevance,
        )
