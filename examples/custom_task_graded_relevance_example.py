"""
Custom Graded-Relevance Ranking Task

Extends ``custom_task_example.py`` with graded relevance: each (query, target)
pair carries an ordinal grade instead of a binary relevant/not-relevant flag.

How a graded task differs from a binary one:

1. Pass an additional ``target_relevance`` list to ``RankingDataset``, aligned
   1-to-1 with ``target_indices``. Items NOT listed in ``target_indices`` are
   implicitly grade 0 (irrelevant).
2. Add ``ndcg@k`` to ``default_metrics``. The (2^rel - 1) gain in nDCG uses the
   grades.
3. Choose ``binary_relevance_threshold`` on the task to control which graded
   items count as positives for binary metrics (``map``, ``mrr``, ``recall@k``,
   ``hit@k``, ``rp@k``). Items with relevance >= threshold are positives;
   items below are dropped from the binary positive set but still contribute
   to nDCG.

Why the threshold matters
-------------------------
The threshold *defines* what "relevant" means for the binary metrics on a
graded dataset. Changing it changes the values of MAP/MRR/recall/hit/rp.
nDCG is unaffected — it always sees the full graded list.

- Default ``1e-9``: every listed grade > 0 counts. Numbers match the
  ``target_relevance=None`` (binary-only) case exactly. Safe default if you
  just want graded nDCG without disturbing legacy binary metrics.
- Stricter (e.g. ``2.0`` on a 1-3 scale): only secondary/primary count.
  Nice-to-haves stop counting toward MAP but still help nDCG.

The bottom of this file runs the same dataset under two thresholds so you
can see nDCG stay constant while MAP/MRR/recall shift.

The grading scale is up to the task. Common choices: {1, 2, 3} (this example),
{1, 2, 3, 4} (TREC-style), or fractional in [0, 1]. Values must be non-negative.
"""

import numpy as np

import workrb
from workrb.metrics.ranking import calculate_ranking_metrics
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTaskGroup
from workrb.types import ModelInputType


@register_task()
class GradedJob2SkillTask(workrb.tasks.RankingTask):
    """Job-to-skill ranking with three relevance grades.

    Grade 3 — primary skill (the role is defined by it).
    Grade 2 — secondary skill (clearly expected).
    Grade 1 — nice-to-have (mentioned but not required).
    Grade 0 — irrelevant; never appears in ``target_indices`` (implicit).
    """

    @property
    def name(self) -> str:
        return "GradedJob2SkillTask"

    @property
    def description(self) -> str:
        return "Job-to-skill ranking with graded relevance for nDCG evaluation."

    @property
    def query_input_type(self) -> ModelInputType:
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        return ModelInputType.SKILL_NAME

    @property
    def default_metrics(self) -> list[str]:
        # nDCG@k is the headline metric here. MAP/MRR/recall are kept as a
        # binary sanity check on the high-grade subset (see the threshold
        # below): they treat every passing graded positive as relevant=1.
        return ["ndcg@5", "ndcg@10", "map", "mrr", "recall@5"]

    @property
    def binary_relevance_threshold(self) -> float:
        # Only "secondary" (grade 2) and "primary" (grade 3) skills count as
        # positives for binary metrics; "nice-to-have" (grade 1) items drop
        # out of MAP/MRR/recall but still contribute to nDCG.
        return 2.0

    @property
    def task_group(self) -> RankingTaskGroup:
        return RankingTaskGroup.JOB2SKILL

    @property
    def label_type(self) -> LabelType:
        return LabelType.MULTI_LABEL

    @property
    def supported_query_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        return [Language.EN]

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load a tiny in-memory graded dataset.

        target_indices and target_relevance are aligned 1-to-1: position i in
        ``target_relevance[q]`` is the grade for ``target_indices[q][i]``.
        """
        queries = [
            "Machine learning engineer",
            "Data scientist",
            "Backend software developer",
        ]

        targets = [
            "Python programming",  # 0
            "Machine learning",  # 1
            "Data analysis",  # 2
            "Software engineering",  # 3
            "Statistics",  # 4
            "Deep learning",  # 5
            "Web development",  # 6
            "Database management",  # 7
        ]

        # Each list pairs (skill_idx -> grade). Skills not listed for a query
        # are implicitly grade 0.
        target_indices = [
            [1, 5, 0, 3],  # ML engineer: ML(3), DL(3), Python(2), SWE(1)
            [1, 2, 4, 5],  # Data scientist: ML(3), DA(3), Stats(2), DL(2)
            [0, 3, 7, 6],  # Backend dev: Python(3), SWE(3), DB(2), Web(1)
        ]
        target_relevance = [
            [3.0, 3.0, 2.0, 1.0],
            [3.0, 3.0, 2.0, 2.0],
            [3.0, 3.0, 2.0, 1.0],
        ]

        return RankingDataset(
            query_texts=queries,
            target_indices=target_indices,
            target_space=targets,
            dataset_id=dataset_id,
            target_relevance=target_relevance,
        )


def _demo_threshold_effect() -> None:
    """Show how ``binary_relevance_threshold`` changes binary metrics on the same data.

    Construction is deliberate: a single query with positives at grades {3, 3, 2, 1},
    and a hand-picked ranking that places the grade-1 "nice-to-have" at rank 1.
    That makes the trade-off legible:

    - At threshold ``1e-9`` (default), the nice-to-have counts as a positive, so
      MRR is a perfect 1.0 — even though the model put a low-value item at the top.
    - At threshold ``2.0``, the nice-to-have is dropped from the positive set, so
      MRR collapses to the rank of the first surviving positive.
    - nDCG is invariant: it always sees the full graded list, and the (2^rel - 1)
      gain already discounts the nice-to-have appropriately.

    Same ``prediction_matrix``, same ``target_relevance`` — only the threshold differs.
    """
    target_indices = [[0, 1, 2, 3]]
    target_relevance = [[3.0, 3.0, 2.0, 1.0]]  # primary, primary, secondary, nice-to-have
    n_targets = 8

    # Forced ranking (best -> worst):
    #   rank 1: idx 3 (grade 1, nice-to-have ranked first)
    #   rank 2: idx 0 (grade 3)
    #   rank 5: idx 1 (grade 3, just inside top-5)
    #   rank 7: idx 2 (grade 2, outside top-5)
    order = [3, 0, 5, 6, 1, 7, 2, 4]
    prediction_matrix = np.zeros((1, n_targets), dtype=float)
    for rank, idx in enumerate(order):
        prediction_matrix[0, idx] = float(n_targets - rank)

    metrics = ["ndcg@5", "map", "mrr", "recall@5"]
    thresholds = {
        "1e-9 (every listed grade counts)": 1e-9,
        "2.0  (only grade >= 2 counts)": 2.0,
    }

    print("\n--- Effect of binary_relevance_threshold on the same dataset ---")
    print("Same prediction_matrix, same target_relevance — only the threshold differs.\n")
    header = f"{'threshold':<36}" + "".join(f"{m:>10}" for m in metrics)
    print(header)
    print("-" * len(header))
    for label, threshold in thresholds.items():
        results = calculate_ranking_metrics(
            prediction_matrix=prediction_matrix,
            pos_label_idxs=target_indices,
            metrics=metrics,
            pos_label_relevance=target_relevance,
            binary_relevance_threshold=threshold,
        )
        row = f"{label:<36}" + "".join(f"{results[m]:>10.4f}" for m in metrics)
        print(row)
    print(
        "\nnDCG@5 is identical across thresholds — it always consumes the full graded list."
        "\nMAP / MRR / recall@5 shift because the threshold redefines the binary positive set:"
        "\n  - 1e-9: the nice-to-have (grade 1) at rank 1 counts as a positive, so MRR=1.0"
        "\n          and the model 'looks great' on binary metrics despite ranking a"
        "\n          low-value item first."
        "\n  - 2.0:  the nice-to-have is dropped from the positive set; MRR drops to the"
        "\n          rank of the first surviving positive, MAP drops, and recall@5 is"
        "\n          computed against a smaller denominator (3 instead of 4)."
        "\nLesson: on a graded dataset, binary metrics answer 'how does the model rank"
        "\nthe items I declared relevant', and *what counts as relevant* is exactly the"
        "\nthreshold. Pick it deliberately."
    )


if __name__ == "__main__":
    print("🚀 Custom Graded-Relevance Task Example")
    print("=" * 50)

    model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")
    tasks = [GradedJob2SkillTask(languages=["en"], split="test")]

    # Run with the task's default_metrics, which mixes nDCG (graded) with
    # MAP/MRR/recall (binary). The same dataset feeds both — graded metrics
    # consult target_relevance, binary metrics see only the thresholded set.
    results = workrb.evaluate(
        model,
        tasks,
        output_folder="results/graded_task_demo",
        description="Custom ranking task with graded relevance.",
        force_restart=True,
    )
    print(results)

    # Show why binary_relevance_threshold matters: same data, two thresholds.
    _demo_threshold_effect()
