"""
Run the benchmark with flat averaging on a selected set of languages.

Aggregation mode: SKIP_LANGUAGE_AGGREGATION
    All datasets contribute equally to the per-task score as a flat
    average, with no language-based grouping or filtering. This means
    cross-lingual and multilingual datasets are included alongside
    monolingual ones. The final results do not include per-language
    averages, since no language grouping criterion is defined and
    there is no unambiguous way to assign cross-lingual or
    multilingual datasets to a single language bucket.

Task-level language filtering:
    The `langs` list restricts which datasets each task loads during
    initialization. Only languages in this list are considered.

Execution mode: ALL
    Explicitly set here, but has no practical effect under
    SKIP_LANGUAGE_AGGREGATION since no datasets are ever filtered
    out by the aggregation mode.
"""

import workrb
from workrb.types import ExecutionMode, Language, LanguageAggregationMode

if __name__ == "__main__":
    # Models
    models = [
        # Lexical baselines
        workrb.models.RandomRankingModel(),
        workrb.models.BM25Model(lowercase=True),
        # DL model
        workrb.models.JobBERTModel(),
    ]

    # Languages (as strings via .value)
    langs = [
        Language.DA.value,
        Language.DE.value,
        Language.EN.value,
        Language.ES.value,
        Language.FR.value,
        Language.HU.value,
        Language.IT.value,
        Language.LT.value,
        Language.NL.value,
        Language.PL.value,
        Language.PT.value,
        Language.SL.value,
        Language.SV.value,
    ]
    split = "test"

    # Tasks
    tasks = [
        # Tasks with monolingual datasets
        workrb.tasks.ESCOJob2SkillRanking(split=split, languages=langs),
        workrb.tasks.ESCOSkill2JobRanking(split=split, languages=langs),
        # Tasks with monolingual, cross-lingual, and multilingual datasets
        workrb.tasks.ProjectCandidateRanking(split=split, languages=langs),
        workrb.tasks.SearchQueryCandidateRanking(split=split, languages=langs),
        # TODO: add MELO and MELS tasks when PR #37 is merged
    ]

    # Evaluate
    # NOTE: execution_mode=ALL has no effect when using SKIP_LANGUAGE_AGGREGATION,
    # because no datasets are ever filtered out regardless of execution mode.
    all_results = workrb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template="../results/flat_average/{model_name}",
        description="Flat average benchmark",
        force_restart=True,
        language_aggregation_mode=LanguageAggregationMode.SKIP_LANGUAGE_AGGREGATION,
        execution_mode=ExecutionMode.ALL,
    )

    # Display results
    for model_name, results in all_results.items():
        print(f"\nResults for {model_name}:")
        print(results)
