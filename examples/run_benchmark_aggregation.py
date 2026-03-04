"""
Run benchmarks with different language aggregation strategies.

WorkRB supports two main aggregation modes that control how per-dataset
results are rolled up into per-task scores:

1. MONOLINGUAL_ONLY (default, "language-weighted")
   - Groups datasets by language, averages within each group, then averages
     across languages. Each language gets equal weight.
   - Cross-lingual datasets (input lang != output lang) are filtered out.
   - Use execution_mode=LAZY (default) to skip evaluating filtered datasets.

2. SKIP_LANGUAGE_AGGREGATION ("flat average")
   - All datasets (mono-, cross-, multilingual) are flat-averaged per task.
   - No per-language breakdown to obtain final results.
   - execution_mode has no effect (nothing is ever filtered).

Language filtering (the `languages` parameter on tasks) is orthogonal:
  - Pass a list of language codes to restrict which datasets are loaded.
  - Pass None to load all languages the task supports.

Usage:
    python examples/run_benchmark_aggregation.py                    # language-weighted (default)
    python examples/run_benchmark_aggregation.py --flat             # flat average
    python examples/run_benchmark_aggregation.py --all-languages    # use all available languages
"""

import argparse

import workrb
from workrb.types import ExecutionMode, Language, LanguageAggregationMode

# --- Configuration -------------------------------------------------------

# Subset of languages (used unless --all-languages is passed)
SELECTED_LANGUAGES = [
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


def main():
    parser = argparse.ArgumentParser(
        description="Run WorkRB benchmark with different aggregation modes."
    )
    parser.add_argument(
        "--flat", action="store_true", help="Use flat averaging (SKIP_LANGUAGE_AGGREGATION)"
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Evaluate all languages (default: selected subset)",
    )
    args = parser.parse_args()

    # Choose aggregation strategy
    if args.flat:
        aggregation_mode = LanguageAggregationMode.SKIP_LANGUAGE_AGGREGATION
        execution_mode = ExecutionMode.ALL
        run_name = "flat_average"
    else:
        aggregation_mode = LanguageAggregationMode.MONOLINGUAL_ONLY
        execution_mode = ExecutionMode.LAZY
        run_name = "language_weighted"

    # Choose languages
    langs = None if args.all_languages else SELECTED_LANGUAGES
    if args.all_languages:
        run_name += "_all_langs"

    # Models
    models = [
        workrb.models.RandomRankingModel(),
        workrb.models.BM25Model(lowercase=True),
        workrb.models.JobBERTModel(),
    ]

    # Tasks (mix of monolingual-only and cross-lingual tasks)
    split = "test"
    tasks = [
        workrb.tasks.ESCOJob2SkillRanking(split=split, languages=langs),
        workrb.tasks.ESCOSkill2JobRanking(split=split, languages=langs),
        workrb.tasks.ProjectCandidateRanking(split=split, languages=langs),
        workrb.tasks.SearchQueryCandidateRanking(split=split, languages=langs),
        workrb.tasks.MELORanking(split=split, languages=langs),
        workrb.tasks.MELSRanking(split=split, languages=langs),
    ]

    # Evaluate
    all_results = workrb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template=f"../results/{run_name}/{{model_name}}",
        description=f"{run_name} benchmark",
        force_restart=True,
        language_aggregation_mode=aggregation_mode,
        execution_mode=execution_mode,
    )

    # Display results
    for model_name, results in all_results.items():
        print(f"\nResults for {model_name}:")
        print(results)


if __name__ == "__main__":
    main()
