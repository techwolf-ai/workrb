"""
Run the benchmark with language-weighted aggregation on all available languages.

Aggregation mode: MONOLINGUAL_ONLY
    Within each task, datasets are grouped by language and averaged per
    group, then the per-language means are averaged to produce the
    per-task score. This gives equal weight to each language regardless
    of how many datasets it has. Datasets where input and output
    languages differ (cross-lingual) are filtered out of aggregation.

Task-level language filtering: None
    Setting langs=None means each task loads all languages it supports,
    with no filtering at the task level.

Execution mode: LAZY (default)
    Datasets that would be filtered out by the aggregation mode are
    not evaluated at all, saving compute.
"""

import workrb
from workrb.types import ExecutionMode, LanguageAggregationMode

if __name__ == "__main__":
    # Models
    models = [
        # Lexical baselines
        workrb.models.RandomRankingModel(),
        workrb.models.BM25Model(lowercase=True),
        # DL model
        workrb.models.JobBERTModel(),
    ]

    # No language filtering: each task loads all languages it supports
    langs = None
    split = "test"

    # Tasks
    tasks = [
        # Tasks with monolingual datasets
        workrb.tasks.ESCOJob2SkillRanking(split=split, languages=langs),
        workrb.tasks.ESCOSkill2JobRanking(split=split, languages=langs),
        # Tasks with monolingual, cross-lingual, and multilingual datasets
        workrb.tasks.ProjectCandidateRanking(split=split, languages=langs),
        workrb.tasks.SearchQueryCandidateRanking(split=split, languages=langs),
        workrb.tasks.MELORanking(split=split, languages=langs),
        workrb.tasks.MELSRanking(split=split, languages=langs),
    ]

    # Evaluate
    all_results = workrb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template="../results/language_weighted_all_langs/{model_name}",
        description="Language-weighted benchmark (all languages)",
        force_restart=True,
        language_aggregation_mode=LanguageAggregationMode.MONOLINGUAL_ONLY,
        execution_mode=ExecutionMode.LAZY,
    )

    # Display results
    for model_name, results in all_results.items():
        print(f"\nResults for {model_name}:")
        print(results)
