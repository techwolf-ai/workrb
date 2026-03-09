"""
Run paper experiments: two model groups with different language aggregation.

- Group 1 (Multilingual): BM25, Random, JobBERT-v3, BiEncoder-Qwen
  → All tasks with all languages, SKIP_LANGUAGE_AGGREGATION
- Group 2 (EN-only): ConTeXTMatch, CurriculumMatch
  → All tasks with languages=["en"], MONOLINGUAL_ONLY aggregation

Note: MELSRanking has no EN-query datasets — it will have 0 dataset_ids for Group 2
and be skipped gracefully.

Usage:
    python examples/run_paper_results.py                # run all (with confirmation)
    python examples/run_paper_results.py --list         # print tasks and models
    python examples/run_paper_results.py -y             # skip confirmation prompt
"""

import argparse

import workrb
from workrb.models import (
    BiEncoderModel,
    BM25Model,
    ConTeXTMatchModel,
    CurriculumMatchModel,
    JobBERTModel,
    RandomRankingModel,
)
from workrb.registry import TaskRegistry
from workrb.types import ExecutionMode, LanguageAggregationMode

# All ranking tasks, grouped by language structure for documentation.
ALL_TASKS = [
    "ESCOJob2SkillRanking",
    "ESCOSkill2JobRanking",
    "HouseSkillExtractRanking",
    "ESCOSkillNormRanking",
    "JobTitleSimilarityRanking",
    "JobBERTJobNormRanking",
    "MELORanking",
    "MELSRanking",
    "ProjectCandidateRanking",
    "SearchQueryCandidateRanking",
    "SkillSkapeExtractRanking",
    "TechSkillExtractRanking",
    "SkillMatch1kSkillSimilarityRanking",
]


def create_tasks(split: str, languages: list[str] | None = None) -> list:
    """Create all ranking tasks with the given language setting."""
    tasks = []
    for name in ALL_TASKS:
        tasks.append(TaskRegistry.create(name, split=split, languages=languages))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments with two model groups.")
    parser.add_argument("--list", action="store_true", help="Print tasks and models, then exit")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if args.list:
        print(f"Multilingual tasks ({len(ALL_TASKS)}):")
        for name in ALL_TASKS:
            print(f"  {name}")
        print(f"\nTotal: {len(ALL_TASKS)} tasks")
        print("\nGroup 1 — Multilingual (SKIP_LANGUAGE_AGGREGATION, all languages):")
        print("  BM25Model, RandomRankingModel, JobBERT-v3, BiEncoder-Qwen3-Embedding-0.6B")
        print("\nGroup 2 — EN-only (MONOLINGUAL_ONLY, languages=['en']):")
        print("  ConTeXTMatchModel, CurriculumMatchModel")
        return

    # Summary
    print(f"Tasks: {len(ALL_TASKS)} ranking tasks")
    print(f"Split: {args.split}")
    print("\nGroup 1 — Multilingual: BM25, Random, JobBERT-v3, BiEncoder-Qwen (all languages)")
    print("Group 2 — EN-only:      ConTeXTMatch, CurriculumMatch (languages=['en'])")
    print()

    if not args.yes:
        answer = input("Continue? [Y/n] ").strip().lower()
        if answer not in ("", "y"):
            print("Aborted.")
            return

    # --- Group 1: Multilingual models (all languages) ---
    print("Creating tasks (all languages)...")
    tasks_all_langs = create_tasks(args.split, languages=None)

    print("\nCreating multilingual models...")
    multilingual_models = [
        BM25Model(),
        RandomRankingModel(),
        JobBERTModel(model_name="TechWolf/JobBERT-v3"),
        BiEncoderModel(model_name="Qwen/Qwen3-Embedding-0.6B"),
    ]

    print("Running Group 1 (multilingual)...")
    results_multilingual = workrb.evaluate_multiple_models(
        models=multilingual_models,
        tasks=tasks_all_langs,
        output_folder_template="../results/run_paper_results/multilingual/{model_name}",
        description="Paper results - multilingual models",
        force_restart=False,
        language_aggregation_mode=LanguageAggregationMode.SKIP_LANGUAGE_AGGREGATION,
        execution_mode=ExecutionMode.LAZY,
    )

    # --- Group 2: Monolingual EN-only models ---
    print("\nCreating tasks (EN only)...")
    tasks_en_only = create_tasks(args.split, languages=["en"])

    print("Creating monolingual EN-only models...")
    monolingual_models = [
        ConTeXTMatchModel(),
        CurriculumMatchModel(),
    ]

    print("Running Group 2 (monolingual EN-only)...")
    results_monolingual = workrb.evaluate_multiple_models(
        models=monolingual_models,
        tasks=tasks_en_only,
        output_folder_template="../results/run_paper_results/monolingual_en/{model_name}",
        description="Paper results - monolingual EN-only models",
        force_restart=False,
        language_aggregation_mode=LanguageAggregationMode.MONOLINGUAL_ONLY,  # No cross-lingual datasets
        execution_mode=ExecutionMode.LAZY,
    )

    # --- Print all results ---
    all_results = {**results_multilingual, **results_monolingual}
    for model_name, results in all_results.items():
        print(f"\nResults for {model_name}:")
        print(results)


if __name__ == "__main__":
    main()
