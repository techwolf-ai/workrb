"""
Run all discoverable ranking tasks with selected models.

Auto-discovers every registered ranking task via the TaskRegistry and
evaluates them with the chosen models. By default all registered models
are used; pass --models to select a subset.

Usage:
    python examples/run_all_ranking_tasks.py                          # all models (with confirmation)
    python examples/run_all_ranking_tasks.py --models BM25Model,TfIdfModel
    python examples/run_all_ranking_tasks.py --list                   # print tasks and models
    python examples/run_all_ranking_tasks.py -y                       # skip confirmation prompt
"""

import argparse
import sys

import workrb
from workrb.registry import ModelRegistry, TaskRegistry
from workrb.tasks.abstract.ranking_base import RankingTask


def discover_ranking_tasks() -> dict[str, str]:
    """Return registry entries that are ranking tasks."""
    return {
        name: path
        for name, path in TaskRegistry.list_available().items()
        if issubclass(TaskRegistry.get(name), RankingTask)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run all discoverable ranking tasks with selected models."
    )
    parser.add_argument(
        "--list", action="store_true", help="Print discovered tasks and models, then exit"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model registry names (default: all discovered models)",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    ranking_tasks = discover_ranking_tasks()
    available_models = ModelRegistry.list_available()

    if args.list:
        print("Ranking tasks:")
        for name in sorted(ranking_tasks):
            print(f"  {name}")
        print("\nModels:")
        for name in sorted(available_models):
            print(f"  {name}")
        return

    # Resolve model names
    if args.models:
        model_names = [n.strip() for n in args.models.split(",")]
        unknown = [n for n in model_names if n not in available_models]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(available_models))}")
            sys.exit(1)
    else:
        model_names = list(available_models)

    # Show summary and ask for confirmation
    print(f"Ranking tasks ({len(ranking_tasks)}):")
    for name in sorted(ranking_tasks):
        print(f"  {name}")
    print(f"\nModels ({len(model_names)}):")
    for name in model_names:
        print(f"  {name}")
    print()

    if not args.yes:
        answer = input("Continue? [Y/n] ").strip().lower()
        if answer not in ("", "y"):
            print("Aborted.")
            return

    print("Creating tasks...")
    tasks = [TaskRegistry.create(name, split=args.split, languages=None) for name in ranking_tasks]

    print("Creating models...")
    models = [ModelRegistry.create(name) for name in model_names]

    all_results = workrb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template="../results/run_all_ranking/{model_name}",
        description="All ranking tasks",
        force_restart=True,
    )

    for model_name, results in all_results.items():
        print(f"\nResults for {model_name}:")
        print(results)


if __name__ == "__main__":
    main()
