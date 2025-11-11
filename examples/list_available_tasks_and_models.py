"""
Generate tables for tasks and models in the WorkRB.

Use --no-load-tasks and --no-load-models flags to skip loading (faster).
"""

import argparse
from typing import Any, cast

import pandas as pd

from workrb.registry import ModelRegistry, TaskRegistry
from workrb.tasks.abstract.base import Language


def get_task_properties(task_class: type, registry_name: str, load_full: bool) -> dict[str, Any]:
    """Extract task properties, optionally loading the task instance."""
    if not load_full:
        return {
            "Registry Name": registry_name,
            "Class Name": task_class.__name__,
            "Module": task_class.__module__,
        }

    # Instantiate task to get dynamic properties
    task = task_class(languages=["en"], split="test")
    num_langs = len(task.supported_languages)

    # Get size for English if supported, otherwise first language
    size_lang = (
        Language.EN if Language.EN in task.supported_languages else task.supported_languages[0]
    )
    size_info = task.get_size_oneliner(size_lang)

    return {
        "Registry Name": registry_name,
        "Task Name": task.name,
        "Task Type": task.task_type.value,
        "Label Type": task.label_type.value,
        "Dataset Size": size_info or "N/A",
        "Languages": num_langs,
        "Class Name": task_class.__name__,
        "Module": task_class.__module__,
    }


def get_model_properties(model_class: type, registry_name: str, load_full: bool) -> dict[str, Any]:
    """Extract model properties."""
    props = {
        "Registry Name": registry_name,
        "Class Name": model_class.__name__,
        "Module": model_class.__module__,
    }

    if load_full:
        model = ModelRegistry.create(registry_name)
        props["Model Name"] = model.name
        props["Fixed Classifier"] = "✅" if model.classification_label_space is not None else "❌"
        props["Description"] = model.description

    return props


def generate_tasks_table(
    show_columns: list[str] | None, task_type_filter: str | None, load_full: bool
) -> pd.DataFrame:
    """Generate table of all registered tasks."""
    tasks_data = [
        get_task_properties(TaskRegistry.get(name), name, load_full)
        for name in TaskRegistry.list_available()
    ]

    df = pd.DataFrame(tasks_data)

    # Filter by task type if loading full properties
    if task_type_filter and load_full and "Task Type" in df.columns:
        df = cast(pd.DataFrame, df[df["Task Type"] == task_type_filter])

    # Select columns if specified
    if show_columns and len(df) > 0:
        cols = [c for c in show_columns if c in df.columns]
        return cast(pd.DataFrame, df[cols] if len(cols) > 1 else df[[cols[0]]])

    return df


def generate_models_table(show_columns: list[str] | None, load_full: bool) -> pd.DataFrame:
    """Generate table of all registered models."""
    models_data = [
        get_model_properties(ModelRegistry.get(name), name, load_full)
        for name in ModelRegistry.list_available()
    ]

    df = pd.DataFrame(models_data)

    # Select columns if specified
    if show_columns and len(df) > 0:
        cols = [c for c in show_columns if c in df.columns]
        return cast(pd.DataFrame, df[cols] if len(cols) > 1 else df[[cols[0]]])

    return df


def format_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table."""
    if df.empty:
        return "No data available"

    lines = []
    headers = df.columns.tolist()

    # Header and separator
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Data rows
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables for WorkRB tasks and models")
    parser.add_argument("--no-load-tasks", action="store_true", help="Skip loading tasks (faster)")
    parser.add_argument(
        "--no-load-models", action="store_true", help="Skip loading models (faster)"
    )
    args = parser.parse_args()

    load_tasks = not args.no_load_tasks
    load_models = not args.no_load_models

    print("=" * 80)
    print("WorkRB TASKS AND MODELS")
    print("=" * 80)
    if not load_tasks or not load_models:
        print(f"Fast mode - loading tasks: {load_tasks}, models: {load_models}")
    print()

    # Task columns
    task_cols = (
        ["Task Name", "Label Type", "Dataset Size", "Languages"]
        if load_tasks
        else ["Registry Name", "Label Type"]
    )

    # Ranking tasks
    print("RANKING TASKS")
    print("---" * 80)
    print(
        format_markdown(
            generate_tasks_table(task_cols, "ranking" if load_tasks else None, load_tasks)
        )
    )
    print("\n")

    # Classification tasks
    print("CLASSIFICATION TASKS")
    print("-" * 80)
    print(
        format_markdown(
            generate_tasks_table(task_cols, "classification" if load_tasks else None, load_tasks)
        )
    )
    print("\n")

    # Models
    model_cols = (
        ["Class Name", "Description", "Fixed Classifier"]
        if load_models
        else ["Class Name", "Description"]
    )
    print("MODELS")
    print("-" * 80)
    print(format_markdown(generate_models_table(model_cols, load_models)))
