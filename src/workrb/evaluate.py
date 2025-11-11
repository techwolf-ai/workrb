"""
Main WorkRB class for evaluating models on benchmark tasks.

Supports both simple single-model evaluation and advanced features like
checkpointing, resuming, and efficient multi-model evaluation.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from workrb.config import BenchmarkConfig
from workrb.logging import setup_logger
from workrb.metrics.reporting import format_results
from workrb.models.base import ModelInterface
from workrb.registry import TaskRegistry
from workrb.results import (
    BenchmarkMetadata,
    BenchmarkResults,
    MetricsResult,
    TaskResultMetadata,
    TaskResults,
)
from workrb.tasks.abstract.base import Language, Task

logger = logging.getLogger(__name__)
setup_logger(__name__, verbose=False)


def evaluate(
    model: ModelInterface,
    tasks: Sequence[Task],
    output_folder: str,
    metrics: dict[str, list[str]] | None = None,
    description: str = "",
    force_restart: bool = False,
) -> BenchmarkResults:
    """
    Run benchmark evaluation for a single model.

    Args:
        model: Model to evaluate (must implement ModelInterface)
        tasks: List of Task instances to run evaluation on
        output_folder: Optional folder to save results
        metrics: Optional dict mapping task names to custom metrics lists
        description: Description for the benchmark run
        force_restart: If True, ignore checkpoints and restart from beginning
        unsupported_lang_mode: If the task does not support a language,
            "error" will raise an error, stopping execution.
            "skip" will skip the language in the evaluation, and final results.

    Returns
    -------
        Dictionary containing all evaluation results
    """
    # Load config for output folder
    config = BenchmarkConfig.from_workrb(
        model=model,
        tasks=tasks,
        output_folder=output_folder,
        custom_metrics=metrics,
        description=description,
    )
    results, pending_work = _init_checkpointing(
        tasks=tasks,
        config=config,
        model=model,
        force_restart=force_restart,
    )

    if len(pending_work) == 0:
        logger.info("All work already completed!")
        return results

    logger.info(f"Running WorkRB for model: {model.name}")
    logger.info(get_tasks_overview(tasks))
    logger.info(f"{'=' * 60}")
    logger.info(f"Pending work: {len(pending_work)} / {_get_total_evaluations(tasks)} evaluations")

    # Group pending work by task for better organization
    work_by_task = {}
    for task, language in pending_work:
        if task.name not in work_by_task:
            work_by_task[task.name] = {"task": task, "languages": []}
        work_by_task[task.name]["languages"].append(language)

    # Run pending work
    start_time_benchmark = time.time()
    results = _run_pending_work(
        tasks=tasks,
        config=config,
        work_by_task=work_by_task,
        results=results,
        model=model,
        metrics=metrics,
    )
    if results.metadata.resumed_from_checkpoint:
        logger.info("✓ Successfully resuming from checkpoint")

    # Update metadata
    results.metadata.total_evaluation_time = time.time() - start_time_benchmark
    results.metadata.resumed_from_checkpoint = len(pending_work) < sum(
        len(task.languages) for task in tasks
    )

    # Save config and results
    config.save_final_result_artifacts(results)

    logger.info(f"{'=' * 60}")
    logger.info("✓ WorkRB COMPLETE")
    logger.info(f"Total time: {results.metadata.total_evaluation_time:.2f}s")
    logger.info(
        format_results(
            results,
            display_per_task=False,
            display_per_task_group=False,
            display_per_language=False,
            display_overall=True,
        )
    )
    logger.info(f"{'=' * 60}")
    logger.info("✅ Benchmark completed!")

    return results


def evaluate_multiple_models(
    models: Sequence[ModelInterface],
    tasks: Sequence[Task],
    output_folder_template: str = "results/{model_name}",
    **run_kwargs,
) -> dict[str, BenchmarkResults]:
    """
    Easily run multiple models with wrapper. State is stored in model-specific output folders.

    Args:
        models: List of models to evaluate
        output_folder_template: Template for output folders (use {model_name} placeholder)
        **run_kwargs: Additional arguments passed to run()

    Returns
    -------
        Dictionary mapping model names to their results
    """
    # Input checks
    model_names = [model.name for model in models]
    model_name_counts = Counter(model_names)
    assert sum(model_name_counts.values()) == len(model_names), (
        f"All models must have unique names, "
        f"duplicates found: {', '.join(n for n, c in model_name_counts.items() if c > 1)}"
    )
    assert "{model_name}" in output_folder_template, (
        "Output folder template must contain {model_name}, for example: results/{model_name}"
    )

    all_results = {}
    for model in models:
        # Set up output folder for this model
        model_output_folder = output_folder_template.format(model_name=model.name)
        run_kwargs["output_folder"] = model_output_folder

        logger.info(f"{'=' * 60}")
        logger.info(f"Running model: {model.name}")
        logger.info(f"Output folder: {model_output_folder}")
        logger.info(f"{'=' * 60}")

        try:
            results = evaluate(tasks, model, **run_kwargs)
            all_results[model.name] = results
        except Exception as e:
            logger.error(f"Error running model {model.name}: {e}")
            raise e

    return all_results


def get_tasks_overview(tasks: Sequence[Task]) -> str:
    """Get information about configured tasks as a formatted string summary."""
    # Calculate summary statistics
    num_tasks = len(tasks)
    task_groups = {task.task_group for task in tasks if task.task_group}
    num_task_groups = len(task_groups)
    languages = _get_all_languages(tasks)
    num_languages = len(languages)

    # Build the summary string
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append(
        f"Tasks: {num_tasks} | Task Groups: {num_task_groups} | Languages: {num_languages}"
    )
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'Task Name':<40} {'Group':<20} {'Languages':<20}")
    lines.append("-" * 80)

    # Add each task's information
    for task in tasks:
        task_name = task.name
        group = task.task_group.value
        task_languages = ", ".join([lang.value for lang in task.languages])

        lines.append(f"{task_name:<40} {group:<20} {task_languages:<20}")

        # Add size one-liner for each language
        for lang in task.languages:
            size_info = task.get_size_oneliner(lang)
            if size_info:
                lines.append(f"  └─ {lang}: {size_info}")

    lines.append("-" * 80)

    return "\n".join(lines)


def load_results(results_path: str = "./results.json") -> BenchmarkResults:
    """
    Load results from specified folder.

    Useful for external usage of the results, when only the folder is available.
    """
    with open(results_path) as f:
        data = json.load(f)
    return BenchmarkResults.model_validate(data)


def list_available_tasks() -> dict[str, str]:
    """List all available task classes that can be used in configs."""
    return TaskRegistry.list_available()


def _get_all_languages(tasks: Sequence[Task]) -> list[str]:
    """Get all unique languages across tasks."""
    languages = set()
    for task in tasks:
        languages.update(task.languages)
    return sorted([str(lang) for lang in languages])


def _get_total_evaluations(tasks: Sequence[Task]) -> int:
    """Get the total number of evaluations."""
    return sum(len(task.languages) for task in tasks)


def _validate_tasks(tasks: Sequence[Task]):
    """Validate that all tasks are properly configured."""
    if not tasks:
        raise ValueError("At least one task must be provided")

    for task in tasks:
        if not isinstance(task, Task):
            raise TypeError(f"All tasks must inherit from Task, got {type(task)}")


def _init_checkpointing(
    tasks: Sequence[Task],
    config: BenchmarkConfig,
    model: ModelInterface,
    force_restart: bool,
) -> tuple[BenchmarkResults, list[tuple[Task, str]]]:
    """Initialize checkpointing.

    Returns
    -------
        Tuple containing the results and the pending work.
    """
    existing_results = None
    if not force_restart and config and config.has_checkpoint():
        logger.info("Resuming from existing checkpoint...")
        assert model.name == config.model_name, (
            f"Model name mismatch. Trying to run model '{model.name}' "
            f"with checkpoint from model '{config.model_name}' "
            f"in checkpoint '{config.get_checkpoint_path()}'."
        )

        existing_results = config.restore_results_from_checkpoint()
        config.validate_results_contained_in_tasks(results=existing_results, tasks=tasks)

        if existing_results:
            logger.info(
                f"Restored {len(existing_results.task_results)} completed tasks from checkpoint"
            )

    pending_work = config.get_pending_work(
        results=existing_results,
        tasks=tasks,
    )

    results = existing_results
    if results is None:
        # Collect key metrics by task group
        key_metrics_by_task_group = {}
        for task in tasks:
            key_metrics_by_task_group[task.task_group.value] = task.default_metrics

        # Create new BenchmarkResults
        results = BenchmarkResults(
            task_results={},
            metadata=BenchmarkMetadata(
                model_name=model.name,
                total_evaluation_time=0.0,
                timestamp=time.time(),
                num_tasks=len(tasks),
                languages=_get_all_languages(tasks),
                resumed_from_checkpoint=False,
            ),
            key_metrics_by_task_group=key_metrics_by_task_group,
        )
    return results, pending_work


def _run_pending_work(
    tasks: Sequence[Task],
    config: BenchmarkConfig,
    work_by_task: dict[str, dict[str, Any]],
    results: BenchmarkResults,
    model: ModelInterface,
    metrics: dict[str, list[str]] | None,
) -> BenchmarkResults:
    """Run pending evaluations.

    Args:
        work_by_task: Dictionary of task names to their pending languages.
        results: BenchmarkResults object to store results.
        model: ModelInterface object to evaluate.
        metrics: Dictionary of task names to their custom metrics.
    """
    # Run pending evaluations
    run_idx = results.get_num_evaluation_results()  # Already completed evaluations
    for work_info in work_by_task.values():
        task: Task = work_info["task"]
        pending_languages: list[str] = work_info["languages"]

        logger.info(f"{'=' * 60}")
        logger.info(f"Evaluating task: {task.name}")
        logger.info(f"Completed {run_idx} / {_get_total_evaluations(tasks)} evaluations. ")
        logger.info(f"Pending languages for this task: {len(pending_languages)}")

        # Initialize task results if not exists
        if task.name not in results.task_results:
            results.task_results[task.name] = TaskResults(
                metadata=TaskResultMetadata(
                    task_group=task.task_group.value,
                    task_type=task.task_type.value,
                    label_type=task.label_type.value,
                    description=task.description,
                    split=task.split.value,
                ),
                language_results={},
            )

        # Evaluate pending languages
        for language in pending_languages:
            logger.info(
                f"* Running language: {language} ({task.get_size_oneliner(Language(language))})"
            )

            # Get metrics for this task
            task_metrics = None
            if metrics and task.name in metrics:
                task_metrics = metrics[task.name]

            try:
                start_time_eval = time.time()
                lang_results: dict[str, float] = task.evaluate(
                    model=model, metrics=task_metrics, language=Language(language)
                )
                evaluation_time = time.time() - start_time_eval

                # Store results
                results.task_results[task.name].language_results[language] = MetricsResult(
                    evaluation_time=evaluation_time,
                    metrics_dict=lang_results,
                )

                # Save incremental results to checkpoint
                if config:
                    config.save_results_checkpoint(results)

                # Show key metrics
                key_metric = task.default_metrics[0]
                logger.info(f"\t{key_metric}: {lang_results[key_metric]:.3f}")
                run_idx += 1
            except Exception as e:
                logger.error(f"Error: {e}")
                raise e

    logger.info(f"Completed {run_idx} / {_get_total_evaluations(tasks)} evaluations. ")
    return results
