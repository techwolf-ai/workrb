"""
Main WorkRB class for evaluating models on benchmark tasks.

Supports both simple single-model evaluation and advanced features like
checkpointing, resuming, and efficient multi-model evaluation.
"""

import logging
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from workrb.config import BenchmarkConfig, _get_workrb_version
from workrb.logging import setup_logger
from workrb.metrics.reporting import format_results
from workrb.models.base import ModelInterface
from workrb.rankings import (
    RankingsArtifactInvalid,
    RankingsArtifactMissing,
    load_rankings_artifact,
    materialize_prediction_matrix,
    rankings_filename,
    validate_header,
)
from workrb.results import (
    BenchmarkMetadata,
    BenchmarkResults,
    MetricsResult,
    TaskResultMetadata,
    TaskResults,
)
from workrb.tasks.abstract.base import Task
from workrb.tasks.abstract.ranking_base import RankingTask
from workrb.types import ExecutionMode, LanguageAggregationMode, get_language_grouping_key

logger = logging.getLogger(__name__)
setup_logger(__name__, verbose=False)


def evaluate(
    model: ModelInterface,
    tasks: Sequence[Task],
    output_folder: str,
    metrics: dict[str, list[str]] | None = None,
    description: str = "",
    force_restart: bool = False,
    save_rankings: bool = False,
    language_aggregation_mode: LanguageAggregationMode = LanguageAggregationMode.MONOLINGUAL_ONLY,
    execution_mode: ExecutionMode = ExecutionMode.LAZY,
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
        save_rankings: If True, save per-target ranking score arrays for each
            ranking task dataset under
            ``<output_folder>/rankings/<model_name>/`` as JSON artifacts.
            Has no effect for non-ranking tasks. Defaults to False.
        language_aggregation_mode: How per-language results should be grouped
            when calling ``get_summary_metrics()`` on the returned results.
            When ``execution_mode`` is ``LAZY``, datasets that are
            incompatible with the chosen mode are also skipped before
            evaluation to avoid unnecessary compute.
            Defaults to ``MONOLINGUAL_ONLY``.
        execution_mode: Controls whether incompatible datasets are skipped
            (``LAZY``, default) or evaluated regardless (``ALL``).

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
        language_aggregation_mode=language_aggregation_mode,
    )

    # Determine which datasets are in scope for this run
    dataset_ids_to_evaluate = _get_dataset_ids_to_evaluate(
        tasks, language_aggregation_mode, execution_mode
    )

    pending_work = _filter_pending_work(pending_work, dataset_ids_to_evaluate)
    total_evaluations = sum(len(dids) for dids in dataset_ids_to_evaluate.values())

    if len(pending_work) == 0:
        logger.info("All work already completed!")
        return results

    logger.info(f"Running WorkRB for model: {model.name}")
    logger.info(get_tasks_overview(tasks, dataset_ids_to_evaluate=dataset_ids_to_evaluate))
    logger.info(f"{'=' * 60}")
    logger.info(f"Pending work: {len(pending_work)} / {total_evaluations} evaluations")

    # Group pending work by task for better organization
    work_by_task = {}
    for task, dataset_id in pending_work:
        if task.name not in work_by_task:
            work_by_task[task.name] = {"task": task, "dataset_ids": []}
        work_by_task[task.name]["dataset_ids"].append(dataset_id)

    # Run pending work
    start_time_benchmark = time.time()
    results = _run_pending_work(
        config=config,
        work_by_task=work_by_task,
        results=results,
        model=model,
        metrics=metrics,
        save_rankings=save_rankings,
        total_evaluations=total_evaluations,
    )
    if results.metadata.resumed_from_checkpoint:
        logger.info("✓ Successfully resuming from checkpoint")

    # Update metadata
    results.metadata.total_evaluation_time = time.time() - start_time_benchmark
    results.metadata.resumed_from_checkpoint = len(pending_work) < total_evaluations

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
            language_aggregation_mode=language_aggregation_mode,
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
    duplicate_model_names = [n for n, c in model_name_counts.items() if c > 1]
    assert not duplicate_model_names, (
        f"All models must have unique names, duplicates found: {', '.join(duplicate_model_names)}"
    )
    assert "{model_name}" in output_folder_template, (
        "Output folder template must contain {model_name}, for example: results/{model_name}"
    )
    assert len(models) > 0, "At least one model must be provided to evaluate multiple models."

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
            results = evaluate(model=model, tasks=tasks, **run_kwargs)
            all_results[model.name] = results
        except Exception as e:
            logger.error(f"Error running model {model.name}: {e}")
            raise e

    return all_results


def evaluate_rankings(
    rankings_dir: str | Path,
    tasks: Sequence[Task],
    output_folder: str,
    metrics: dict[str, list[str]] | None = None,
    description: str = "",
    language_aggregation_mode: LanguageAggregationMode = LanguageAggregationMode.MONOLINGUAL_ONLY,
    execution_mode: ExecutionMode = ExecutionMode.LAZY,
) -> BenchmarkResults:
    """Compute benchmark metrics by replaying saved ranking artifacts.

    No model is required. ``rankings_dir`` is expected to be the
    ``<output_folder>/rankings/<model_name>/`` directory produced by a
    previous ``evaluate(..., save_rankings=True)`` run.

    Always rescores from scratch: any existing ``results.json`` or
    ``checkpoint.json`` under ``output_folder`` is overwritten without
    consulting it.

    Args:
        rankings_dir: Directory containing per-``(task, dataset_id)`` JSON
            artifacts produced by :meth:`BenchmarkConfig.save_rankings_artifact`.
        tasks: Tasks to score. Must all be ``RankingTask`` instances; passing
            any other task type raises ``ValueError``.
        output_folder: Where to write ``results.json``, ``config.yaml``, and
            ``checkpoint.json`` (same layout as :func:`evaluate`).
        metrics: Optional dict mapping task names to custom metrics lists.
        description: Description for the benchmark run.
        language_aggregation_mode: How per-language results should be grouped
            in the returned ``BenchmarkResults``.
        execution_mode: ``LAZY`` skips datasets incompatible with the chosen
            aggregation mode, ``ALL`` scores everything for which an artifact
            exists.

    Returns
    -------
        ``BenchmarkResults`` populated from the artifacts.

    Raises
    ------
        ValueError: ``tasks`` contains any non-``RankingTask`` entry.
        FileNotFoundError: ``rankings_dir`` does not exist, or exists but
            contains no ``*.json`` artifacts.
        RankingsArtifactMissing: A specific ``(task, dataset_id)`` artifact
            expected by the scoring loop is not on disk. Subclasses
            ``FileNotFoundError``, so a single ``except FileNotFoundError``
            catches both this and the directory-level case above.
        RankingsArtifactInvalid: An artifact's header does not match the live
            dataset, or multiple model_names are present in ``rankings_dir``.

    Notes
    -----
    Cross-version replay (an artifact produced by a different ``workrb_version``)
    is supported and is in fact the main reason this entry point exists: it
    lets you recompute metrics after metric definitions change without rerunning
    the model. Such a version drift only logs a warning. The real safety net
    is :func:`workrb.rankings.validate_header`'s structural and canary checks,
    which catch any change to dataset construction (deduplication,
    postprocessing, source data) and raise :class:`RankingsArtifactInvalid`.
    An unknown ``schema_version`` is a hard reject regardless of workrb
    version.
    """
    rankings_dir = Path(rankings_dir)
    if not rankings_dir.is_dir():
        raise FileNotFoundError(
            f"rankings_dir does not exist or is not a directory: {rankings_dir}"
        )

    non_ranking = [task.name for task in tasks if not isinstance(task, RankingTask)]
    if non_ranking:
        raise ValueError(
            f"evaluate_rankings only supports RankingTask, got non-ranking tasks: "
            f"{non_ranking}. Filter your task list before calling."
        )
    ranking_tasks: list[RankingTask] = [task for task in tasks if isinstance(task, RankingTask)]

    model_name, source_workrb_version = _peek_artifact_origin(rankings_dir)

    config = BenchmarkConfig(
        model_name=model_name,
        task_configs=[task.get_task_config() for task in tasks],
        languages=sorted({lang.value for task in tasks for lang in task.languages}),
        custom_metrics=metrics,
        output_folder=output_folder,
        description=description,
    )

    dataset_ids_to_evaluate = _get_dataset_ids_to_evaluate(
        tasks, language_aggregation_mode, execution_mode
    )

    key_metrics_by_task_group = {task.task_group.value: task.default_metrics for task in tasks}
    results = BenchmarkResults(
        task_results={},
        metadata=BenchmarkMetadata(
            model_name=model_name,
            total_evaluation_time=0.0,
            timestamp=time.time(),
            num_tasks=len(tasks),
            languages=_get_all_languages(tasks),
            resumed_from_checkpoint=False,
            language_aggregation_mode=language_aggregation_mode.value,
            replayed_from_workrb_version=source_workrb_version,
        ),
        key_metrics_by_task_group=key_metrics_by_task_group,
    )

    running_version = _get_workrb_version()
    start_time_benchmark = time.time()
    logger.info(f"Replaying rankings from {rankings_dir} (model: {model_name})")

    for task in ranking_tasks:
        scoped = dataset_ids_to_evaluate.get(task.name, [])
        if not scoped:
            continue

        if task.name not in results.task_results:
            results.task_results[task.name] = TaskResults(
                metadata=TaskResultMetadata(
                    task_group=task.task_group.value,
                    task_type=task.task_type.value,
                    label_type=task.label_type.value,
                    description=task.description,
                    split=task.split.value,
                ),
                datasetid_results={},
            )

        logger.info(f"{'=' * 60}")
        logger.info(f"Scoring task: {task.name}")

        for dataset_id in scoped:
            path = rankings_dir / rankings_filename(task.name, dataset_id)
            if not path.exists():
                raise RankingsArtifactMissing(path, task.name, dataset_id)

            header, scores = load_rankings_artifact(path)
            if header.get("model_name") != model_name:
                raise RankingsArtifactInvalid(
                    path,
                    f"model_name mismatch with sibling artifacts: header has "
                    f"'{header.get('model_name')}', expected '{model_name}'",
                )
            dataset = task.datasets[dataset_id]
            validate_header(
                header=header,
                dataset=dataset,
                task_name=task.name,
                dataset_id=dataset_id,
                split=task.split.value,
                path=path,
                running_workrb_version=running_version,
            )

            start_time_eval = time.time()
            prediction_matrix = materialize_prediction_matrix(
                scores=scores,
                num_queries=header["num_queries"],
                num_targets=header["num_targets"],
            )
            task_metrics = metrics.get(task.name) if metrics else None
            metrics_dict = task.compute_metrics_from_prediction_matrix(
                prediction_matrix=prediction_matrix,
                dataset_id=dataset_id,
                metrics=task_metrics,
            )
            evaluation_time = time.time() - start_time_eval

            logger.info(f"* {dataset_id} ({task.get_size_oneliner(dataset_id)})")
            _record_dataset_result(
                results=results,
                config=config,
                task=task,
                dataset_id=dataset_id,
                metrics_dict=metrics_dict,
                evaluation_time=evaluation_time,
            )

    results.metadata.total_evaluation_time = time.time() - start_time_benchmark
    config.save_final_result_artifacts(results)

    logger.info(f"{'=' * 60}")
    logger.info("✓ evaluate_rankings COMPLETE")
    logger.info(f"Total time: {results.metadata.total_evaluation_time:.2f}s")
    logger.info(
        format_results(
            results,
            display_per_task=False,
            display_per_task_group=False,
            display_per_language=False,
            display_overall=True,
            language_aggregation_mode=language_aggregation_mode,
        )
    )
    logger.info(f"{'=' * 60}")
    return results


def _peek_artifact_origin(rankings_dir: Path) -> tuple[str, str | None]:
    """Read ``(model_name, workrb_version)`` from the first artifact in ``rankings_dir``.

    All artifacts in one directory must agree on ``model_name``; later
    artifacts that disagree raise :class:`RankingsArtifactInvalid` when
    they are loaded for scoring. ``workrb_version`` is informational and
    surfaced in ``BenchmarkMetadata.replayed_from_workrb_version`` so the
    replay's ``results.json`` records which version produced the scores.
    """
    json_files = sorted(rankings_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No ranking artifacts (*.json) found in {rankings_dir}")
    header, _ = load_rankings_artifact(json_files[0])
    model_name = header.get("model_name")
    if not model_name:
        raise RankingsArtifactInvalid(json_files[0], "header missing 'model_name'")
    return model_name, header.get("workrb_version")


def get_tasks_overview(
    tasks: Sequence[Task],
    dataset_ids_to_evaluate: dict[str, list[str]] | None = None,
) -> str:
    """Get information about configured tasks as a formatted string summary.

    Parameters
    ----------
    tasks : Sequence[Task]
        All tasks configured for this benchmark run.
    dataset_ids_to_evaluate : dict[str, list[str]] or None
        When provided, only tasks present as keys with non-empty lists are
        shown, and only the listed dataset IDs appear under each task.
        When ``None``, all tasks and their full ``dataset_ids`` are shown.
    """
    # When filtering, only keep tasks that have at least one dataset to evaluate
    if dataset_ids_to_evaluate is not None:
        tasks = [t for t in tasks if dataset_ids_to_evaluate.get(t.name)]

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

        # Add size one-liner for each dataset
        dataset_ids = (
            dataset_ids_to_evaluate[task.name]
            if dataset_ids_to_evaluate is not None
            else task.dataset_ids
        )
        for dataset_id in dataset_ids:
            size_info = task.get_size_oneliner(dataset_id)
            if size_info:
                lines.append(f"  └─ {dataset_id}: {size_info}")

    lines.append("-" * 80)

    return "\n".join(lines)


def _get_all_languages(tasks: Sequence[Task]) -> list[str]:
    """Get all unique languages across tasks."""
    languages = set()
    for task in tasks:
        languages.update(task.languages)
    return sorted([str(lang) for lang in languages])


def _get_dataset_ids_to_evaluate(
    tasks: Sequence[Task],
    language_aggregation_mode: LanguageAggregationMode,
    execution_mode: ExecutionMode,
) -> dict[str, list[str]]:
    """Compute which dataset IDs per task are in scope for this run.

    Single source of truth for run scope. The returned dict drives the
    overview display, total-evaluation count, and pending-work filtering.
    Under ``ExecutionMode.ALL`` every dataset is in scope; under
    ``ExecutionMode.LAZY`` datasets incompatible with the chosen
    aggregation mode are dropped (and a warning is logged for each).

    Parameters
    ----------
    tasks : Sequence[Task]
        All tasks configured for this benchmark run.
    language_aggregation_mode : LanguageAggregationMode
        The aggregation mode to check compatibility against (only used
        when ``execution_mode`` is ``LAZY``).
    execution_mode : ExecutionMode
        ``LAZY`` filters incompatible datasets; ``ALL`` keeps everything.

    Returns
    -------
    dict[str, list[str]]
        Mapping of task name → list of dataset IDs to evaluate.
        Tasks whose datasets are all incompatible still appear as keys
        with an empty list.
    """
    if execution_mode == ExecutionMode.ALL:
        return {task.name: list(task.dataset_ids) for task in tasks}

    if language_aggregation_mode == LanguageAggregationMode.SKIP_LANGUAGE_AGGREGATION:
        return {task.name: list(task.dataset_ids) for task in tasks}

    result: dict[str, list[str]] = {}
    for task in tasks:
        filtered = []
        for dataset_id in task.dataset_ids:
            dataset_languages = task.get_dataset_languages(dataset_id)
            input_langs = sorted(lang.value for lang in dataset_languages.input_languages)
            output_langs = sorted(lang.value for lang in dataset_languages.output_languages)
            key = get_language_grouping_key(input_langs, output_langs, language_aggregation_mode)
            if key is None:
                logger.warning(
                    "Skipping dataset '%s' of task '%s': incompatible with "
                    "language_aggregation_mode '%s' (input_languages=%s, output_languages=%s).",
                    dataset_id,
                    task.name,
                    language_aggregation_mode.value,
                    input_langs,
                    output_langs,
                )
            else:
                filtered.append(dataset_id)
        result[task.name] = filtered
    return result


def _filter_pending_work(
    pending_work: list[tuple[Task, str]],
    dataset_ids_to_evaluate: dict[str, list[str]],
) -> list[tuple[Task, str]]:
    """Keep only pending work items whose dataset ID is in the evaluation scope.

    Parameters
    ----------
    pending_work : list of (Task, dataset_id) tuples
        The pending evaluations (already filtered by checkpoint).
    dataset_ids_to_evaluate : dict[str, list[str]]
        Mapping of task name → dataset IDs that are in scope for this run,
        as returned by :func:`_get_dataset_ids_to_evaluate`.

    Returns
    -------
    list of (Task, dataset_id) tuples
        Filtered list containing only in-scope work items.
    """
    scope = {
        (task_name, did) for task_name, dids in dataset_ids_to_evaluate.items() for did in dids
    }
    return [(task, did) for task, did in pending_work if (task.name, did) in scope]


def _init_checkpointing(
    tasks: Sequence[Task],
    config: BenchmarkConfig,
    model: ModelInterface,
    force_restart: bool,
    language_aggregation_mode: LanguageAggregationMode = LanguageAggregationMode.MONOLINGUAL_ONLY,
) -> tuple[BenchmarkResults, list[tuple[Task, str]]]:
    """Initialize checkpointing.

    Parameters
    ----------
    tasks : Sequence[Task]
        Tasks to evaluate.
    config : BenchmarkConfig
        Benchmark configuration.
    model : ModelInterface
        Model to evaluate.
    force_restart : bool
        If True, ignore checkpoints and restart from beginning.
    language_aggregation_mode : LanguageAggregationMode
        Language aggregation mode to store in the results metadata.

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
            # Warn if the checkpoint was saved with a different aggregation mode
            stored_mode = existing_results.metadata.language_aggregation_mode
            if stored_mode != language_aggregation_mode.value:
                logger.warning(
                    "Checkpoint was saved with language_aggregation_mode='%s', "
                    "but current call uses '%s'. Updating stored mode to '%s'.",
                    stored_mode,
                    language_aggregation_mode.value,
                    language_aggregation_mode.value,
                )
            existing_results.metadata.language_aggregation_mode = language_aggregation_mode.value

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
                language_aggregation_mode=language_aggregation_mode.value,
            ),
            key_metrics_by_task_group=key_metrics_by_task_group,
        )
    return results, pending_work


def _run_pending_work(
    config: BenchmarkConfig,
    work_by_task: dict[str, dict[str, Any]],
    results: BenchmarkResults,
    model: ModelInterface,
    metrics: dict[str, list[str]] | None,
    save_rankings: bool,
    total_evaluations: int,
) -> BenchmarkResults:
    """Run pending evaluations.

    Args:
        config: Benchmark configuration for checkpointing.
        work_by_task: Dictionary of task names to their pending datasets.
        results: BenchmarkResults object to store results.
        model: ModelInterface object to evaluate.
        metrics: Dictionary of task names to their custom metrics.
        save_rankings: If True, save full ranking score artifacts for ranking tasks.
        total_evaluations: Total number of compatible evaluations (for progress display).
    """
    # Run pending evaluations
    run_idx = results.get_num_evaluation_results()  # Already completed evaluations
    for work_info in work_by_task.values():
        task: Task = work_info["task"]
        pending_dataset_ids: list[str] = work_info["dataset_ids"]

        logger.info(f"{'=' * 60}")
        logger.info(f"Evaluating task: {task.name}")
        logger.info(f"Completed {run_idx} / {total_evaluations} evaluations. ")
        logger.info(f"Pending datasets for this task: {len(pending_dataset_ids)}")

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
                datasetid_results={},
            )

        # Evaluate pending datasets
        for dataset_id in pending_dataset_ids:
            logger.info(f"* Running dataset: {dataset_id} ({task.get_size_oneliner(dataset_id)})")

            # Get metrics for this task
            task_metrics = None
            if metrics and task.name in metrics:
                task_metrics = metrics[task.name]

            try:
                start_time_eval = time.time()
                if save_rankings and isinstance(task, RankingTask):
                    prediction_matrix = task.compute_prediction_matrix(
                        model=model, dataset_id=dataset_id
                    )
                    dataset_results = task.compute_metrics_from_prediction_matrix(
                        prediction_matrix=prediction_matrix,
                        dataset_id=dataset_id,
                        metrics=task_metrics,
                    )
                    rankings_path = config.save_rankings_artifact(
                        task_name=task.name,
                        dataset_id=dataset_id,
                        split=task.split.value,
                        dataset=task.datasets[dataset_id],
                        prediction_matrix=prediction_matrix,
                    )
                    logger.info(f"\tSaved ranking scores to: {rankings_path}")
                else:
                    dataset_results: dict[str, float] = task.evaluate(
                        model=model, metrics=task_metrics, dataset_id=dataset_id
                    )
                evaluation_time = time.time() - start_time_eval

                _record_dataset_result(
                    results=results,
                    config=config,
                    task=task,
                    dataset_id=dataset_id,
                    metrics_dict=dataset_results,
                    evaluation_time=evaluation_time,
                )
                run_idx += 1
            except Exception as e:
                logger.error(f"Error: {e}")
                raise e

    logger.info(f"Completed {run_idx} / {total_evaluations} evaluations. ")
    return results


def _record_dataset_result(
    results: BenchmarkResults,
    config: BenchmarkConfig,
    task: Task,
    dataset_id: str,
    metrics_dict: dict[str, float],
    evaluation_time: float,
) -> None:
    """Attach a per-dataset metrics result, checkpoint, and log the key metric.

    Shared between :func:`evaluate` and :func:`evaluate_rankings` so both paths
    produce identical ``BenchmarkResults`` bookkeeping.
    """
    dataset_languages = task.get_dataset_languages(dataset_id)
    results.task_results[task.name].datasetid_results[dataset_id] = MetricsResult(
        evaluation_time=evaluation_time,
        metrics_dict=metrics_dict,
        input_languages=sorted(lang.value for lang in dataset_languages.input_languages),
        output_languages=sorted(lang.value for lang in dataset_languages.output_languages),
    )

    if config:
        config.save_results_checkpoint(results)

    # Prefer the task's first default metric, but fall back to whatever was
    # actually computed: a caller-supplied `metrics` set need not include the
    # default key metric, and indexing it directly would raise KeyError.
    key_metric = task.default_metrics[0]
    if key_metric not in metrics_dict:
        key_metric = next(iter(metrics_dict), None)
    if key_metric is not None:
        logger.info(f"\t{key_metric}: {metrics_dict[key_metric]:.3f}")
