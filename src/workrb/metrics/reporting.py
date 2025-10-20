"""
Simple results display system using BenchmarkResults aggregation methods.
"""

import logging
from collections import defaultdict
from typing import Literal

from workrb.results import BenchmarkResults

logger = logging.getLogger(__name__)


def format_results(
    results: BenchmarkResults,
    display_per_task: bool = False,
    display_per_task_group: bool = False,
    display_per_language: bool = False,
    display_overall: bool = True,
    value_format: str = "{value:.2f}",
    show_error: bool = True,
    error_type: Literal["ci_margin", "stderr", "std"] = "ci_margin",
    show_only_key_metrics: bool = True,
) -> str:
    """
    Display benchmark results using BenchmarkResults aggregation methods.

    Args:
        results: BenchmarkResults object to display
        display_per_task: If True, display results aggregated per task
        display_per_task_group: If True, display results aggregated per task group
        display_per_language: If True, display results aggregated per language
        display_overall: If True, display overall benchmark results
        value_format: Format string for values. Use {value} as placeholder.
                     Examples: "{value:.1%}", "{value:.3f}", "{value:.2f}%"
        show_error: Whether to show error bars
        error_type: Type of error to show - "ci_margin", "stderr", or "std"
        show_only_key_metrics: If True, only show key metrics defined in task groups

    Returns
    -------
        String containing formatted results
    """
    # Get aggregations - always include mean and error_type
    aggregations = ("mean", error_type) if show_error else ("mean",)

    # Get key metrics if filtering
    key_metrics = set()
    if show_only_key_metrics:
        for metrics in results.key_metrics_by_task_group.values():
            key_metrics.update(metrics)

    # Display each requested aggregation level
    metric_strs = []
    if display_per_task:
        agg_results = results._aggregate_per_task(aggregations=aggregations)
        metric_strs.append(
            _display_aggregation(agg_results, key_metrics, value_format, show_error, error_type)
        )

    if display_per_task_group:
        agg_results = results._aggregate_per_task_group(aggregations=aggregations)
        metric_strs.append(
            _display_aggregation(agg_results, key_metrics, value_format, show_error, error_type)
        )

    if display_per_language:
        agg_results = results._aggregate_per_language(aggregations=aggregations)
        metric_strs.append(
            _display_aggregation(agg_results, key_metrics, value_format, show_error, error_type)
        )

    if display_overall:
        agg_results = results._aggregate_benchmark(aggregations=aggregations)
        metric_strs.append(
            _display_aggregation(agg_results, key_metrics, value_format, show_error, error_type)
        )

    return "\n".join(metric_strs)


def _display_aggregation(
    agg_results: dict,
    key_metrics: set,
    value_format: str,
    show_error: bool,
    error_type: str,
) -> str:
    """Display results from an aggregation method."""
    # Organize results by grouping
    grouped_data = defaultdict(lambda: defaultdict(dict))

    for tag, value in agg_results.items():
        # Filter to key metrics if requested
        if key_metrics and tag.metric_name not in key_metrics:
            continue

        grouping = tag.grouping_name
        metric = tag.metric_name
        agg_type = tag.aggregation

        grouped_data[grouping][metric][agg_type] = value

    # Display each group
    metric_strs = []
    for grouping in sorted(grouped_data.keys(), key=lambda x: (x is None, x)):
        metric_strs.append(
            _display_row(grouping, grouped_data[grouping], value_format, show_error, error_type)
        )

    return "\n".join(metric_strs)


def _display_row(
    row_label: str | None,
    metrics_data: dict[str, dict[str, float]],
    value_format: str,
    show_error: bool,
    error_type: str,
) -> str:
    """Display a single row of results."""
    metric_strs = []

    for metric_name in sorted(metrics_data.keys()):
        metric_values = metrics_data[metric_name]
        mean_val = metric_values.get("mean", 0.0)

        # Format value
        value_str = value_format.format(value=mean_val)

        # Add error if requested
        if show_error and error_type in metric_values:
            error_val = metric_values[error_type]
            error_str = value_format.format(value=error_val)
            value_str = f"{value_str}±{error_str}"

        metric_strs.append(f"{metric_name} {value_str}")

    # Log the row
    if row_label is None:
        return "Overall: " + ",\t".join(metric_strs)

    metrics_display = ",\t".join(metric_strs)
    return f"{row_label:<30} {metrics_display}"
