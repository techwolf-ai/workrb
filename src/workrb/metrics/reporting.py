"""
Simple results display system using BenchmarkResults aggregation methods.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from workrb.results import BenchmarkResults
from workrb.types import LanguageAggregationMode

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
    language_aggregation_mode: LanguageAggregationMode | None = None,
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
        language_aggregation_mode: How to determine the grouping language for
            aggregation. When ``None``, reads the mode stored in
            ``results.metadata.language_aggregation_mode``.

    Returns
    -------
        String containing formatted results
    """
    if language_aggregation_mode is None:
        language_aggregation_mode = LanguageAggregationMode(
            results.metadata.language_aggregation_mode
        )

    # Get aggregations - always include mean and error_type
    aggregations = ("mean", error_type) if show_error else ("mean",)

    # Get key metrics if filtering
    key_metrics = set()
    if show_only_key_metrics:
        for metrics in results.key_metrics_by_task_group.values():
            key_metrics.update(metrics)

    # Compute all aggregation levels at once
    all_results = results._get_summary_metrics(
        aggregations=aggregations,
        language_aggregation_mode=language_aggregation_mode,
    )

    # Partition results by tag name prefix for selective display
    results_by_level: dict[str, dict] = {
        "mean_per_task": {},
        "mean_per_task_group": {},
        "mean_per_language": {},
        "mean_benchmark": {},
    }
    for tag, value in all_results.items():
        if tag.name in results_by_level:
            results_by_level[tag.name][tag] = value

    # Display each requested aggregation level
    metric_strs = []
    if display_per_task:
        metric_strs.append(
            _display_aggregation(
                results_by_level["mean_per_task"],
                key_metrics,
                value_format,
                show_error,
                error_type,
            )
        )

    if display_per_task_group:
        metric_strs.append(
            _display_aggregation(
                results_by_level["mean_per_task_group"],
                key_metrics,
                value_format,
                show_error,
                error_type,
            )
        )

    if display_per_language:
        metric_strs.append(
            _display_aggregation(
                results_by_level["mean_per_language"],
                key_metrics,
                value_format,
                show_error,
                error_type,
            )
        )

    if display_overall:
        metric_strs.append(
            _display_aggregation(
                results_by_level["mean_benchmark"],
                key_metrics,
                value_format,
                show_error,
                error_type,
            )
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


# ---------------------------------------------------------------------------
# LaTeX table formatting
# ---------------------------------------------------------------------------


def format_results_latex(
    results_list: Sequence[BenchmarkResults],
    target_metric: str = "map",
    aggregation_level: Literal["task_group", "task"] = "task_group",
    value_format: str = "{value:.1f}",
    scale_factor: float = 100.0,
    caption: str = "Benchmark Results",
    label: str = "tab:benchmark_results",
    model_groups: list[list[int]] | None = None,
    short_names: dict[str, str] | None = None,
    highlight_best: bool = True,
    resize: str | None = r"\columnwidth",
    show_dataset_counts: bool = False,
) -> str:
    r"""Build a LaTeX table comparing multiple model results.

    Parameters
    ----------
    results_list:
        One :class:`BenchmarkResults` per model, each with its own stored
        ``language_aggregation_mode``.
    target_metric:
        Metric name to extract (e.g. ``"map"``, ``"mrr"``).
    aggregation_level:
        Column granularity — ``"task_group"`` or ``"task"``.
    value_format:
        Python format string applied to each cell value.  Use ``{value}``
        as the placeholder (e.g. ``"{value:.1f}"``).
    scale_factor:
        Multiply raw metric values by this factor before formatting
        (default ``100.0`` gives percentage-style numbers).
    caption:
        LaTeX ``\caption`` text.
    label:
        LaTeX ``\label`` text.
    model_groups:
        Optional grouping of model indices for ``\midrule`` separation.
        E.g. ``[[0, 1, 2], [3, 4]]`` draws a ``\midrule`` between index 2
        and 3.  When *None*, all models are in a single group.
    short_names:
        Rename dictionary applied to both model names and column headers.
        Keys in insertion order also determine column ordering when the key
        matches a column name.
    highlight_best:
        Bold the highest value in each column.
    resize:
        Width argument for ``\resizebox``.  When set (default
        ``r"\columnwidth"``), the tabular is wrapped in
        ``\resizebox{<value>}{!}{…}``.  Pass *None* to disable resizing.
    show_dataset_counts:
        When *True*, append a ``#D`` row after each model group showing
        the number of datasets that contributed to each column's score.
        If all models in a group share the same counts, a single row is
        shown; otherwise one row per model is emitted.

    Returns
    -------
    str
        Complete LaTeX ``table`` environment string.
    """
    short_names = short_names or {}
    tag_level = "mean_per_task_group" if aggregation_level == "task_group" else "mean_per_task"

    # ---- collect per-model data ------------------------------------------
    rows: dict[str, dict[str, float]] = {}  # model_name -> {col_name: value}
    missing_warnings: list[str] = []

    for results in results_list:
        mode = LanguageAggregationMode(results.metadata.language_aggregation_mode)
        all_metrics = results._get_summary_metrics(
            aggregations=("mean",),
            language_aggregation_mode=mode,
        )

        model_name = results.metadata.model_name
        row: dict[str, float] = {}

        for tag, value in all_metrics.items():
            if tag.aggregation != "mean":
                continue
            if tag.metric_name != target_metric:
                continue

            if tag.name == tag_level and tag.grouping_name is not None:
                row[tag.grouping_name] = value
            elif tag.name == "mean_benchmark":
                row["Overall"] = value

        rows[model_name] = row

    # ---- collect dataset counts per model --------------------------------
    dataset_counts: dict[str, dict[str, int]] = {}  # model_name -> {col: count}
    if show_dataset_counts:
        for results in results_list:
            mode = LanguageAggregationMode(results.metadata.language_aggregation_mode)
            counts = results.get_dataset_counts(
                aggregation_level=aggregation_level,
                language_aggregation_mode=mode,
            )
            dataset_counts[results.metadata.model_name] = counts

    # ---- build DataFrame -------------------------------------------------
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Model"

    # Warn about completely missing columns (metric not present for any model)
    all_columns = set(df.columns) - {"Overall"}
    missing_cols = [c for c in sorted(all_columns) if df[c].isna().all()]
    if missing_cols:
        msg = (
            f"Metric '{target_metric}' not found for "
            f"{'task groups' if aggregation_level == 'task_group' else 'tasks'}: "
            f"{', '.join(missing_cols)}. These columns will show '--'."
        )
        logger.warning(msg)
        missing_warnings.append(msg)

    # Warn about partially missing cells
    for model_name in df.index:
        missing_for_model = [
            c for c in df.columns if c != "Overall" and pd.isna(df.loc[model_name, c])
        ]
        if missing_for_model:
            msg = (
                f"Metric '{target_metric}' missing for model '{model_name}' in: "
                f"{', '.join(missing_for_model)}."
            )
            logger.warning(msg)
            missing_warnings.append(msg)

    # ---- column ordering -------------------------------------------------
    # Use short_names insertion order for columns that match, then alphabetical remainder
    ordered_cols: list[str] = []
    remaining = set(df.columns) - {"Overall"}
    for key in short_names:
        # key could be the original name or the short name target
        if key in remaining:
            ordered_cols.append(key)
            remaining.discard(key)
    for col in sorted(remaining):
        ordered_cols.append(col)
    ordered_cols.append("Overall")
    df = df[[c for c in ordered_cols if c in df.columns]]

    # ---- apply scale factor ----------------------------------------------
    df = df * scale_factor

    # ---- apply short_names -----------------------------------------------
    df = df.rename(columns=short_names, index=short_names)

    # ---- find best per column per group for highlighting -------------------
    # Build a set of best (row_index, col) pairs to bold
    best_cells: set[tuple[int, str]] = set()
    if highlight_best:
        resolved_groups = model_groups if model_groups is not None else [list(range(len(df)))]
        for group in resolved_groups:
            group_df = df.iloc[group]
            for col in group_df.columns:
                numeric_vals = pd.to_numeric(group_df[col], errors="coerce")
                if numeric_vals.notna().any():
                    best_val = numeric_vals.max()
                    for idx in group:
                        if df.iloc[idx][col] == best_val:
                            best_cells.add((idx, col))

    # ---- format cells ----------------------------------------------------
    def _fmt_cell(val: float, row_idx: int, col: str) -> str:
        if pd.isna(val):
            return "--"
        formatted = value_format.format(value=val)
        if highlight_best and (row_idx, col) in best_cells:
            return f"\\textbf{{{formatted}}}"
        return formatted

    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        formatted_df[col] = [_fmt_cell(v, i, col) for i, v in enumerate(df[col])]

    # ---- build LaTeX string ----------------------------------------------
    n_cols = len(formatted_df.columns)
    col_spec = "l" + "c" * (n_cols - 1) + "|c"  # last col (Overall) separated

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if resize:
        lines.append(rf"\resizebox{{{resize}}}{{!}}{{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]
    )

    # Header row
    header = "Model & " + " & ".join(formatted_df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Determine group boundaries
    if model_groups is None:
        model_groups = [list(range(len(formatted_df)))]

    # ---- prepare dataset count rows per group (if requested) ---------------
    # For each group, check if all models share the same counts; if so emit
    # one shared row, otherwise emit one row per model.
    group_count_rows: list[list[str]] = []  # parallel to model_groups
    if show_dataset_counts:
        # Map original model names to their df index order
        original_names = list(rows.keys())  # preserves insertion order
        for group in (model_groups if model_groups is not None else [list(range(len(formatted_df)))]):
            # Collect count dicts for models in this group
            group_counts: list[dict[str, int]] = []
            group_model_names: list[str] = []
            for i in group:
                orig_name = original_names[i]
                group_model_names.append(orig_name)
                group_counts.append(dataset_counts.get(orig_name, {}))

            # Check if all models in group have identical counts
            all_same = len(set(tuple(sorted(c.items())) for c in group_counts)) <= 1

            count_lines: list[str] = []
            if all_same and group_counts:
                # Single shared row
                cells = []
                for col in formatted_df.columns:
                    # Reverse-lookup original column name from short_names
                    orig_col = col
                    for k, v in short_names.items():
                        if v == col:
                            orig_col = k
                            break
                    cells.append(str(group_counts[0].get(orig_col, "--")))
                count_lines.append(r"\textit{\#D} & " + " & ".join(cells) + r" \\")
            else:
                for model_name, counts in zip(group_model_names, group_counts):
                    display_name = short_names.get(model_name, model_name)
                    cells = []
                    for col in formatted_df.columns:
                        orig_col = col
                        for k, v in short_names.items():
                            if v == col:
                                orig_col = k
                                break
                        cells.append(str(counts.get(orig_col, "--")))
                    count_lines.append(
                        rf"\textit{{{display_name} \#D}} & " + " & ".join(cells) + r" \\"
                    )
            group_count_rows.append(count_lines)

    row_idx = 0
    for group_i, group in enumerate(model_groups):
        for i in group:
            model = formatted_df.index[i]
            cells = " & ".join(formatted_df.iloc[i])
            lines.append(f"{model} & {cells}" + r" \\")
            row_idx += 1
        # Add dataset count rows for this group
        if show_dataset_counts and group_count_rows:
            for count_line in group_count_rows[group_i]:
                lines.append(count_line)
        # Add midrule between groups (not after the last)
        if group_i < len(model_groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    # Print warnings summary if any
    if missing_warnings:
        warning_block = "\n".join(f"% WARNING: {w}" for w in missing_warnings)
        latex_str = warning_block + "\n" + latex_str

    return latex_str
