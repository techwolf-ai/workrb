import pprint
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


class TaskResultMetadata(BaseModel):
    """Metadata for a task result."""

    task_group: str
    task_type: str
    label_type: str
    description: str
    split: str


class MetricsResult(BaseModel):
    """Metric results for a single evaluation run.

    In the becnhmark, this is a single evaluation run for a single language.
    """

    evaluation_time: float = Field(ge=0)
    metrics_dict: dict[str, Any] = Field(default_factory=dict)
    """ Dictionary of metric names to their computed values. """


class TaskResults(BaseModel):
    """Results for a task."""

    metadata: TaskResultMetadata
    language_results: dict[str, MetricsResult]  # language -> results
    """ Dictionary of language codes to their computed results. """


class BenchmarkMetadata(BaseModel):
    """Metadata for a benchmark run."""

    model_name: str
    total_evaluation_time: float = Field(ge=0)
    timestamp: float
    num_tasks: int = Field(ge=1)
    languages: list[str]
    resumed_from_checkpoint: bool = False


class ResultTagString(BaseModel):
    """String representation of a result tag."""

    model_config = {"frozen": True}
    """Make pydantic immutable."""

    name: str
    metric_name: str
    aggregation: str
    grouping_name: str | None = None

    def __str__(self) -> str:
        """Represent string result tag.

        For example:
        - With grouping: "mean_per_task/cls_job2skill/f1_macro/mean"
        - Without grouping: "mean_benchmark/f1_macro/mean"
        """
        ret = [self.name]
        if self.grouping_name:
            ret.append(self.grouping_name)
        ret.append(self.metric_name)
        ret.append(self.aggregation)
        return "/".join(ret)


class BenchmarkResults(BaseModel):
    """Top-level benchmark results."""

    task_results: dict[str, TaskResults] = Field(default_factory=dict)
    """ Dictionary tracking results per task. """
    metadata: BenchmarkMetadata
    key_metrics_by_task_group: dict[str, list[str]] = Field(default_factory=dict)
    """ Dictionary mapping task groups to their key metric names for reporting. """

    def __str__(self) -> str:
        """String representation of the benchmark results."""
        lines = [
            "BenchmarkResults",
            "=" * 80,
            pprint.pformat(self.get_summary_metrics()),
        ]
        return "\n".join(lines)

    def get_num_evaluation_results(self) -> int:
        """Get the total number of evaluation results."""
        return sum(len(task.language_results) for task in self.task_results.values())

    def get_summary_metrics(self, aggregations: tuple = ("mean", "ci_margin")) -> dict[str, float]:
        """
        Get summary metrics for the benchmark results.
        """
        mean_per_task = self._aggregate_per_task(
            aggregations=aggregations,
        )
        mean_per_task_group = self._aggregate_per_task_group(
            aggregations=aggregations, task_results=mean_per_task
        )
        mean_per_task_type = self._aggregate_per_task_type(
            aggregations=aggregations, task_group_results=mean_per_task_group
        )
        mean_benchmark = self._aggregate_benchmark(
            aggregations=aggregations, task_type_results=mean_per_task_type
        )
        mean_per_language = self._aggregate_per_language(
            aggregations=aggregations,
        )

        combined = {
            **mean_per_language,
            **mean_per_task,
            **mean_per_task_group,
            **mean_per_task_type,
            **mean_benchmark,
        }
        return {str(k): v for k, v in combined.items()}

    def _aggregate_per_task(
        self,
        tag_name: str = "mean_per_task",
        aggregations: tuple = ("mean", "stderr", "ci_margin"),
    ) -> dict[ResultTagString, float]:
        """Aggregate results per task, by aggregating over languages within tasks."""
        # Collect metric values per task
        raw_results = defaultdict(list)
        for task_name, task_result in self.task_results.items():
            for lang_metrics_result in task_result.language_results.values():
                for metric_name, metric_value in lang_metrics_result.metrics_dict.items():
                    raw_results[(task_name, metric_name)].append(metric_value)

        # Compute stats
        results = {}
        for (task_name, metric_name), values in raw_results.items():
            stats = self._compute_stats(values)
            for agg in aggregations:
                assert agg in stats, f"Aggregation {agg} not found in stats: {stats.keys()}"
                tag = ResultTagString(
                    name=tag_name, metric_name=metric_name, aggregation=agg, grouping_name=task_name
                )
                results[tag] = stats[agg]
        return results

    def _aggregate_per_task_group(
        self,
        tag_name: str = "mean_per_task_group",
        aggregations: tuple = ("mean", "stderr", "ci_margin"),
        task_results: dict[ResultTagString, float] | None = None,
    ) -> dict[ResultTagString, float]:
        """Aggregate results per task group.

        First aggregates over languages within tasks, then over tasks within task groups.
        """
        task_results = task_results or self._aggregate_per_task(aggregations=("mean",))

        task_group_list_results = defaultdict(list)
        for task_result_tag, value in task_results.items():
            task_name = task_result_tag.grouping_name
            metric_name = task_result_tag.metric_name
            aggregation = task_result_tag.aggregation

            if aggregation != "mean":  # Collect means only
                continue

            assert task_name in self.task_results, f"Task {task_name} not found in task results"
            task_group_name = self.task_results[task_name].metadata.task_group

            task_group_list_results[(task_group_name, metric_name)].append(value)

        # Compute task group stats
        task_group_results = {}
        for (task_group_name, metric_name), values in task_group_list_results.items():
            stats = self._compute_stats(values)

            for agg in aggregations:
                assert agg in stats, f"Aggregation {agg} not found in stats: {stats.keys()}"
                tag = ResultTagString(
                    name=tag_name,
                    metric_name=metric_name,
                    aggregation=agg,
                    grouping_name=task_group_name,
                )
                task_group_results[tag] = stats[agg]
        return task_group_results

    def _aggregate_per_task_type(
        self,
        tag_name: str = "mean_per_task_type",
        aggregations: tuple = ("mean", "stderr", "ci_margin"),
        task_group_results: dict[ResultTagString, float] | None = None,
    ) -> dict[ResultTagString, float]:
        """Aggregate results per task type.

        First aggregates over languages within tasks, then over tasks within task groups,
        then over task groups within task types.
        """
        task_group_results = task_group_results or self._aggregate_per_task_group(
            aggregations=("mean",)
        )

        # Mapping from task group name to task type name
        task_group_to_task_type = {}
        for task_result in self.task_results.values():
            task_group_to_task_type[task_result.metadata.task_group] = (
                task_result.metadata.task_type
            )
            assert task_group_to_task_type[task_result.metadata.task_group] is not None, (
                f"Task type not found for task group {task_result.metadata.task_group}"
            )

        # Collect mean metric values per task type
        task_type_list_results = defaultdict(list)
        for task_group_result_tag, value in task_group_results.items():
            metric_name = task_group_result_tag.metric_name
            aggregation = task_group_result_tag.aggregation
            task_group_name = task_group_result_tag.grouping_name
            task_type_name = task_group_to_task_type[task_group_name]

            if aggregation != "mean":  # Collect means only
                continue

            task_type_list_results[(task_type_name, metric_name)].append(value)

        # Compute task type stats
        task_type_results = {}
        for (task_type_name, metric_name), values in task_type_list_results.items():
            stats = self._compute_stats(values)

            for agg in aggregations:
                assert agg in stats, f"Aggregation {agg} not found in stats: {stats.keys()}"
                tag = ResultTagString(
                    name=tag_name,
                    metric_name=metric_name,
                    aggregation=agg,
                    grouping_name=task_type_name,
                )
                task_type_results[tag] = stats[agg]
        return task_type_results

    def _aggregate_benchmark(
        self,
        tag_name: str = "mean_benchmark",
        aggregations: tuple = ("mean", "stderr", "ci_margin"),
        task_type_results: dict[ResultTagString, float] | None = None,
    ) -> dict[ResultTagString, float]:
        """Aggregate results over all task types.

        It applies the following aggregation steps:
        1. Aggregates over languages within tasks (e.g. en, fr, de, nl)
        2. Aggregates over tasks within task groups (e.g. ESCOjob2skill, Customjob2skill)
        3. Aggregates over task groups per task type (e.g. classification, ranking)
        4. Aggregates over task types for final benchmark scores
        """
        task_type_results = task_type_results or self._aggregate_per_task_type(
            aggregations=("mean",)
        )

        metric_list_results = defaultdict(list)
        for task_type_result_tag, value in task_type_results.items():
            aggregation = task_type_result_tag.aggregation
            if aggregation != "mean":  # Collect means only
                continue

            metric_name = task_type_result_tag.metric_name
            metric_list_results[metric_name].append(value)

        metric_results = {}
        for metric_name, values in metric_list_results.items():
            stats = self._compute_stats(values)
            for agg in aggregations:
                assert agg in stats, f"Aggregation {agg} not found in stats: {stats.keys()}"
                tag = ResultTagString(
                    name=tag_name, metric_name=metric_name, aggregation=agg, grouping_name=None
                )
                metric_results[tag] = stats[agg]
        return metric_results

    def _aggregate_per_language(
        self,
        tag_name: str = "mean_per_language",
        aggregations: tuple = ("mean", "stderr", "ci_margin"),
    ) -> dict[ResultTagString, float]:
        """Aggregate results per language.

        Collects language-specific results over all tasks, and aggregates all availble results.
        Results may be imbalanced if tasks support different languages.
        """
        # Collect metric values per task
        raw_results = defaultdict(list)
        for task_result in self.task_results.values():
            for language, metrics_result in task_result.language_results.items():
                for metric_name, metric_value in metrics_result.metrics_dict.items():
                    raw_results[(language, metric_name)].append(metric_value)

        # Compute stats
        results = {}
        for (language, metric_name), values in raw_results.items():
            stats = self._compute_stats(values)
            for agg in aggregations:
                assert agg in stats, f"Aggregation {agg} not found in stats: {stats.keys()}"
                tag = ResultTagString(
                    name=tag_name, metric_name=metric_name, aggregation=agg, grouping_name=language
                )
                results[tag] = stats[agg]
        return results

    def _compute_stats(self, values: list[float]) -> dict[str, float]:
        """Compute comprehensive statistics for a group of values."""
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        stderr_val = float(std_val / np.sqrt(len(values)))

        # 95% confidence interval margin
        if len(values) > 1:
            dof = len(values) - 1
            t_crit = stats.t.ppf(0.975, dof)  # 95% CI
            ci_margin = float(t_crit * stderr_val)
        else:
            ci_margin = 0.0

        return {
            "mean": mean_val,
            "std": std_val,
            "stderr": stderr_val,
            "ci_margin": ci_margin,
            "count": float(len(values)),
        }

    def _get_flat_dataframe(self) -> pd.DataFrame:
        """Get flat dataframe of the benchmark results with each metric value as a separate row."""
        data = []
        for task_name, task_result in self.task_results.items():
            for language, metrics_result in task_result.language_results.items():
                for metric_name, metric_value in metrics_result.metrics_dict.items():
                    data.append(
                        {
                            "task_name": str(task_name),
                            "task_group": str(task_result.metadata.task_group),
                            "task_type": str(task_result.metadata.task_type),
                            # "task_label_type": str(task_result.metadata.label_type),
                            # "task_split": str(task_result.metadata.split),
                            "task_language": str(language),
                            "metric_name": str(metric_name),
                            "metric_value": float(metric_value),
                        }
                    )

        return pd.DataFrame(data)
