"""
Configuration management for WorkRB benchmarks.

BenchmarkConfig handles both configuration settings and checkpoint management,
using unified BenchmarkResults storage for both checkpoints and final results.
"""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from workrb.results import BenchmarkResults
from workrb.tasks.abstract import Task

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for a WorkRB benchmark run.

    This config can be saved to YAML and used for resuming interrupted benchmarks.
    """

    # Model configuration
    model_name: str
    model_class: str = "BiEncoderModel"
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # Benchmark configuration
    task_configs: list[dict[str, Any]] = field(default_factory=list)
    """ List of task configurations initialized with user parameters. """
    languages: list[str] = field(default_factory=lambda: ["en"])
    """ List of languages defined for benchmark to evaluate on. """

    # Evaluation configuration
    custom_metrics: dict[str, list[str]] | None = None
    output_folder: str = "results"

    # Checkpoint configuration
    checkpoint_enabled: bool = True
    checkpoint_frequency: str = "per_task"  # "per_task", "per_language", "per_evaluation"

    # Metadata
    created_at: str = ""
    description: str = ""

    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_workrb(
        cls,
        model,
        tasks: Sequence[Task],
        output_folder: str,
        custom_metrics: dict[str, list[str]] | None = None,
        description: str = "",
    ) -> "BenchmarkConfig":
        """
        Create config from WorkRB model and tasks.

        Args:
            model: Model instance
            tasks: List of task instances
            output_folder: Output directory
            custom_metrics: Custom metrics per task
            description: Description of the benchmark

        Returns
        -------
            BenchmarkConfig instance
        """
        # Extract model info
        model_name = model.name
        model_class = model.__class__.__name__

        # Extract task info
        task_configs = []
        all_languages = set()

        for task in tasks:
            task_configs.append(task.get_task_config())
            all_languages.update(task.languages)

        return cls(
            model_name=model_name,
            model_class=model_class,
            task_configs=task_configs,
            languages=sorted([lang.value for lang in all_languages]),
            custom_metrics=custom_metrics,
            output_folder=output_folder,
            description=description,
        )

    def save_yaml(self, filepath: str | Path) -> None:
        """Save config to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, filepath: str | Path) -> "BenchmarkConfig":
        """Load config from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        return Path(self.output_folder)

    def get_config_path(self) -> Path:
        """Get the path where config should be saved."""
        return self.get_output_path() / "config.yaml"

    def get_checkpoint_path(self) -> Path:
        """Get the path where checkpoint should be saved."""
        return self.get_output_path() / "checkpoint.json"

    def get_results_path(self) -> Path:
        """Get the path where final results should be saved."""
        return self.get_output_path() / "results.json"

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return self.get_checkpoint_path().exists()

    def has_results(self) -> bool:
        """Check if final results exist."""
        return self.get_results_path().exists()

    def save_results_checkpoint(self, results: BenchmarkResults) -> None:
        """
        Save current BenchmarkResults as checkpoint.

        This unified approach eliminates the need for separate checkpoint format.
        """
        if not self.checkpoint_enabled:
            return

        checkpoint_path = self.get_checkpoint_path()

        # Create checkpoint metadata
        checkpoint_data = {
            "config": asdict(self),
            "last_updated": time.time(),
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
            "results": results.model_dump(),
        }

        # Save checkpoint
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.debug(f"Incremental results checkpoint saved to {checkpoint_path}")

    def save_final_result_artifacts(self, results: BenchmarkResults) -> None:
        """Save final results to separate files."""
        results_path = self.get_results_path()
        results_path.parent.mkdir(parents=True, exist_ok=True)

        data = results.model_dump()
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Results saved to {results_path}")

        # Save config in separate file
        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_yaml(config_path)
        logger.debug(f"Config saved to {config_path}")

    def restore_results_from_checkpoint(self) -> BenchmarkResults | None:
        """Restore BenchmarkResults from checkpoint."""
        if not self.has_checkpoint():
            return None

        checkpoint_path = self.get_checkpoint_path()
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            if "results" in checkpoint_data:
                results = BenchmarkResults.model_validate(checkpoint_data["results"])
                logger.debug(f"Restored results from checkpoint: {checkpoint_path}")
                return results
        except Exception as e:
            logger.warning(f"Could not load from checkpoint: {e}")

    def get_pending_work(
        self,
        results: BenchmarkResults | None,
        tasks: Sequence[Task],
    ) -> list[tuple]:
        """Determine what work still needs to be done.

        Work is defined as a (task, language) combination that is not completed.
        """
        pending_work = []
        for task in tasks:
            for language in task.languages:
                # Successful completed (task, language) combination
                if (
                    results is not None
                    and task.name in results.task_results
                    and language in results.task_results[task.name].language_results
                ):
                    continue

                # Add to pending work
                pending_work.append((task, language))

        return pending_work

    def validate_results_contained_in_tasks(
        self, results: BenchmarkResults | None, tasks: Sequence[Task]
    ) -> None:
        """
        Validate that pre-existing results are defined in the current benchmark tasks.

        This ensures both the benchmark results and the workrb tasks are consistent, preventing
        unexpected behavior for undefined tasks in the results.
        """
        if results is None:
            return

        current_task_names = {task.name for task in tasks}

        for result_task_name in results.task_results:
            if result_task_name not in current_task_names:
                raise ValueError(
                    f"Result checkpoint contains Task {result_task_name}, but is not defined in current benchmark tasks."
                )
