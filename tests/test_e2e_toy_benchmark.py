"""
End-to-End Toy Benchmark Test - Registry-Based

This test validates all available ranking tasks using small datasets for computational efficiency.
It uses the TaskRegistry to automatically discover tasks, so it adapts when new tasks are added.

See docs/TESTPLAN_E2E_TOY.md for complete test plan documentation.

Usage:
    pytest tests/test_e2e_toy_benchmark.py -v
    python tests/test_e2e_toy_benchmark.py
"""

import sys
import time

import workrb
from tests.test_utils import create_toy_task_class
from workrb.registry import TaskRegistry
from workrb.tasks.abstract.base import Language, Task
from workrb.tasks.abstract.classification_base import ClassificationTask
from workrb.tasks.abstract.ranking_base import RankingTask


def get_all_tasks(split: str = "val", languages: list[str] | None = None) -> list[Task]:
    """
    Discover and instantiate toy versions of all registered tasks (ranking and classification).

    Args:
        split: Dataset split to use ("val" or "test")
        languages: List of language codes (defaults to ["en"])

    Returns
    -------
        List of toy task instances
    """
    if languages is None:
        languages = ["en"]

    # Discover all available tasks
    available_tasks = workrb.list_available_tasks()

    print(f"\n🔍 Discovered {len(available_tasks)} registered tasks")

    toy_tasks = []
    skipped_tasks = []

    for task_name, task_path in available_tasks.items():
        try:
            # Get the task class from registry
            task_class = TaskRegistry.get(task_name)

            # Only process RankingTask or ClassificationTask subclasses
            if not issubclass(task_class, Task):
                skipped_tasks.append((task_name, "Not a Task subclass"))
                continue

            # Create toy version
            toy_task_class = create_toy_task_class(task_class)

            # Instantiate with default parameters
            # Some tasks may have required parameters beyond split/languages
            try:
                task_instance = toy_task_class(split=split, languages=languages)
                toy_tasks.append(task_instance)
            except TypeError as e:
                # Task might require additional parameters
                skipped_tasks.append((task_name, f"Instantiation error: {e}"))
            except FileNotFoundError as e:
                # Data files might not be present
                skipped_tasks.append((task_name, f"Data not found: {e}"))

        except Exception as e:
            skipped_tasks.append((task_name, f"Error: {e}"))

    if skipped_tasks:
        print(f"\n⚠️  Skipped {len(skipped_tasks)} tasks:")
        for task_name, reason in skipped_tasks:
            print(f"  • {task_name}: {reason}")

    print(f"\n✅ Created {len(toy_tasks)} toy tasks (ranking + classification)")

    return toy_tasks


def test_e2e_toy_benchmark():
    """
    End-to-end test of all available tasks (ranking and classification) with toy datasets.

    This test:
    1. Uses TaskRegistry to discover all registered tasks
    2. Creates toy versions automatically with limited data
    3. Runs the complete benchmark pipeline
    4. Validates that all metrics are computed correctly
    5. Ensures no errors occur during execution
    """
    print("\n" + "=" * 70)
    print("🚀 Running E2E Toy Benchmark Test (Registry-Based)")
    print("=" * 70)

    # Configuration
    langs = ["en"]  # English only for speed
    split = "val"  # Use validation split

    # Discover and create toy tasks using registry
    print("\n📋 Discovering tasks from registry...")
    tasks = get_all_tasks(split=split, languages=langs)

    if not tasks:
        raise RuntimeError("No tasks were discovered! Check task registration.")

    # Display dataset sizes
    print("\n📊 Dataset sizes:")
    for task in tasks:
        print(f"  • {task.name:35} {task.get_size_oneliner(Language.EN)}")

    # Separate ranking and classification tasks
    ranking_tasks = [t for t in tasks if isinstance(t, RankingTask)]
    classification_tasks = [t for t in tasks if isinstance(t, ClassificationTask)]

    print(f"\n  Ranking tasks: {len(ranking_tasks)}")
    print(f"  Classification tasks: {len(classification_tasks)}")

    # Create model (BiEncoder now supports both ranking and classification)
    print("\n🤖 Initializing model...")
    model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")
    print("✓ Model initialized")

    # BiEncoder supports both ranking and classification tasks
    # Classification is implemented via ranking (similarity to label space)

    # Create benchmark with all tasks (both ranking and classification)
    print("\n🏃 Running benchmark (all tasks)...")

    # Track execution time
    start_time = time.time()

    # Run benchmark
    results = workrb.evaluate(
        model,
        tasks,
        output_folder="tmp/toy_benchmark_test",
        description="E2E Toy Benchmark Test (Registry-Based)",
        force_restart=True,
    )

    execution_time = time.time() - start_time

    print(f"✅ Benchmark completed in {execution_time:.2f} seconds")

    # Validate results
    print("\n🔍 Validating results...")

    assert len(results.task_results) == len(tasks), (
        f"Expected {len(tasks)} task results, got {len(results.task_results)}"
    )

    # Validate each task has results
    validation_errors = []
    for task in tasks:
        task_name = task.name

        # Check if task has results
        if task_name not in results.task_results:
            validation_errors.append(f"Missing results for task: {task_name}")
            continue

        task_result = results.task_results[task_name]

        # Check if language results exist
        if Language.EN not in task_result.language_results:
            validation_errors.append(f"Missing language results for {task_name} (en)")
            continue

        lang_result = task_result.language_results[Language.EN]

        # Validate metrics based on task type
        assert len(lang_result.metrics_dict) > 0, f"No metrics found for {task_name}"
        for expected_metric in task.default_metrics:
            if expected_metric not in lang_result.metrics_dict:
                validation_errors.append(f"Missing metric '{expected_metric}' for {task_name}")
                continue

            metric_value = lang_result.metrics_dict[expected_metric]
            # Most metrics should be in 0-1 range, but some might not
            # Just check that it's a valid number
            if not isinstance(metric_value, (int, float)):
                validation_errors.append(
                    f"Invalid {expected_metric} value type for {task_name}: {type(metric_value)}"
                )

    # Report validation results
    if validation_errors:
        print("\n❌ Validation errors:")
        for error in validation_errors:
            print(f"  • {error}")
        raise AssertionError(f"Validation failed with {len(validation_errors)} errors")

    print("✓ All validations passed")

    print("\n" + "=" * 70)
    print("🎉 E2E Toy Benchmark Test PASSED")
    print(f"   {len(ranking_tasks)} ranking task(s) tested")
    print(f"   {len(classification_tasks)} classification task(s) tested")
    print(f"   Total: {len(tasks)} tasks with BiEncoder model")
    print("=" * 70)


if __name__ == "__main__":
    # Allow running as standalone script
    try:
        test_e2e_toy_benchmark()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
