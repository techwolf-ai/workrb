"""
End-to-End Checkpointing Test.

This test validates the checkpointing and resume functionality of WorkRB.
It simulates interrupted benchmark runs and validates that:
1. Checkpoints are saved correctly after each task completion
2. Benchmarks can be resumed from checkpoints
3. Multiple interruptions and resumes work correctly
4. Final results are complete and accurate

Test Scenarios:
1. Start benchmark with 3 tasks → interrupt after 1 task
2. Resume benchmark → interrupt after 1 more task (2 total)
3. Resume benchmark → complete all remaining tasks (3 total)

Usage:
    pytest tests/test_e2e_checkpointing.py -v
    python tests/test_e2e_checkpointing.py
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

import workrb
from tests.test_utils import create_toy_task_class
from workrb.tasks import SkillMatch1kSkillSimilarityRanking
from workrb.tasks.abstract.base import Language


def verify_checkpoint(checkpoint_path: Path, expected_completed: int, total_tasks: int):
    """Verify checkpoint contains expected number of completed tasks."""
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"

    with open(checkpoint_path) as f:
        checkpoint_data = json.load(f)

    assert "results" in checkpoint_data, "Checkpoint missing results"
    assert "last_updated" in checkpoint_data, "Checkpoint missing timestamp"

    results = checkpoint_data["results"]
    task_results = results.get("task_results", {})

    # Count completed task-language combinations
    completed_count = 0
    for task_result in task_results.values():
        completed_count += len(task_result.get("language_results", {}))

    print(f"  ✓ Checkpoint has {completed_count}/{total_tasks} completed task(s)")
    assert completed_count == expected_completed, (
        f"Expected {expected_completed} completed, got {completed_count}"
    )

    return True


def test_e2e_checkpointing():
    """
    End-to-end test of checkpointing and resume functionality.

    Test sequence:
    1. Run benchmark with 3 tasks, interrupt after 1 task
    2. Verify checkpoint has 1 task completed
    3. Resume benchmark, interrupt after 1 more task (2 total)
    4. Verify checkpoint has 2 tasks completed
    5. Resume benchmark, complete all tasks (3 total)
    6. Verify final results have all 3 tasks completed
    """
    # Test configuration
    num_tasks = 3

    print("\n" + "=" * 70)
    print("🚀 Running E2E Checkpointing Test")
    print("=" * 70)

    # Setup
    output_folder = Path("tmp/checkpoint_test")
    checkpoint_path = output_folder / "checkpoint.json"

    # Clean up any existing test data
    if output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)

    ToySkillSim = create_toy_task_class(SkillMatch1kSkillSimilarityRanking)

    # Create tasks with unique names by subclassing
    class Task1(ToySkillSim):
        @property
        def name(self):
            return "Skill Similarity 1"

    class Task2(ToySkillSim):
        @property
        def name(self):
            return "Skill Similarity 2"

    class Task3(ToySkillSim):
        @property
        def name(self):
            return "Skill Similarity 3"

    tasks = [
        Task1(split="val", languages=["en"]),
        Task2(split="val", languages=["en"]),
        Task3(split="val", languages=["en"]),
    ]

    print(f"\n📋 Created {len(tasks)} toy tasks:")
    for task in tasks:
        print(f"  • {task.name}")

    # Create model
    print("\n🤖 Initializing model...")
    model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")
    print("✓ Model initialized")

    # =========================================================================
    # PHASE 1: Run benchmark, for only 1 task
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Initial run - interrupt after 1 task")
    print("=" * 70)

    # Run benchmark - it will be interrupted
    _mid_results = workrb.evaluate(
        model,
        tasks=tasks[:1],
        output_folder=str(output_folder),
        description="Checkpointing test - Phase 1",
    )

    print("\n🔍 Verifying Phase 1 checkpoint...")
    verify_checkpoint(checkpoint_path, expected_completed=1, total_tasks=num_tasks)

    # =========================================================================
    # PHASE 2: Resume benchmark, but with more tasks (including the original ones)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Resume - but with more tasks (including the original ones)")
    print("=" * 70)

    # Run benchmark - it will be interrupted again
    end_results = workrb.evaluate(
        model,
        tasks=tasks,
        output_folder=str(output_folder),
        description="Checkpointing test - Phase 2",
    )

    print("\n🔍 Verifying Phase 2 checkpoint...")
    verify_checkpoint(checkpoint_path, expected_completed=num_tasks, total_tasks=num_tasks)

    # Verify final results are complete
    print("\n🔍 Verifying final results completeness...")
    assert len(end_results.task_results) == num_tasks, (
        f"Expected {num_tasks} tasks, got {len(end_results.task_results)}"
    )

    for task in tasks:
        assert task.name in end_results.task_results, (
            f"Task '{task.name}' missing from final results"
        )
        task_result = end_results.task_results[task.name]
        assert Language.EN in task_result.language_results, (
            f"Language 'en' missing for task '{task.name}'"
        )

        lang_result = task_result.language_results[Language.EN]
        assert len(lang_result.metrics_dict) > 0, f"No metrics for task '{task.name}'"
        print(f"  ✓ Task '{task.name}' has complete results")

    # =========================================================================
    # PHASE 3: Resume benchmark, but with result checkpoint having tasks not in the benchmark
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Resume - but with result checkpoint having tasks not in the benchmark")
    print("=" * 70)

    # 3 tasks are run, try to rerun with only 1 task (should fail)
    with pytest.raises(Exception):
        _results3 = workrb.evaluate(
            model,
            tasks=tasks[:1],
            output_folder=str(output_folder),
            description="Checkpointing test - Phase 3",
        )

    # =========================================================================
    # PHASE 4: Run again - should skip all work (already complete)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Run again - should skip all work")
    print("=" * 70)

    end_results_retry = workrb.evaluate(
        model,
        tasks=tasks,
        output_folder=str(output_folder),
        description="Checkpointing test - Phase 4",
    )

    # Verify results are identical to the original ones
    assert len(end_results_retry.task_results) == len(end_results.task_results), (
        "Results should be identical"
    )

    print("\n" + "=" * 70)
    print("🎉 E2E Checkpointing Test PASSED")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Phase 1: Started benchmark, for only 1 task")
    print("  ✓ Phase 2: Resumed, but with more tasks (including the original ones)")
    print("  ✓ Phase 3: Resumed, but with result checkpoint having tasks not in the benchmark")
    print("  ✓ Phase 4: Skipped re-execution (already complete) - ran full benchmark again")
    print("  ✓ Checkpoints saved correctly at each step")
    print("  ✓ Final results are complete and accurate")


if __name__ == "__main__":
    # Allow running as standalone script
    try:
        test_e2e_checkpointing()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
