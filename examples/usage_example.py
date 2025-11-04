"""
Simple WTEB Example - Basic Usage

This minimal example demonstrates:
1. Setting up a model and task
2. Running a benchmark
3. Resuming from checkpoint (optional)
"""

if __name__ == "__main__":
    # 1. Setup model and tasks
    model = wb.models.BiEncoderModel("all-MiniLM-L6-v2")
    tasks = [wb.tasks.ESCOSkill2JobRanking(split="val", languages=[["en", "fr", "de", "nl"]])]

    # 2. Create and run benchmark
    benchmark = wb.WTEB(tasks)

    results = benchmark.run(
        model,
        output_folder="results/demo",
        description="WTEB demo",
        force_restart=True,
    )
