"""
Simple WorkRB Example - Basic Usage

This minimal example demonstrates:
1. Setting up a model and task
2. Running a benchmark
3. Resuming from checkpoint (optional)
"""

if __name__ == "__main__":
    # 1. Setup model and tasks
    model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")
    tasks = [workrb.tasks.ESCOSkill2JobRanking(split="val", languages=[["en", "fr", "de", "nl"]])]

    # 2. Run the benchmark
    results = workrb.evaluate(
        model,
        tasks,
        output_folder="results/demo",
        description="WorkRB demo",
        force_restart=True,
    )
