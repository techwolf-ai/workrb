"""
Reproduce benchmark results.
"""

if __name__ == "__main__":
    # 1. Setup model and tasks
    models = [
        wb.models.BiEncoderModel(),
        wb.models.JobBERTModel(),
    ]

    # Cfg
    langs = [
        "en",
        "fr",
        "de",
        "nl",
    ]
    split = "test"

    tasks = [
        wb.tasks.ESCOJob2SkillRanking(split=split, languages=langs),
        wb.tasks.ESCOSkill2JobRanking(split=split, languages=langs),
        wb.tasks.ESCOSkillNormRanking(split=split, languages=langs),
        wb.tasks.SkillMatch1kSkillSimilarityRanking(split=split, languages=langs),
        wb.tasks.JobBERTJobNormRanking(split=split, languages=langs),
        wb.tasks.HouseSkillExtractRanking(split=split, languages=langs),
        wb.tasks.TechSkillExtractRanking(split=split, languages=langs),
        wb.tasks.ESCOJob2SkillClassification(split=split, languages=langs),
    ]

    # 2. Create and run benchmark
    benchmark = wb.WTEB(tasks)

    results = benchmark.run_multiple_models(
        models=models,
        output_folder_template="../results/simple_demo/{model_name}",
        description="Simple WTEB demo",
    )
