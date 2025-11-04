"""
Reproduce benchmark results.
"""

import wteb

if __name__ == "__main__":
    # 1. Setup model and tasks
    models = [
        wteb.models.BiEncoderModel(),
        wteb.models.JobBERTModel(),
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
        wteb.tasks.ESCOJob2SkillRanking(split=split, languages=langs),
        wteb.tasks.ESCOSkill2JobRanking(split=split, languages=langs),
        wteb.tasks.ESCOSkillNormRanking(split=split, languages=langs),
        wteb.tasks.SkillMatch1kSkillSimilarityRanking(split=split, languages=langs),
        wteb.tasks.JobBERTJobNormRanking(split=split, languages=langs),
        wteb.tasks.HouseSkillExtractRanking(split=split, languages=langs),
        wteb.tasks.TechSkillExtractRanking(split=split, languages=langs),
        wteb.tasks.ESCOJob2SkillClassification(split=split, languages=langs),
    ]

    # 2. Create and run benchmark
    benchmark = wteb.WTEB(tasks)

    results = benchmark.run_multiple_models(
        models=models,
        output_folder_template="../results/simple_demo/{model_name}",
        description="Simple WTEB demo",
    )
