"""
Reproduce benchmark results.
"""

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

    results = wteb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template="../results/simple_demo/{model_name}",
        description="Simple WTEB demo",
        force_restart=True,
    )
