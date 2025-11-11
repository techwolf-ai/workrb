"""
Reproduce benchmark results.
"""

if __name__ == "__main__":
    # 1. Setup model and tasks
    models = [
        workrb.models.BiEncoderModel(),
        workrb.models.JobBERTModel(),
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
        workrb.tasks.ESCOJob2SkillRanking(split=split, languages=langs),
        workrb.tasks.ESCOSkill2JobRanking(split=split, languages=langs),
        workrb.tasks.ESCOSkillNormRanking(split=split, languages=langs),
        workrb.tasks.SkillMatch1kSkillSimilarityRanking(split=split, languages=langs),
        workrb.tasks.JobBERTJobNormRanking(split=split, languages=langs),
        workrb.tasks.HouseSkillExtractRanking(split=split, languages=langs),
        workrb.tasks.TechSkillExtractRanking(split=split, languages=langs),
        workrb.tasks.ESCOJob2SkillClassification(split=split, languages=langs),
    ]

    results = workrb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template="../results/simple_demo/{model_name}",
        description="Simple WorkRB demo",
        force_restart=True,
    )
