"""
Reproduce benchmark results.
"""

import workrb

if __name__ == "__main__":
    # 1. Setup model and tasks
    models = [
        workrb.models.RandomRankingModel(),
        workrb.models.TfIdfModel(tokenization="word"),
        workrb.models.TfIdfModel(lowercase=False, tokenization="word"),
        workrb.models.TfIdfModel(tokenization="char"),
        workrb.models.TfIdfModel(lowercase=False, tokenization="char"),
        workrb.models.BM25Model(),
        workrb.models.BM25Model(lowercase=False),
        workrb.models.EditDistanceModel(),
        workrb.models.EditDistanceModel(lowercase=False),
    ]

    # Config
    langs = [
        "en",
        "fr",
        "de",
        "es",
        "nl",
    ]
    split = "test"

    tasks = [
        workrb.tasks.JobTitleSimilarityRanking(split=split, languages=langs),
    ]

    results = workrb.evaluate_multiple_models(
        models=models,
        tasks=tasks,
        output_folder_template="../results/lexical_baselines/{model_name}",
        description="WorkRB demo with lexical baselines",
        force_restart=True,
    )
