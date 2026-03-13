<div align="center">

# WorkRB

<h3 style="border-bottom: none;">Easy benchmarking of AI progress in the work domain</h3>

[![syntax checking](https://github.com/techwolf-ai/workrb/actions/workflows/test.yml/badge.svg)](https://github.com/techwolf-ai/workrb/actions/workflows/test.yml)
[![GitHub release](https://img.shields.io/github/v/release/techwolf-ai/workrb.svg)](https://github.com/techwolf-ai/workrb/releases)
[![License](https://img.shields.io/github/license/techwolf-ai/workrb)](https://github.com/techwolf-ai/workrb/blob/main/LICENSE)


<h4>
    <p>
        <a href="#installation">Installation</a> |
        <a href="#features">Features</a> |
        <a href="#supported-tasks--models">Tasks & Models</a> |
        <a href="#usage-guide">Usage Guide</a> |
        <a href="#contributing">Contributing</a> |
        <a href="#citation">Citing</a>
    <p>
</h4>


<p style="margin-top:-10px; margin-bottom:-10px;">
    <img src="https://storage.googleapis.com/public-workrb/worker_bee_logo_transparent.png?raw=1" width="130px"/>
</p>


</div>

**WorkRB** (~pronounced worker bee) is an open-source evaluation toolbox for *benchmarking AI models in the work research domain*. 
It provides a standardized framework that is easy to use and community-driven, scaling evaluation over a wide range of tasks, ontologies, and models.

## Features

- 🐝 **One Buzzing Work Toolkit** — Easily download & access ontologies, datasets, and  baselines in a single toolkit
- 🧪 **Extensive tasks** — Evaluate models on job–skill matching, normalization, extraction, and similarity
- 🌍 **Dynamic Multilinguality** — Evaluate over languages driven by  multilingual ontologies
- 🧠 **Ready-to-go Baselines** — Leverage provided baseline models for comparison
- 🧩 **Extensible design** — Add your custom tasks and models with simple interfaces
<!-- - 📊 **Standardized evaluation** — Measure unified metrics over ranking and classification tasks -->
<!-- - 🔄 **Automatic checkpointing** — Resume interrupted or partial benchmarks seamlessly -->

## Example Usage

```python
import workrb

# 1. Initialize a model
model = workrb.models.BiEncoderModel("all-MiniLM-L6-v2")

# 2. Select (multilingual) tasks to evaluate
tasks = [
    workrb.tasks.ESCOJob2SkillRanking(split="val", languages=["en"]),
    workrb.tasks.ESCOSkillNormRanking(split="val", languages=["de", "fr"])
]

# 3. Run benchmark & view results
results = workrb.evaluate(  # Returns BenchmarkResults (Pydantic model)
    model,
    tasks,
    output_folder="results/my_model",
)
print(results)  # Benchmark/Per-task/Per-language metrics
```

## Installation
Install WorkRB simply via pip. 
```bash
pip install workrb
```
**Requirements:** Python 3.10+, see [pyproject.toml](pyproject.toml) for all dependencies.


## Supported tasks & models

### Tasks
| Task Name                      | Class Name | Label Type | Dataset Size (English)              | Languages |
|--------------------------------| --- | --- |-------------------------------------|-----------|
| **Ranking**
| Job to Skills WorkBench        | `ESCOJob2SkillRanking` | multi_label | 3039 queries x 13939 targets        | 28        |
| Job Title Similarity           | `JobTitleSimilarityRanking` | multi_label | 105 queries x 2619 targets          | 11        |
| Job Normalization              | `JobBERTJobNormRanking` | single_label | 15463 queries x 2942 targets        | 28        |
| Job Normalization MELO         | `MELORanking` | multi_label | 633 queries x 33813 targets         | 21        |
| Skill to Job WorkBench         | `ESCOSkill2JobRanking` | multi_label | 13492 queries x 3039 targets        | 28        |
| Skill Extraction House         | `HouseSkillExtractRanking` | multi_label | 262 queries x 13891 targets         | 28        |
| Skill Extraction Tech          | `TechSkillExtractRanking` | multi_label | 338 queries x 13891 targets         | 28        |
| Skill Extraction SkillSkape    | `SkillSkapeExtractRanking` | multi_label | 1191 queries x 13891 targets        | 28        |
| Skill Similarity SkillMatch-1K | `SkillMatch1kSkillSimilarityRanking` | single_label | 900 queries x 2648 targets          | 1         |
| Skill Normalization ESCO       | `ESCOSkillNormRanking` | multi_label | 72008 queries x 13939 targets       | 28        |
| Skill Normalization MELS       | `MELSRanking` | multi_label | 1722 queries x 19466 targets        | 5         |
| Query-Candidate Matching       | `SearchQueryCandidateRanking` | multi_label | 200 queries x 4019 (x-lang) targets | 5         |
| Project-Candidate Matching     | `ProjectCandidateRanking` | multi_label | 200 queries x 4019 (x-lang) targets | 5         |
| **Classification**
| Job-Skill Classification       | `ESCOJob2SkillClassification` | multi_label | 3039 samples, 13939 classes         | 28        |


### Models
| Model Name | Description | Adaptive Targets |
| --- | --- | --- |
| **Embedding Models**
| BiEncoderModel | BiEncoder model using sentence-transformers for ranking and classification tasks. | ✅ |
| JobBERTModel | Job-Normalization BiEncoder from Techwolf: https://huggingface.co/TechWolf/JobBERT-v2 | ✅ |
| ConTeXTMatchModel | ConTeXT-Skill-Extraction-base from Techwolf: https://huggingface.co/TechWolf/ConTeXT-Skill-Extraction-base | ✅ |
| CurriculumMatchModel | CurriculumMatch bi-encoder from Aleksandruz: https://huggingface.co/Aleksandruz/skillmatch-mpnet-curriculum-retriever | ✅ |
| **Lexical Baselines**
| BM25Model | BM25 Okapi probabilistic ranking baseline. | ✅ |
| TfIdfModel | TF-IDF baseline with configurable word-level or character n-gram tokenization. | ✅ |
| EditDistanceModel | Edit distance (Levenshtein ratio) baseline for near-exact matching. | ✅ |
| RandomRankingModel | Random scoring baseline for sanity checking evaluation pipelines. | ✅ |
| **Classification Baselines**
| RndESCOClassificationModel | Random baseline for multi-label classification with random prediction head for ESCO. | ❌ |

## Usage Guide

This section covers common usage patterns. Table of Contents:
- [Custom Tasks & Models](#custom-tasks--models)
- [Checkpointing & Resuming](#checkpointing--resuming)
- [Results & Aggregation](#results--aggregation)
- [Running Multiple Models](#running-multiple-models)


### Custom Tasks & Models

Add your custom task or model by (1) inheriting from a predefined base class and implementing the abstract methods, and (2) adding it to the registry: 
- **Custom Tasks**: Inherit from `RankingTask`, `MultilabelClassificationTask`,... Implement the abstract methods. Register via `@register_task()`.
- **Custom models**: Inherit from `ModelInterface`. Implement the abstract methods. Register via `@register_model()`.

```python
from workrb.tasks.abstract.ranking_base import RankingTask
from workrb.models.base import ModelInterface
from workrb.registry import register_task, register_model

@register_task()
class MyCustomTask(RankingTask):
    name: str = "MyCustomTask"
    ...


@register_model()
class MyCustomModel(ModelInterface):
    name: str = "MyCustomModel"
    ...

# Use your custom model and task:
model_results = workrb.evaluate(MyCustomModel(),[MyCustomTask()])
```

**For detailed examples**, see:
- [examples/custom_task_example.py](examples/custom_task_example.py) for a complete custom task implementation
- [examples/custom_model_example.py](examples/custom_model_example.py) for a complete custom model implementation

Feel free to make a PR to add your models & tasks to the official package! See [CONTRIBUTING guidelines](CONTRIBUTING.md) for details.

### Checkpointing & Resuming

WorkRB automatically saves result checkpoints after each dataset evaluation within a task.

**Automatic Resuming** - Simply rerun with the same `output_folder`:

```python
# Run 1: Gets interrupted after 2 tasks
tasks = [
    workrb.tasks.ESCOJob2SkillRanking(
        split="val", 
        languages=["en"],
    )
]

results = workrb.evaluate(model, tasks, output_folder="results/my_model")

# Run 2: Automatically resumes from checkpoint
results = workrb.evaluate(model, tasks, output_folder="results/my_model")
# ✓ Skips completed tasks, continues from where it left off
```
**Extending Benchmarks** - Want to extend your results with additional tasks or languages? Add the new tasks or languages when resuming:

```python
# Resume from previous & extend with new task and languages
tasks_extended = [
    workrb.tasks.ESCOJob2SkillRanking(  # Add de, fr
        split="val",
        languages=["en", "de", "fr"],
    ),
    workrb.tasks.ESCOSkillNormRanking(  # Add new task
        split="val",
        languages=["en"],
    ),
]
results = workrb.evaluate(model, tasks_extended, output_folder="results/my_model")
# ✓ Reuses English results, only evaluates new languages/tasks
```

❌**You cannot reduce scope** when resuming. This is by design to avoid ambiguity. Finished tasks in the checkpoint should also be included in your WorkRB initialization. If you want to start fresh in the same output folder, use `force_restart=True`:
```python
results = workrb.evaluate(model, tasks, output_folder="results/my_model", force_restart=True)
```


### Results & Metric Aggregation

**Results** are automatically saved to your `output_folder`:

```
results/my_model/
├── checkpoint.json       # Incremental checkpoint (for resuming)
├── results.json          # Final results dump
└── config.yaml           # Final benchmark configuration dump
```

To load & parse results from a run:

```python
results = workrb.load_results("results/my_model/results.json")
print(results)
```

#### Aggregation Chain

The final benchmark score `mean_benchmark/<metric>/mean` is computed via the following chain:
`dataset`  → `language`  → `task` → `task_group` → `task_type`.
This enables sequential macro-averaging in each of the stages:
- `dataset`: Is the individual unit to start aggregation from. Each task contains a set of datasets, each with a unique `dataset_id`. *Example: The MELO task language/region subsets `usa_q_en_c_en` and `swe_q_sv_c_en`.*
- `language`: Aggregate over languages within the task's datasets. *Example: Group all monolingual French datasets in ESCOSkill2JobRanking*
- `task`: Aggregate over tasks in the same task group. *Example: HouseSkillExtractRanking and TechSkillExtractRanking tasks in the Skill Extraction task group.*
- `task_group`: Aggregate over task groups under a specific task type. *Example: Skill Extraction, Skill Normalization, and Job Normalization task groups, are all part of the ranking task type*
- `task_type`: Aggregate over different task types for final benchmark performance, e.g. the Ranking and Classification task types.

Per-language performance is available under language-grouped modes: `mean_per_language/<lang>/<metric>/mean`.
Each aggregation provides 95% confidence intervals (replace `mean` with `ci_margin`).

#### Available Metrics

**Ranking metrics** (used in `RankingTask`):

| Metric | Description |
| --- | --- |
| `map` | Mean Average Precision |
| `mrr` | Mean Reciprocal Rank |
| `ndcg@k` | Normalized Discounted Cumulative Gain with support for top-k cutoff.|
| `recall@k` | Recall at k (e.g. `recall@5`, `recall@10`) |
| `hit@k` | Hit rate at k — binary: is any relevant item in the top-k? |
| `rp@k` | R-Precision at k — precision relative to total relevant items |

**Classification metrics** (used in `ClassificationTask`):

| Metric | Description |
| --- | --- |
| `f1_macro`, `f1_micro`, `f1_weighted` | F1 score variants |
| `f1_samples` | Per-sample F1 (multilabel only) |
| `precision_macro`, `precision_micro`, `precision_weighted` | Precision variants |
| `recall_macro`, `recall_micro`, `recall_weighted` | Recall variants |
| `accuracy` | Overall accuracy (subset accuracy for multilabel) |
| `roc_auc`, `roc_auc_micro` | Area under ROC curve (threshold-independent) |

You can override the default metrics per task via the `metrics` parameter of `evaluate()`:

```python
results = workrb.evaluate(
    model, tasks, output_folder="results/my_model",
    metrics={"ESCOJob2SkillRanking": ["map", "mrr", "recall@5", "recall@10"]},
)
```

#### Language Aggregation Modes

The `language_aggregation_mode` parameter controls how dataset results are grouped during metric aggregation. There are 4 modes (`LanguageAggregationMode`):

| Mode | Behavior |
| --- | --- |
| `MONOLINGUAL_ONLY` (default) | Group by language; only include monolingual datasets (input lang == output lang). Cross-lingual datasets are filtered out. |
| `CROSSLINGUAL_GROUP_INPUT_LANGUAGES` | Group by the input/query language. Requires a single input language per dataset (skip multilingual inputs). |
| `CROSSLINGUAL_GROUP_OUTPUT_LANGUAGES` | Group by the output/target language. Requires a single output language per dataset (skip multilingual outputs). |
| `SKIP_LANGUAGE_AGGREGATION` | No language grouping or filtering. All datasets are directly macro-averaged per task. No per-language metrics are produced. |



#### Execution Mode
As `datasets` may be filtered out by the aggregation mode, you may want to skip evaluations that are not used in the final metrics. The `execution_mode` parameter controls whether incompatible datasets are evaluated:

| Mode | Behavior |
| --- | --- |
| `ExecutionMode.LAZY` (default) | Skip datasets that are incompatible with the chosen `language_aggregation_mode`, saving compute. |
| `ExecutionMode.ALL` | Evaluate all datasets regardless. Useful when you want to store all results and re-aggregate later with a different mode. |

> **Note:** Under `SKIP_LANGUAGE_AGGREGATION`, no datasets are ever incompatible, so `execution_mode` can be ignored.

An example of why you could choose for `ExecutionMode.ALL`:

```python
from workrb.types import LanguageAggregationMode, ExecutionMode

# Benchmark returns a detailed Pydantic model
results: BenchmarkResults = workrb.evaluate(
    model,
    tasks,
    language_aggregation_mode=LanguageAggregationMode.MONOLINGUAL_ONLY,
    execution_mode=ExecutionMode.ALL, # Execute all so we can later switch language_aggregation_mode
)
# No lazy mode was used; We can override the aggregation mode at summary time
summary = results.get_summary_metrics(
    language_aggregation_mode=LanguageAggregationMode.CROSSLINGUAL_GROUP_INPUT_LANGUAGES,
)
```

For a complete runnable example of different aggregation strategies, see [examples/run_benchmark_aggregation.py](examples/run_benchmark_aggregation.py).


### Running Multiple Models

Evaluate multiple models in a single call with `evaluate_multiple_models()`. Each model runs the `evaluate()` and automatically gets its own output folder (with checkpointing) based on the model name:

```python
results = workrb.evaluate_multiple_models(
    models=[model_a, model_b],
    tasks=tasks,
    output_folder_template="results/{model_name}", # Use explicit '{model_name}', used for templating
)
# results["model_a_name"] -> BenchmarkResults
# results["model_b_name"] -> BenchmarkResults
```

All keyword arguments from `evaluate()` (e.g. `language_aggregation_mode`, `execution_mode`) are passed through. See [examples/run_multiple_models.py](examples/run_multiple_models.py) for a complete example.



## Contributing
Want to contribute new tasks, models, or metrics?
Read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for all details.

### Development environment

```sh
# Clone repository
git clone https://github.com/techwolf-ai/workrb.git && cd workrb

# Create and install a virtual environment
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate

# Install the pre-commit hooks
pre-commit install --install-hooks

# Run tests (excludes model benchmarking by default)
uv run poe test

# Run model benchmark tests only, checks reproducibility of original results
uv run poe test-benchmark
```


<details>
<summary>Developing details</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `uv add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `uv.lock`. Add `--dev` to install a development dependency.
- Run `uv sync --upgrade` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`. Add `--only-dev` to upgrade the development dependencies only.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag. Then push the changes and the git tag with `git push origin main --tags`.

</details>


## Citation


<details>
<summary>WorkRB builds upon the unifying WorkBench benchmark, consider citing:</summary>

```bibtex
@misc{delange2025unifiedworkembeddings,
      title={Unified Work Embeddings: Contrastive Learning of a Bidirectional Multi-task Ranker}, 
      author={Matthias De Lange and Jens-Joris Decorte and Jeroen Van Hautte},
      year={2025},
      eprint={2511.07969},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07969}, 
}
```
</details>

<details>
<summary>WorkRB has a community paper coming up!</summary>
WIP
</details>

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
