<div align="center">

# 🛠️ WTEB

<h3 style="border-bottom: none;">Easy benchmarking of AI progress in the work domain</h3>

[![syntax checking](https://github.com/techwolf-ai/wteb/actions/workflows/test.yml/badge.svg)](https://github.com/techwolf-ai/wteb/actions/workflows/test.yml)
[![GitHub release](https://img.shields.io/github/release/techwolf-ai/wteb-toolkit.svg)](https://github.com/techwolf-ai/wteb/releases)
[![License](https://img.shields.io/github/license/techwolf-ai/wteb.svg?color=green)](https://github.com/techwolf-ai/wteb/blob/main/LICENSE)

<h4>
    <p>
        <a href="#installation">Installation</a> |
        <a href="#features">Features</a> |
        <a href="#usage-guide">Usage Guide</a> |
        <a href="#contributing">Contributing</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

</div>

**WTEB** is an open-source library to *benchmark embedding models in the work domain*. 
It provides a standardized framework that is easy to use and community-driven, scaling evaluation over a wide range of state-of-the-art tasks and models.

## Features

- 🧪 **7+ benchmark tasks** — Evaluate models on job–skill matching, normalization, extraction, and similarity
- 🌍 **Dynamic Multilinguality** — Test tasks dynamically across 27+ EU languages via ESCO ontologies
- 🧩 **Extensible design** — Add your custom tasks and models with simple interfaces
- 📊 **Standardized metrics** — Measure unified metrics over ranking and classification tasks
- 🔄 **Automatic checkpointing** — Resume interrupted or partial benchmarks seamlessly

## Example Usage

```python
import wteb

# 1. Initialize a model
model = wteb.models.BiEncoderModel("all-MiniLM-L6-v2")

# 2. Select (multilingual) tasks to evaluate
tasks = [
    wteb.tasks.ESCOJob2SkillRanking(split="val", languages=["en"]),
    wteb.tasks.ESCOSkillNormRanking(split="val", languages=["de", "fr"])
]

# 3. Run benchmark & view results
results = wteb.evaluate(
    model,
    tasks,
    output_folder="results/my_model",
)
print(results)
```

## Installation
*Note: PyPI installation is WIP, for now follow the [dev setup]().*

Install WTEB simply via pip. 
```bash
pip install wteb
```
**Requirements:** Python 3.10+, see [pyproject.toml](pyproject.toml) for all dependencies.

## Usage Guide

This section covers common usage patterns. Table of Contents:
- [Custom Tasks & Models](#custom-tasks--models)
- [Checkpointing & Resuming](#checkpointing--resuming)
- [Results & Aggregation](#results--aggregation)


### Custom Tasks & Models

Add your custom task or model by (1) inheriting from a predefined base class and implementing the abstract methods, and (2) adding it to the registry: 
- **Custom Tasks**: Inherit from `RankingTask`, `MultilabelClassificationTask`,... Implement the abstract methods. Register via `@register_task()`.
- **Custom models**: Inherit from `ModelInterface`. Implement the abstract methods. Register via `@register_model()`.

```python
from wteb.tasks.abstract.ranking_base import RankingTask
from wteb.models.base import ModelInterface
from wteb.registry import register_task, register_model

@register_task()
class MyCustomTask(RankingTask):
    name: str = "MyCustomTask"
    ...


@register_model()
class MyCustomModel(ModelInterface):
    name: str = "MyCustomModel"
    ...

# Use your custom model and task:
model_results = wteb.evaluate(MyCustomModel(),[MyCustomTask()])
```

**For detailed examples**, see:
- [examples/custom_task_example.py](examples/custom_task_example.py) for a complete custom task implementation
- [examples/custom_model_example.py](examples/custom_model_example.py) for a complete custom model implementation

Feel free to make a PR to add your models & tasks to the official package! See [CONTRIBUTING guidelines](CONTRIBUTING.md) for details.

### Checkpointing & Resuming

WTEB automatically saves result checkpoints after each task completion in a specific language.

**Automatic Resuming** - Simply rerun with the same `output_folder`:

```python
# Run 1: Gets interrupted after 2 tasks
tasks = [
    wteb.tasks.ESCOJob2SkillRanking(
        split="val", 
        languages=["en"],
    )
]

results = wteb.evaluate(model, tasks, output_folder="results/my_model")

# Run 2: Automatically resumes from checkpoint
results = wteb.evaluate(model, tasks, output_folder="results/my_model")
# ✓ Skips completed tasks, continues from where it left off
```
**Extending Benchmarks** - Want to extend your results with additional tasks or languages? Add the new tasks or languages when resuming:

```python
# Resume from previous & extend with new task and languages
tasks_extended = [
    wteb.tasks.ESCOJob2SkillRanking( # Add de, fr
        split="val", 
        languages=["en", "de", "fr"]), 
    wteb.tasks.ESCOJob2SkillRanking( # Add new task
        split="val", 
        languages=["en"],
]
results = wteb.evaluate(model, tasks, output_folder="results/my_model")
# ✓ Reuses English results, only evaluates new languages/tasks
```

❌**You cannot reduce scope** when resuming. This is by design to avoid ambiguity. Finished tasks in the checkpoint should also be included in your WTEB initialization. If you want to start fresh in the same output folder, use `force_restart=True`:
```python
results = wteb.evaluate(model, tasks, output_folder="results/my_model", force_restart=True)
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
results = wteb.load_results("results/my_model/results.json")
print(results)
```

**Metrics**: The main benchmark metrics `mean_benchmark/<metric>/mean` require 4 aggregation steps:

1. First, Macro-average languages per task (e.g. ESCOJob2SkillRanking)   (`mean_per_task/<task_name>/<metric>/mean`)
2. Macro-average tasks per task group (e.g. Job2SkillRanking)  (`mean_per_task_group/<group>/<metric>/mean`)
3. Macro-average task groups per task type (e.g. RankingTask, ClassificationTask) `mean_per_task_type/<type>/<metric>/mean`
4. Macro-average over task types.

Per-language performance is also available: `mean_per_language/<lang>/<metric>/mean`.
Each aggregation provides 95% confidence intervals (replace `mean` with `ci_margin`) 

```python
# Benchmark returns a detailed Pydantic model
results: BenchmarkResults = wteb.evaluate(...)

# Calculate aggregated metrics
summary: dict[str, float] = results.get_summary_metrics()

# Show all results
print(summary)
print(results) # Equivalent: internally runs get_summary_metrics()

# Access metric via tag
lang_result = summary["mean_per_language/en/f1_macro/mean"]
lang_result_ci = summary["mean_per_language/en/f1_macro/ci_margin"]
```


## Supported tasks & models

### Tasks
| Task Name | Label Type | Dataset Size (English) | Languages |
| --- | --- | --- | --- |
| **Ranking** 
| Job to Skills | multi_label | 3039 queries x 13939 targets | 28  |
| Job Normalization | multi_class | 15463 queries x 2942 targets | 28  |
| Skill to Job | multi_label | 13492 queries x 3039 targets | 28  |
| Skill Extraction House | multi_label | 262 queries x 13891 targets | 28  |
| Skill Extraction Tech | multi_label | 338 queries x 13891 targets | 28  |
| Skill Similarity | multi_class | 900 queries x 2648 targets | 1 |
| ESCO Skill Normalization | multi_label | 72008 queries x 13939 targets | 28  |
| **Classification**
| Job-Skill Classification | multi_label | 3039 samples, 13939 classes | 28  |


### Models
| Model Name | Description | Fixed Classifier |
| --- | --- | --- |
| BiEncoderModel | BiEncoder model using sentence-transformers for ranking and classification tasks. | ❌ |
| JobBERTModel | Job-Normalization BiEncoder from Techwolf: https://huggingface.co/TechWolf/JobBERT-v2 | ❌ |
| RndESCOClassificationModel | Random baseline for multi-label classification with random prediction head for ESCO. | ✅ |



## Contributing
Want to contribute new tasks, models, or metrics?
Read our [CONTRIBUTING.md](CONTRIBUTING.md) guide for all details.

### Development environment

```sh
# Clone repository
git clone https://github.com/techwolf-ai/wteb.git && cd wteb

# Create and install a virtual environment
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate

# Install the pre-commit hooks
pre-commit install --install-hooks
```


<details>
<summary>Developing details</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `uv add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `uv.lock`. Add `--dev` to install a development dependency.
- Run `uv sync --upgrade` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`. Add `--only-dev` to upgrade the development dependencies only.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag. Then push the changes and the git tag with `git push origin main --tags`.

</details>


<!-- ## Citation

If you use WTEB in your research, please cite:

```bibtex
UWE-PLACEHOLDER
``` -->

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
