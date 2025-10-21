<div align="center">

# 🛠️ WorkBench

<h2>Benchmarking AI progress in the work domain, made easy</h2>

[![syntax checking](https://github.com/techwolf-ai/workbench/actions/workflows/test.yml/badge.svg)](https://github.com/techwolf-ai/workbench/actions/workflows/test.yml)
[![GitHub release](https://img.shields.io/github/release/techwolf-ai/workbench-toolkit.svg)](https://github.com/techwolf-ai/workbench/releases)
[![License](https://img.shields.io/github/license/techwolf-ai/workbench.svg?color=green)](https://github.com/techwolf-ai/workbench/blob/main/LICENSE)

<h4>
    <p>
        <a href="#installation">Installation</a> |
        <a href="#quick-start">Quick Start</a> |
        <a href="#usage-guide">Usage Guide</a> |
        <a href="#contributing--development">Contributing</a>
    <p>
</h4>

</div>

**WorkBench** is an open-source library to *benchmark AI systems in the work domain*. 

It provides a standardized framework to evaluate the performance of state-of-the-art models with ease.


with the unique goal of  mission to standardize progress in the work domain is measured by providing clarity and transparency with an easy-to-use evaluation framework.

**Design Principles:**
- **Ease of Use** – Quick setup, clean APIs, minimal boilerplate
- **Transparency** – Clear metrics and datasets, even in complex evaluation settings
- **Community-Driven** – New tasks, models, and metrics evolve from the open-source community


## Example Usage

```python
import workbench as wb

# 1. Initialize a model
model = wb.models.BiEncoderModel("all-MiniLM-L6-v2")

# 2. Select (multilingual) tasks to evaluate
tasks = [
    wb.tasks.ESCOJob2SkillRanking(split="val", languages=["en"]),
    wb.tasks.ESCOSkillNormRanking(split="val", languages=["de", "fr"])
]

# 3. Run benchmark & view results
benchmark = wb.WorkBench(tasks)
results = benchmark.run(
    model,
    output_folder="results/my_model",
)
print(results)
```

## Installation
*Note: PyPI installation is WIP, for now follow the [dev setup]().*

Install WorkBench simply via pip. 
```bash
pip install workbench-ai
```
**Requirements:** Python 3.10+, see [pyproject.toml](pyproject.toml) for all dependencies.

## Features

- **7+ Benchmark Tasks** – Evaluate models on job-skill matching, normalization, extraction, and similarity
- **Multilingual Support** – Test across 27+ European languages via ESCO datasets
- **Standardized Metrics** – MAP, MRR, Recall@K, Precision@K for ranking; F1, accuracy for classification
- **Automatic Checkpointing** – Resume interrupted or partial benchmarks seamlessly
- **Extensible Design** – Add custom tasks and models with simple interfaces

## Usage Guide

This section covers common usage patterns. For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Table of Contents:**
- [Custom Tasks & Models](#custom-tasks--models)
- [Discovering Available Tasks & Models](#discovering-available-tasks--models)
- [Checkpointing & Resuming](#checkpointing--resuming)
- [Results & Aggregation](#results--aggregation)


### Custom Tasks & Models

Add your custom task or model by (1) inheriting from a predefined base class and implementing the abstract methods, and (2) adding it to the registry: 
- **Custom Tasks**: Inherit from `RankingTask`, `MultilabelClassificationTask`,... Implement the abstract methods. Register via `@register_task()`.
- **Custom models**: Inherit from `ModelInterface`. Implement the abstract methods. Register via `@register_model()`.

```python
from workbench.tasks.abstract.ranking_base import RankingTask
from workbench.models.base import ModelInterface
from workbench.registry import register_task, register_model

@register_task()
class MyCustomTask(RankingTask):
    name: str = "MyCustomTask"
    ...


@register_model()
class MyCustomModel(ModelInterface):
    name: str = "MyCustomModel"
    ...

# Use your custom model and task:
benchmark = wb.WorkBench(tasks=[MyCustomTask()])
model_results = benchmark.run(MyCustomModel())
```

**For detailed examples**, see:
- [examples/custom_task_example.py](examples/custom_task_example.py) for a complete custom task implementation
- [CONTRIBUTING.md](CONTRIBUTING.md) for step-by-step guides

---

### Discovering Available Tasks & Models

List all registered tasks and models:

```python
from workbench.registry import TaskRegistry, ModelRegistry

# List all available tasks and models
available_tasks = TaskRegistry.list_available()
available_models = ModelRegistry.list_available()

# Create by name
task = TaskRegistry.create("ESCOJob2SkillRanking", split="val", languages=["en"])
```

---

### Checkpointing & Resuming

WorkBench automatically saves result checkpoints after each task completion in a specific language.

**Automatic Resuming** - Simply rerun with the same `output_folder`:

```python
# Run 1: Gets interrupted after 2 tasks
tasks = [
    wb.tasks.ESCOJob2SkillRanking(
        split="val", 
        languages=["en"],
    )
]
benchmark = wb.WorkBench(tasks)
results = benchmark.run(model, output_folder="results/my_model")

# Run 2: Automatically resumes from checkpoint
results = benchmark.run(model, output_folder="results/my_model")
# ✓ Skips completed tasks, continues from where it left off
```
**Extending Benchmarks** - Want to extend your results with additional tasks or languages? Add the new tasks or languages when resuming:

```python
# Resume from previous & extend with new task and languages
tasks_extended = [
    wb.tasks.ESCOJob2SkillRanking( # Add de, fr
        split="val", 
        languages=["en", "de", "fr"]), 
    wb.tasks.ESCOJob2SkillRanking( # Add new task
        split="val", 
        languages=["en"],
]
benchmark = wb.WorkBench(tasks_extended)
results = benchmark.run(model, output_folder="results/my_model")
# ✓ Reuses English results, only evaluates new languages/tasks
```

❌**You cannot reduce scope** when resuming, by design to avoid ambiguity. Finished tasks in the checkpoint should also be included in your WorkBench initialization. If you want to start fresh in the same output folder, use `force_restart=True`:
```python
results = benchmark.run(model, output_folder="results/my_model", force_restart=True)
```

---

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
results = wb.WorkBench.load_results("results/my_model/results.json")
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
results: BenchmarkResults = benchmark.run(...)

# Calculate aggregated metrics
summary: dict[str, float] = results.get_summary_metrics()

# Show all results
print(summary)
print(results) # Equivalently, internally runs get_summary_metrics()

# Access metric via tag
lang_result = summary["mean_per_language/en/f1_macro/mean"]
```

---


## Supported tasks & models

| Task | Type | Description |
|------|------|-------------|
| **ESCOJob2SkillRanking** | Ranking | Map job titles to relevant skills |
| **ESCOSkill2JobRanking** | Ranking | Map skills to relevant job titles |
| **TechSkillExtractRanking** | Ranking | Extract skills from technical text |
| **HouseSkillExtractRanking** | Ranking | Extract skills from general text |
| **JobBERTJobNormRanking** | Ranking | Normalize job titles |
| **ESCOSkillNormRanking** | Ranking | Normalize skills to ESCO taxonomy |
| **SkillMatch1kSkillSimilarityRanking** | Ranking | Find similar skills |
| **ESCOJob2SkillClassification** | Classification | Classify jobs to skill categories |

---


## Contributing
Read our detailed [CONTRIBUTING.md](CONTRIBUTING.md) guide for more details.

### Development setup
Clone this repository and run the following from root of the repository:

```sh
# Clone repository
git clone https://github.com/techwolf-ai/workbench.git && cd workbench

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

If you use WorkBench in your research, please cite:

```bibtex
@misc{workbench2025,
  title={WorkBench: A Community-Driven Benchmark for Work Domain AI Evaluation},
  author={TechWolf Research Team},
  year={2025},
  url={https://github.com/techwolf-ai/workbench-toolkit}
}
``` -->

## License

MIT License - see [LICENSE](LICENSE) for details.
