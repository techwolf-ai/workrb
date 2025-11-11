# Contributing to WorkRB

Thank you for your interest in contributing to WorkRB! We're building a community-driven benchmark for work domain AI evaluation, and your contributions help make it better for everyone.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Adding a New Task](#adding-a-new-task)
- [Adding a New Model](#adding-a-new-model)
- [Adding New Metrics](#adding-new-metrics)
- [Code Standards](#code-standards)
- [Questions & Support](#questions--support)

## Ways to Contribute

We welcome contributions of all kinds:

- **🐛 Report bugs** – Found an issue? Let us know in [GitHub Issues](https://github.com/techwolf-ai/workrb/issues)
- **📊 Add new tasks** – Extend WorkRB with new evaluation tasks
- **🤖 Add new models** – Implement state-of-the-art models or baselines
- **📈 Add new metrics** – Contribute evaluation metrics relevant to the work domain
- **📚 Improve documentation** – Help make WorkRB easier to use
- **✨ Suggest features** – Share ideas for improvements

## Development Setup

### Prerequisites

- [install uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git

### Setup Instructions

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/workrb.git
   cd workrb
   ```

2. **Install dependencies:**
   ```bash
    # Create and install a virtual environment, including dev
    uv sync --all-extras

    # Activate the virtual environment (venv)
    source .venv/bin/activate

    # Install the pre-commit hooks in the venv
    pre-commit install --install-hooks
   ```

3. **Verify installation:**
   ```bash
   # Run example script
   uv run python examples/usage_example.py
   ```

4. **Create a new branch for your changes:**
   ```bash
   git checkout -b feature/my-new-feature
   ```


## Contributing Process

### 1. Make an Issue with your proposal

Before starting any significant work (new feature, task, model, or refactor), please open a proposal issue first. This helps us align on scope and approach before you invest time in an implementation.

- Open an issue at `https://github.com/techwolf-ai/workrb/issues` describing your proposal. Select the 'Feature Request' template to provide additional context.
- Maintainers will triage and respond in the issue with feedback and next steps
- Once there’s agreement on the direction, proceed to the implementation 
with a Pull Request referencing the issue

### 2. Start implementing (local)
Project: 
1. Make a fork of the main branch into your own repo
2. Implement your code. See further in this guide how to add new tasks, models, or metrics.
3. Ensure all linting and tests complete successfully locally before creating a PR:
   ```bash
   uv run poe lint
   uv run pytest test/my_task_tests.py # Just your tests
   uv run poe test # (ideally) For all tests
   ```
4. Having questions? Add them to your Github Issue. 

### 3. Submit Your PR
Make a pull request (PR) from your fork into the main branch of WorkRB, following:

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Open a Pull Request** to `main` branch on WorkRB's GitHub with: 
    - A clear title describing the change
    - Link to the issue by using hashtag identifier (e.g. #123 will refer to issue 123)
    - Filling in the following template:

        ```markdown
        ## Description
        - Description of what changed and why
        - References to any related issues (use #)
        - Screenshots/examples if relevant
        
        ## Checklist
        - [ ] Added new tests for new functionality
        - [ ] Tested locally with example tasks
        - [ ] Code follows project style guidelines
        - [ ] Documentation updated
        - [ ] No new warnings introduced
        ```

### 4. Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, a maintainer will merge your PR


## Adding a New Task

Tasks are the core evaluation units in WorkRB. Follow these steps to add a new task:

### Step 1: Choose the Task Type

- **RankingTask** in [src/workrb/tasks/abstract/ranking_base.py](src/workrb/tasks/abstract/ranking_base.py)
- **ClassificationTask** in [src/workrb/tasks/abstract/classification_base.py](src/workrb/tasks/abstract/classification_base.py)

### Step 2: Create Your Task Class

Create a new file in `src/workrb/tasks/ranking/` or `src/workrb/tasks/classification/` based on the task type. 
For a full example, see also [examples/custom_task_example.py](examples/custom_task_example.py).

```python
# src/workrb/tasks/ranking/my_task.py

from workrb.types import ModelInputType
from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup


@register_task()
class MyCustomRankingTask(RankingTask):
    """
    Description of your task.
    
    This task evaluates models on [specific capability].
    Dataset: [dataset name and source]
    """
    
    @property
    def name(self) -> str:
        return "MyCustomRankingTask"
    
    @property
    def description(self) -> str:
        return "Detailed description of what this task evaluates"
    
    @property
    def task_group(self) -> RankingTaskGroup:
        # Choose appropriate group or add new one to RankingTaskGroup enum
        return RankingTaskGroup.JOB2SKILL
    
    @property
    def query_input_type(self) -> ModelInputType:
        """Type of query texts (e.g., JOB_TITLE, SKILL_NAME, etc.)"""
        return ModelInputType.JOB_TITLE
    
    @property
    def target_input_type(self) -> ModelInputType:
        """Type of target texts"""
        return ModelInputType.SKILL_NAME
    
    @property
    def default_metrics(self) -> list[str]:
        """Override default metrics if needed"""
        return ["map", "mrr", "recall@5", "recall@10"]
    
    def load_monolingual_data(self, split: DatasetSplit, language: Language) -> RankingDataset:
        """
        Load dataset for a specific language and split.
        
        Returns:
            RankingDataset with query_texts, target_indices, and target_space
        """
        # Load your data here (from files, HuggingFace datasets, etc.)
        # Example:
        query_texts = ["Software Engineer", "Data Scientist"]
        target_space = ["Python", "Machine Learning", "SQL"]
        target_indices = [
            [0, 2],  # Software Engineer -> Python, SQL
            [0, 1],  # Data Scientist -> Python, Machine Learning
        ]
        
        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=target_space,
            language=language,
        )
```

### Step 3: Add to Module Exports

Update `src/workrb/tasks/__init__.py`:

```python
from .ranking.my_task import MyCustomRankingTask

__all__ = [
    # ... existing tasks
    "MyCustomRankingTask",
]
```

### Step 4: Create Tests

Create `tests/test_my_task.py`:

```python
import pytest
import workrb
from workrb.tasks.abstract.base import Language


def test_my_custom_task_loads():
    """Test that task loads without errors"""
    task = workrb.tasks.MyCustomRankingTask(split="val", languages=["en"])
    dataset = task.lang_datasets[Language.EN]
    
    assert len(dataset.query_texts) > 0
    assert len(dataset.target_space) > 0
    assert len(dataset.target_indices) == len(dataset.query_texts)
```

### Step 5: Test Your Task

```bash
# Run your specific test
uv run pytest tests/test_my_task.py -v

# Run all tests to ensure no regressions
uv run poe test
```

### Step 6: Document Your Task

Add documentation to your task class docstring:
- Dataset source and version
- Task description and motivation
- Expected model behavior
- Any special considerations

**See [examples/custom_task_example.py](examples/custom_task_example.py) for a complete reference implementation.**

## Adding a New Model

Models in WorkRB implement the `ModelInterface` for unified evaluation.

### Step 1: Implement ModelInterface

Create a new file in `src/workrb/models/`:

```python
# src/workrb/models/my_model.py

import torch
from sentence_transformers import SentenceTransformer

from workrb.types import ModelInputType
from workrb.models.base import ModelInterface
from workrb.registry import register_model


@register_model()
class MyCustomModel(ModelInterface):
    """
    Description of your model.
    
    This model uses [architecture/approach] for [task types].
    """
    
    def __init__(self, model_name_or_path: str = "default-model"):
        """
        Initialize the model.
        
        Args:
            model_name_or_path: Model identifier or path
        """
        self.model = SentenceTransformer(model_name_or_path)
        self.model_name_or_path = model_name_or_path
    
    @property
    def name(self) -> str:
        """Return model name for tracking/logging"""
        return f"MyCustomModel-{self.model_name_or_path}"
    
    @property
    def description(self) -> str:
        """Add description for your model."""
        return f"MyCustomModel is BiEncoder based on..."

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """
        Compute similarity scores between queries and targets.
        
        Args:
            queries: List of query strings
            targets: List of target strings
            query_input_type: Type of query (JOB_TITLE, SKILL_NAME, etc.)
            target_input_type: Type of target
        
        Returns:
            Similarity matrix of shape [n_queries, n_targets]
            Higher scores indicate better matches
        """
        # Encode queries and targets
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        target_embeddings = self.model.encode(targets, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity_matrix = torch.nn.functional.cosine_similarity(
            query_embeddings.unsqueeze(1),
            target_embeddings.unsqueeze(0),
            dim=2
        )
        
        return similarity_matrix
    
    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """
        Compute classification scores.
        
        For ranking-based classification, compute similarity to each class label.
        For true classifiers, return logits from classification head.
        
        Args:
            texts: List of input texts to classify
            targets: List of class labels
            input_type: Type of input
            target_input_type: Type of targets (class labels)
        
        Returns:
            Tensor of shape [n_texts, n_classes] with class scores
        """
        # For embedding models, use similarity to class labels
        text_embeddings = self.model.encode(texts, convert_to_tensor=True)
        target_embeddings = self.model.encode(targets, convert_to_tensor=True)
        
        scores = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1),
            target_embeddings.unsqueeze(0),
            dim=2
        )
        
        return scores
    
    @property
    def classification_label_space(self) -> list[str] | None:
        """
        Return list of class labels if model has a classification head.
        
        For embedding-based models, return None (labels provided at inference time).
        For true classifiers, return the ordered list of labels.
        """
        return None
```

### Step 2: Add to Module Exports

Update `src/workrb/models/__init__.py`:

```python
from .my_model import MyCustomModel

__all__ = [
    # ... existing models
    "MyCustomModel",
]
```

### Step 3: Test Your Model

```python
# tests/test_my_model.py

import pytest
import workrb


def test_my_model_initialization():
    """Test model initialization"""
    model = workrb.models.MyCustomModel("all-MiniLM-L6-v2")
    assert model.name is not None


def test_my_model_ranking():
    """Test ranking computation"""
    model = workrb.models.MyCustomModel("all-MiniLM-L6-v2")
    from workrb.types import ModelInputType
    
    queries = ["Software Engineer", "Data Scientist"]
    targets = ["Python", "Machine Learning", "SQL"]
    
    scores = model.compute_rankings(
        queries=queries,
        targets=targets,
        query_input_type=ModelInputType.JOB_TITLE,
        target_input_type=ModelInputType.SKILL_NAME,
    )
    
    assert scores.shape == (len(queries), len(targets))
```

### Step 4: Register Your Model (if using registry)

If you want your model discoverable via `ModelRegistry.list_available()`, use the `@register_model()` decorator (shown in Step 1).

## Adding New Metrics

To add new evaluation metrics:

### Step 1: Implement Metric Function

Add to `src/workrb/metrics/ranking.py` or `classification.py`:

```python
def my_custom_metric(
    prediction_matrix: np.ndarray,
    pos_label_idxs: list[list[int]],
) -> float:
    """
    Calculate my custom metric.
    
    Args:
        prediction_matrix: Scores of shape [n_queries, n_targets]
        pos_label_idxs: List of lists of positive target indices per query
    
    Returns:
        Metric value (higher is better)
    """
    # Your metric implementation
    pass
```

### Step 2: Register in Metric Calculator

Update the metric calculation function to include your metric:

```python
# In calculate_ranking_metrics() or calculate_classification_metrics()
if "my_custom_metric" in metrics:
    results["my_custom_metric"] = my_custom_metric(prediction_matrix, pos_label_idxs)
```

### Step 3: Add Tests

```python
def test_my_custom_metric():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    pos_labels = [[0], [1]]
    
    result = my_custom_metric(scores, pos_labels)
    assert 0 <= result <= 1  # Adjust based on metric range
```

## Code Standards

We use automated tools to maintain code quality:

### Formatting & Linting

- **Formatting**: ruff (automatic)
- **Linter**: ruff (`uv run poe lint`)
- **Docstring style**: numpy

```bash
# Run all checks & auto-fix where possible
uv run poe lint
```

### Testing Requirements

- All new code must have tests
- Tests must pass before merging
- Aim for >80% code coverage

```bash
# Run your specific tests only
uv run pytest test/my_tests.py

# Run all tests (can take some time)
uv run poe test
```

### Documentation Standards

- All public functions/classes must have docstrings
- Use numpy docstring format
- Include:
  - Brief description
  - Args/Parameters
  - Returns
  - Raises (if applicable)
  - Examples (for complex functions)

Example:
```python
def my_function(arg1: str, arg2: int = 5) -> list[str]:
    """
    Brief one-line description.
    
    Longer description if needed, explaining what the function does
    and any important details.
    
    Parameters
    ----------
    arg1 : str
        Description of arg1
    arg2 : int, optional
        Description of arg2, by default 5
    
    Returns
    -------
    list[str]
        Description of return value
    
    Examples
    --------
    >>> my_function("test", 10)
    ['result1', 'result2']
    """
    pass
```


## Questions & Support

- **🐛 Bug reports**: For problems and bugs, use [GitHub Issues](https://github.com/techwolf-ai/workrb/issues)
- **💡 Feature requests**: For new ideas or additions, use [GitHub Issues](https://github.com/techwolf-ai/workrb/issues)
<!-- - **💬 Questions**: [GitHub Discussions](https://github.com/techwolf-ai/workrb/discussions) -->
- **📧 Email**: For other matters, contact maintainers: workrb@techwolf.ai

---

Thank you for contributing to WorkRB! Your efforts help make AI evaluation in the work domain more accessible and transparent for everyone. 🎉

