"""
Decorator-based Task Registry System.

Tasks can self-register using @register_task() decorator.
"""

import importlib
import logging
import pkgutil
from typing import Any, TypeVar

# Type variables for generic decorators
TaskType = TypeVar("TaskType", bound="Task")
ModelType = TypeVar("ModelType", bound="ModelInterface")
logger = logging.getLogger(__name__)


class TaskRegistry:
    """Global registry for task classes."""

    _registry: dict[str, type["Task"]] = {}

    @classmethod
    def register(cls, name: str, task_class: type["Task"]) -> None:
        """Register a task class."""
        # Import here to avoid circular imports
        from .tasks.abstract import Task

        if not issubclass(task_class, Task):
            raise ValueError(f"Task class {task_class} must inherit from Task")

        if name in cls._registry:
            existing_class = cls._registry[name]
            if existing_class != task_class:
                raise ValueError(
                    f"Task name '{name}' is already registered to "
                    f"{existing_class.__module__}.{existing_class.__name__}. "
                    f"Cannot register {task_class.__module__}.{task_class.__name__} with the same name. "
                    f"Please choose a unique name or use @register_task('different_name')."
                )
            # If it's the same class being registered again (e.g., re-import), that's fine
            logger.debug(f"Task '{name}' re-registered with same class {task_class}")

        cls._registry[name] = task_class

    @classmethod
    def get(cls, name: str) -> type["Task"]:
        """Get a registered task class."""
        if name not in cls._registry:
            # Try auto-discovery if not found
            cls.auto_discover()

        if name not in cls._registry:
            raise ValueError(
                f"Task '{name}' not found. Available tasks: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> "Task":
        """Create an instance of a registered task."""
        task_class = cls.get(name)
        return task_class(**kwargs)

    @classmethod
    def list_available(cls) -> dict[str, str]:
        """List all registered tasks with their module paths."""
        # Ensure all tasks are discovered
        cls.auto_discover()

        return {
            name: task_class.__module__ + "." + task_class.__name__
            for name, task_class in cls._registry.items()
        }

    @classmethod
    def auto_discover(cls):
        """Automatically import all task modules to trigger registration."""
        # Import all submodules in tasks package to trigger registration
        import workrb.tasks as tasks_package

        for importer, modname, ispkg in pkgutil.walk_packages(
            tasks_package.__path__, tasks_package.__name__ + "."
        ):
            importlib.import_module(modname)


def register_task(name: str | None = None):
    """
    Decorator registering a task class.

    Usage:
        @register_task()  # Uses class name automatically
        class MyTask(Task):
            pass

        @register_task("CustomName")  # Uses custom name
        class MyTask(Task):
            pass
    """

    def decorator(cls: type[TaskType]) -> type[TaskType]:
        task_name = name if name is not None else cls.__name__
        TaskRegistry.register(task_name, cls)
        return cls

    # Handle both @register_task() and @register_task
    if isinstance(name, type):
        # Called as @register_task (without parentheses)
        cls = name
        TaskRegistry.register(cls.__name__, cls)
        return cls

    return decorator


# Convenience function for WorkRB integration
def create_task_from_config(task_config: dict[str, Any]) -> "Task":
    """Create a task instance from configuration dictionary."""
    task_class_name = task_config["class"]
    languages = task_config["languages"]

    # Extract task-specific parameters (exclude 'class' and 'languages')
    task_params = {k: v for k, v in task_config.items() if k not in ["class", "languages"]}

    return TaskRegistry.create(task_class_name, languages=languages, **task_params)


class ModelRegistry:
    """Global registry for model classes."""

    _registry: dict[str, type["ModelInterface"]] = {}

    @classmethod
    def register(cls, name: str, model_class: type["ModelInterface"]) -> None:
        """Register a model class."""
        # Import here to avoid circular imports
        from .models.base import ModelInterface

        if not issubclass(model_class, ModelInterface):
            raise ValueError(f"Model class {model_class} must inherit from ModelInterface")

        if name in cls._registry:
            existing_class = cls._registry[name]
            if existing_class != model_class:
                raise ValueError(
                    f"Model name '{name}' is already registered to "
                    f"{existing_class.__module__}.{existing_class.__name__}. "
                    f"Cannot register {model_class.__module__}.{model_class.__name__} with the same name. "
                    f"Please choose a unique name."
                )
            # If it's the same class being registered again (e.g., re-import), that's fine
            logger.debug(f"Model '{name}' re-registered with same class {model_class}")

        cls._registry[name] = model_class

    @classmethod
    def get(cls, name: str) -> type["ModelInterface"]:
        """Get a registered model class."""
        if name not in cls._registry:
            raise ValueError(
                f"Model '{name}' not found. Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> "ModelInterface":
        """Create an instance of a registered model."""
        model_class = cls.get(name)
        return model_class(**kwargs)

    @classmethod
    def list_available(cls) -> dict[str, str]:
        """List all registered models with their module paths."""
        return {
            name: model_class.__module__ + "." + model_class.__name__
            for name, model_class in cls._registry.items()
        }


def register_model(name: str | None = None):
    """
    Decorator for registering a model class.

    Usage:
        @register_model()  # Uses class name automatically
        class MyModel(ModelInterface):
            pass

        @register_model("CustomName")  # Uses custom name
        class MyModel(ModelInterface):
            pass
    """

    def decorator(cls: type[ModelType]) -> type[ModelType]:
        model_name = name if name is not None else cls.__name__
        ModelRegistry.register(model_name, cls)
        return cls

    # Handle both @register_model() and @register_model
    if isinstance(name, type):
        # Called as @register_model (without parentheses)
        cls = name
        ModelRegistry.register(cls.__name__, cls)
        return cls

    return decorator
