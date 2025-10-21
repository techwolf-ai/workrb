"""List all available tasks and models."""

from pprint import pprint

from workbench.registry import ModelRegistry, TaskRegistry

if __name__ == "__main__":
    # List all available tasks and models
    available_tasks = TaskRegistry.list_available()
    available_models = ModelRegistry.list_available()

    print("Available tasks:")
    pprint(available_tasks)
    print("\n\nAvailable models:")
    pprint(available_models)
