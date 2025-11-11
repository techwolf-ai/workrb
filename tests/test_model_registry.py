"""Test task registry system functionality."""

import pytest

from workrb.registry import (
    ModelRegistry,
    register_model,
)


class TestModelRegistry:
    """Test cases for the ModelRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ModelRegistry._registry.clear()

    def test_register_model_decorator_basic(self):
        """Test basic model registration with decorator."""
        from workrb.models.base import ModelInterface

        @register_model()
        class TestModel(ModelInterface):
            def name(self) -> str:
                return "TestModel"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        # Model should be automatically registered
        assert "TestModel" in ModelRegistry._registry
        assert ModelRegistry._registry["TestModel"] == TestModel

    def test_register_model_decorator_custom_name(self):
        """Test model registration with custom name."""
        from workrb.models.base import ModelInterface

        @register_model("CustomModelName")
        class AnotherTestModel(ModelInterface):
            def name(self) -> str:
                return "AnotherTestModel"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        # Model should be registered with custom name
        assert "CustomModelName" in ModelRegistry._registry
        assert "AnotherTestModel" not in ModelRegistry._registry
        assert ModelRegistry._registry["CustomModelName"] == AnotherTestModel

    def test_register_non_model_class_fails(self):
        """Test that registering non-ModelInterface class raises ValueError."""
        with pytest.raises(ValueError, match="must inherit from ModelInterface"):

            @register_model()
            class NotAModel:
                pass

    def test_registry_get_existing_model(self):
        """Test getting an existing model from registry."""
        from workrb.models.base import ModelInterface

        @register_model()
        class GetTestModel(ModelInterface):
            def name(self) -> str:
                return "GetTestModel"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        retrieved_class = ModelRegistry.get("GetTestModel")
        assert retrieved_class == GetTestModel

    def test_registry_get_nonexistent_model_fails(self):
        """Test that getting nonexistent model raises ValueError."""
        with pytest.raises(ValueError, match="Model 'NonexistentModel' not found"):
            ModelRegistry.get("NonexistentModel")

    def test_register_duplicate_model_name_raises_error(self):
        """Test that registering duplicate model names raises ValueError."""
        from workrb.models.base import ModelInterface

        @register_model("DuplicateModel")
        class FirstModel(ModelInterface):
            def name(self) -> str:
                return "FirstModel"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        # Attempting to register another class with the same name should fail
        with pytest.raises(ValueError, match="already registered"):

            @register_model("DuplicateModel")
            class SecondModel(ModelInterface):
                def name(self) -> str:
                    return "SecondModel"

                def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                    pass

                def _compute_classification(
                    self, texts, targets, input_type, target_input_type=None
                ):
                    pass

                @property
                def classification_label_space(self):
                    return None

    def test_register_same_model_class_twice_succeeds(self):
        """Test that re-registering the same model class is allowed (e.g., during re-import)."""
        from workrb.models.base import ModelInterface

        @register_model("ReimportModel")
        class ReimportModel(ModelInterface):
            def name(self) -> str:
                return "ReimportModel"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        # Re-registering the same exact class should not raise an error
        # This simulates module re-imports
        ModelRegistry.register("ReimportModel", ReimportModel)

        assert ModelRegistry.get("ReimportModel") == ReimportModel

    def test_registry_list_available(self):
        """Test listing available models."""
        from workrb.models.base import ModelInterface

        @register_model()
        class ListTestModel1(ModelInterface):
            def name(self) -> str:
                return "ListTestModel1"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        @register_model()
        class ListTestModel2(ModelInterface):
            def name(self) -> str:
                return "ListTestModel2"

            def _compute_rankings(self, queries, targets, query_input_type, target_input_type):
                pass

            def _compute_classification(self, texts, targets, input_type, target_input_type=None):
                pass

            @property
            def classification_label_space(self):
                return None

        available = ModelRegistry.list_available()

        assert "ListTestModel1" in available
        assert "ListTestModel2" in available
        assert len(available) >= 2

        # Check module path format
        assert available["ListTestModel1"].endswith("ListTestModel1")
        assert available["ListTestModel2"].endswith("ListTestModel2")
