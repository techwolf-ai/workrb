"""Base task classes."""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, final

from workrb.types import DatasetSplit, LabelType, Language

logger = logging.getLogger(__name__)


class BaseTaskGroup(str, Enum):
    """
    Task group enum.

    Used for grouping tasks for results aggregation.
    For example, we can aggregate macro-results across all skill normalization tasks.
    """


class TaskType(str, Enum):
    """Task type enum."""

    CLASSIFICATION = "classification"
    RANKING = "ranking"


class Task(ABC):
    def __init__(
        self,
        languages: list[str] | None,
        split: str,
        unsupported_lang_mode: Literal["error", "skip"] = "skip",
    ):
        """Initialize task.

        Args:
            languages: List of languages to evaluate on
            split: Split to evaluate on
            unsupported_lang_mode: Mode to handle unsupported languages. This can occur when
                defining a set of tasks with a target set of languages, but not all tasks support
                all languages.
        """
        self.languages = (
            self.supported_languages
            if languages is None
            else self._parse_languages(languages, unsupported_lang_mode)
        )

        try:
            self.split = DatasetSplit(split)
        except ValueError as e:
            raise ValueError(
                f"Invalid split: '{split}'. Supported splits: {list(DatasetSplit)}"
            ) from e

        # Load datasets for all languages
        self.lang_datasets = self._load_multilingual_data(
            languages=self.languages, split=self.split
        )

    def _parse_languages(
        self, languages: list[str], unsupported_lang_mode: Literal["error", "skip"]
    ) -> list[Language]:
        """Parse languages into Language enum."""
        assert unsupported_lang_mode in ["error", "skip"], (
            f"Invalid unsupported_lang_mode: '{unsupported_lang_mode}'. Supported modes: 'error', 'skip'"
        )
        parsed_languages = []
        for lang_str in languages:
            lang = Language(lang_str)

            if lang not in self.supported_languages:
                msg = f"Language '{lang}' is not supported for task '{self.name}'"

                if unsupported_lang_mode == "error":
                    raise ValueError(msg)
                if unsupported_lang_mode == "skip":
                    logger.warning(msg)
                    continue
            parsed_languages.append(lang)
        return parsed_languages

    def get_task_config(self) -> dict[str, Any]:
        """Get task configuration."""
        return {
            "name": self.name,
            "languages": [lang.value for lang in self.languages],
            "split": self.split.value,
            "class": self.__class__.__name__,
            "task_group": self.task_group.value,
            "task_type": self.task_type.value,
            "label_type": self.label_type.value,
            "description": self.description,
        }

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def supported_languages(self) -> list[Language]:
        """Return all supported languages for the task."""
        return list(set(self.supported_query_languages + self.supported_target_languages))

    @property
    @abstractmethod
    def supported_query_languages(self) -> list[Language]:
        pass

    @property
    @abstractmethod
    def supported_target_languages(self) -> list[Language]:
        pass

    @property
    @abstractmethod
    def task_group(self) -> BaseTaskGroup:
        """Task group for the task."""

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Task type for the task."""

    @property
    @abstractmethod
    def label_type(self) -> LabelType:
        pass

    @property
    @abstractmethod
    def default_metrics(self) -> list[str]:
        pass

    @property
    def citation(self) -> str:
        """Citation for the task."""

    @property
    def split_seed(self) -> int:
        """Split seed for reproducible splits."""
        return 42

    @property
    def split_test_fraction(self) -> float:
        """Default fraction of data to use for test split."""
        return 0.8

    def get_size_oneliner(self, language: Language) -> str:
        """Get dataset size summary to display status."""
        return ""

    @final
    @property
    def split_val_fraction(self) -> float:
        """Default fraction of data to use for validation split."""
        assert 0 <= self.split_test_fraction <= 1, "Split test fraction must be between 0 and 1"
        return 1 - self.split_test_fraction

    def _load_multilingual_data(
        self, languages: list[Language], split: DatasetSplit
    ) -> dict[Language, Any]:
        """Load datasets for all languages."""
        lang_datasets: dict[Language, Any] = {}

        # Check if languages are supported
        non_supported_languages = set(languages) - set(self.supported_languages)
        if non_supported_languages:
            raise ValueError(
                f"The following languages are defined for '{self.name}' but are not supported: {non_supported_languages}. Supported languages: {self.supported_languages}"
            )

        for lang in languages:
            lang_datasets[lang] = self.load_monolingual_data(split=split, language=lang)
        return lang_datasets

    @abstractmethod
    def load_monolingual_data(self, language: Language, split: DatasetSplit) -> Any:
        pass

    @abstractmethod
    def evaluate(self, model, metrics=None, language: Language = Language.EN) -> dict[str, float]:
        pass
