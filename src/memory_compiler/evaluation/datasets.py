"""Dataset loading utilities for evaluation.

This module provides utilities for loading standard dialogue datasets
for evaluating memory compression.
"""

from __future__ import annotations

from typing import Any, Iterator

from loguru import logger


class DatasetLoader:
    """Loader for standard dialogue evaluation datasets.

    Supported datasets:
    - MSC (Multi-Session Chat): Multi-session dialogues with persona consistency
    - LoCoMo: Long conversation memory benchmark
    - MultiWOZ: Task-oriented dialogue dataset
    - DailyDialog: Daily conversation dataset

    Note: Requires the `datasets` library from HuggingFace.
    """

    SUPPORTED_DATASETS = {
        "msc": "bavard/personachat_truecased",
        "locomo": None,  # Custom loading
        "multiwoz": "multi_woz_v22",
        "daily_dialog": "daily_dialog",
    }

    def __init__(self, dataset_name: str, split: str = "test") -> None:
        """Initialize dataset loader.

        Args:
            dataset_name: Name of the dataset to load.
            split: Dataset split (train/validation/test).
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self._dataset = None

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )

    def load(self, max_samples: int | None = None) -> None:
        """Load the dataset.

        Args:
            max_samples: Maximum number of samples to load.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install the datasets library: pip install datasets"
            )

        hf_name = self.SUPPORTED_DATASETS[self.dataset_name]

        if hf_name is None:
            logger.warning(f"Dataset {self.dataset_name} requires custom loading")
            self._dataset = []
            return

        logger.info(f"Loading dataset: {hf_name} ({self.split})")

        try:
            dataset = load_dataset(hf_name, split=self.split)

            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            self._dataset = dataset
            logger.info(f"Loaded {len(self._dataset)} samples")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            self._dataset = []

    def __len__(self) -> int:
        if self._dataset is None:
            return 0
        return len(self._dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._dataset is None:
            return

        for sample in self._dataset:
            yield self._process_sample(sample)

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a raw dataset sample into standard format."""
        if self.dataset_name == "msc":
            return self._process_msc(sample)
        elif self.dataset_name == "multiwoz":
            return self._process_multiwoz(sample)
        elif self.dataset_name == "daily_dialog":
            return self._process_daily_dialog(sample)
        else:
            return sample

    def _process_msc(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process MSC/PersonaChat sample."""
        dialogue = []

        # Get utterances
        utterances = sample.get("utterances", [])
        if utterances:
            last_utterance = utterances[-1]
            history = last_utterance.get("history", [])

            for i, text in enumerate(history):
                role = "user" if i % 2 == 0 else "assistant"
                dialogue.append({"role": role, "content": text})

        # Get persona as potential QA context
        persona = sample.get("personality", [])

        return {
            "dialogue": dialogue,
            "persona": persona,
            "metadata": {"dataset": "msc"},
        }

    def _process_multiwoz(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process MultiWOZ sample."""
        dialogue = []

        turns = sample.get("turns", {})
        utterances = turns.get("utterance", [])
        speakers = turns.get("speaker", [])

        for utterance, speaker in zip(utterances, speakers):
            role = "user" if speaker == 0 else "assistant"
            dialogue.append({"role": role, "content": utterance})

        return {
            "dialogue": dialogue,
            "services": sample.get("services", []),
            "metadata": {"dataset": "multiwoz"},
        }

    def _process_daily_dialog(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process DailyDialog sample."""
        dialogue = []

        utterances = sample.get("dialog", [])
        for i, text in enumerate(utterances):
            role = "user" if i % 2 == 0 else "assistant"
            dialogue.append({"role": role, "content": text})

        return {
            "dialogue": dialogue,
            "emotion": sample.get("emotion", []),
            "act": sample.get("act", []),
            "metadata": {"dataset": "daily_dialog"},
        }

    def get_sample(self, index: int) -> dict[str, Any]:
        """Get a specific sample by index."""
        if self._dataset is None or index >= len(self._dataset):
            raise IndexError(f"Sample index {index} out of range")

        return self._process_sample(self._dataset[index])

    def create_qa_pairs(self, sample: dict[str, Any]) -> list[tuple[str, str]]:
        """Create QA pairs from a sample for evaluation.

        This generates simple factual questions based on the dialogue.
        """
        qa_pairs = []
        dialogue = sample.get("dialogue", [])

        # Generate questions about entities mentioned
        for turn in dialogue:
            content = turn.get("content", "")

            # Look for "my name is X" patterns
            import re

            name_match = re.search(r"my name is (\w+)", content, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                qa_pairs.append(
                    (f"What is the user's name?", name)
                )

            # Look for "I work at X" patterns
            work_match = re.search(r"I work (?:at|for) ([^.]+)", content, re.IGNORECASE)
            if work_match:
                workplace = work_match.group(1).strip()
                qa_pairs.append(
                    (f"Where does the user work?", workplace)
                )

            # Look for "I like X" patterns
            like_match = re.search(r"I (?:like|love|enjoy) ([^.]+)", content, re.IGNORECASE)
            if like_match:
                liked_thing = like_match.group(1).strip()
                qa_pairs.append(
                    (f"What does the user like?", liked_thing)
                )

        # Add persona-based questions if available
        persona = sample.get("persona", [])
        for trait in persona[:3]:  # Limit to first 3
            qa_pairs.append(
                (f"Is it true that: {trait}?", "yes")
            )

        return qa_pairs


def create_synthetic_dialogue(
    num_turns: int = 20,
    include_entities: bool = True,
    include_events: bool = True,
) -> list[dict[str, str]]:
    """Create a synthetic dialogue for testing.

    Args:
        num_turns: Number of dialogue turns to generate.
        include_entities: Include named entities.
        include_events: Include events/actions.

    Returns:
        List of dialogue turns.
    """
    import random

    names = ["Alice", "Bob", "Charlie", "Diana"]
    companies = ["TechCorp", "DataInc", "AILabs", "CloudSoft"]
    projects = ["Project Alpha", "the ML pipeline", "the new API", "the dashboard"]
    topics = ["machine learning", "data processing", "cloud deployment", "user interface"]

    dialogue = []

    # Opening
    user_name = random.choice(names)
    company = random.choice(companies)

    dialogue.append({
        "role": "user",
        "content": f"Hi, my name is {user_name} and I work at {company}."
    })

    dialogue.append({
        "role": "assistant",
        "content": f"Hello {user_name}! Nice to meet you. How can I help you today?"
    })

    # Middle conversation
    project = random.choice(projects)
    topic = random.choice(topics)

    dialogue.append({
        "role": "user",
        "content": f"I'm working on {project} and need help with {topic}."
    })

    dialogue.append({
        "role": "assistant",
        "content": f"I'd be happy to help with {topic} for {project}. What specific aspect are you working on?"
    })

    # Add more turns
    questions = [
        "Can you explain how this works?",
        "What's the best approach here?",
        "I'm getting an error when I try this.",
        "How do I optimize this?",
        "What are the best practices?",
    ]

    for i in range(4, min(num_turns, 20), 2):
        question = random.choice(questions)
        dialogue.append({"role": "user", "content": question})
        dialogue.append({
            "role": "assistant",
            "content": f"That's a great question. Here's how you can approach it..."
        })

    return dialogue[:num_turns]
