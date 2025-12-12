"""LoCoMo (Long Context Memory) benchmark implementation.

LoCoMo is a benchmark for evaluating long-context memory in
conversational AI systems.

Reference: https://github.com/LoCoMo-Eval/LoCoMo
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

from memory_compiler.benchmarks.base import BenchmarkDataset, BenchmarkSample


class LoCoMoBenchmark(BenchmarkDataset):
    """LoCoMo benchmark for long-context memory evaluation.

    LoCoMo focuses on testing memory retention over very long
    conversations with specific probing questions.

    Example:
        >>> benchmark = LoCoMoBenchmark()
        >>> benchmark.load()
        >>> for sample in benchmark:
        ...     print(f"Questions: {len(sample.questions)}")
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        split: str = "test",
        context_length: str = "long",  # 'short', 'medium', 'long'
    ) -> None:
        """Initialize LoCoMo benchmark.

        Args:
            data_dir: Directory containing LoCoMo data.
            split: Data split ('train', 'valid', 'test').
            context_length: Context length category.
        """
        super().__init__(data_dir)
        self.split = split
        self.context_length = context_length
        self._samples: List[BenchmarkSample] = []

    @property
    def name(self) -> str:
        """Get benchmark name."""
        return f"LoCoMo-{self.split}-{self.context_length}"

    def load(self) -> None:
        """Load LoCoMo dataset."""
        if self.data_dir and self.data_dir.exists():
            self._load_from_files()
        else:
            logger.warning("LoCoMo data not found, using synthetic samples")
            self._load_synthetic()

    def _load_from_files(self) -> None:
        """Load from actual LoCoMo data files."""
        data_file = self.data_dir / f"locomo_{self.split}_{self.context_length}.json"

        if not data_file.exists():
            logger.warning(f"LoCoMo file not found: {data_file}")
            self._load_synthetic()
            return

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            sample = self._parse_item(item, idx)
            if sample:
                self._samples.append(sample)

        logger.info(f"Loaded {len(self._samples)} LoCoMo samples")

    def _parse_item(self, item: Dict[str, Any], idx: int) -> Optional[BenchmarkSample]:
        """Parse an item from LoCoMo format."""
        dialogue = item.get("context", "")
        questions = item.get("questions", [])

        # Extract entities mentioned in questions
        key_entities = []
        for q in questions:
            entities = q.get("entities", [])
            key_entities.extend(entities)

        key_entities = list(set(key_entities))

        return BenchmarkSample(
            id=f"locomo_{idx}",
            dialogue=dialogue,
            key_entities=key_entities,
            key_facts=item.get("key_facts", []),
            questions=questions,
            metadata={
                "context_length": len(dialogue),
                "num_questions": len(questions),
            },
        )

    def _load_synthetic(self) -> None:
        """Generate synthetic LoCoMo-like samples."""
        # Generate samples with varying context lengths
        if self.context_length == "short":
            target_turns = 10
        elif self.context_length == "medium":
            target_turns = 30
        else:  # long
            target_turns = 50

        # Sample conversation themes
        themes = [
            {
                "topic": "travel planning",
                "entities": ["Paris", "Tokyo", "Barcelona", "user"],
                "facts": [
                    "User wants to visit Paris in spring",
                    "User prefers boutique hotels",
                    "User is allergic to shellfish",
                    "User visited Tokyo last year",
                ],
                "turns": [
                    ("I'm planning a trip to Europe next spring.", "That sounds exciting! Any specific countries in mind?"),
                    ("I've always wanted to see Paris.", "Paris is beautiful in spring! The cherry blossoms are lovely."),
                    ("I prefer smaller boutique hotels over big chains.", "Boutique hotels in Le Marais area are wonderful."),
                    ("I should mention I'm allergic to shellfish.", "Good to know! French cuisine offers many non-seafood options."),
                    ("I visited Tokyo last year and loved it.", "Japan is amazing! Different vibe from Paris though."),
                    ("Maybe I'll add Barcelona to the trip.", "Barcelona has great architecture and food!"),
                    ("How many days should I spend in each city?", "I'd suggest 4-5 days in Paris, 3-4 in Barcelona."),
                ],
                "questions": [
                    {"question": "Where does the user want to go?", "answer": "Paris and Barcelona", "entities": ["Paris", "Barcelona"]},
                    {"question": "What dietary restriction does the user have?", "answer": "Allergic to shellfish", "entities": ["user"]},
                    {"question": "What type of accommodation does the user prefer?", "answer": "Boutique hotels", "entities": ["user"]},
                ],
            },
            {
                "topic": "career development",
                "entities": ["user", "TechCorp", "MBA", "product management"],
                "facts": [
                    "User works at TechCorp",
                    "User is considering MBA",
                    "User interested in product management",
                    "User has 5 years experience",
                ],
                "turns": [
                    ("I've been at TechCorp for 5 years now.", "That's solid tenure! How's the experience been?"),
                    ("It's been great but I'm thinking about my next step.", "Career growth is important. What are you considering?"),
                    ("I'm interested in transitioning to product management.", "PM roles are in demand. Do you have any product experience?"),
                    ("I've led some product initiatives but not formally.", "That's a good start. Have you considered an MBA?"),
                    ("Actually yes, I'm looking at part-time MBA programs.", "Part-time MBA lets you keep working while studying."),
                    ("My company might even sponsor it.", "TechCorp sponsoring would be great! Check their policy."),
                    ("I'm particularly interested in Stanford or Wharton.", "Both are excellent for tech careers."),
                ],
                "questions": [
                    {"question": "How long has the user worked at their company?", "answer": "5 years", "entities": ["user", "TechCorp"]},
                    {"question": "What role is the user interested in?", "answer": "Product management", "entities": ["product management"]},
                    {"question": "Is the user considering further education?", "answer": "Yes, MBA programs", "entities": ["MBA"]},
                ],
            },
            {
                "topic": "home renovation",
                "entities": ["kitchen", "contractor", "budget", "user"],
                "facts": [
                    "User is renovating kitchen",
                    "Budget is $50,000",
                    "User hired contractor named Mike",
                    "Project timeline is 3 months",
                ],
                "turns": [
                    ("We're finally renovating our kitchen!", "Exciting! Kitchen renos make a huge difference."),
                    ("Our budget is about $50,000.", "That's a good budget for a quality renovation."),
                    ("We hired a contractor named Mike.", "Having a reliable contractor is crucial."),
                    ("He says it'll take about 3 months.", "Three months is reasonable for a full kitchen."),
                    ("We're going with quartz countertops.", "Quartz is durable and low maintenance."),
                    ("The cabinet delivery got delayed.", "Supply chain issues are common. How long?"),
                    ("Two weeks delay, so we're adjusting the timeline.", "Hopefully Mike can work around it."),
                ],
                "questions": [
                    {"question": "What is the renovation budget?", "answer": "$50,000", "entities": ["budget"]},
                    {"question": "Who is the contractor?", "answer": "Mike", "entities": ["contractor"]},
                    {"question": "What is the expected timeline?", "answer": "3 months", "entities": ["user"]},
                ],
            },
        ]

        for idx, theme in enumerate(themes):
            # Build extended dialogue
            dialogue_parts = []
            base_turns = theme["turns"]

            # Repeat and vary turns to reach target length
            turn_count = 0
            while turn_count < target_turns:
                for user_msg, assistant_msg in base_turns:
                    dialogue_parts.append(f"User: {user_msg}")
                    dialogue_parts.append(f"Assistant: {assistant_msg}")
                    turn_count += 1

                    if turn_count >= target_turns:
                        break

                # Add some filler conversation
                if turn_count < target_turns:
                    fillers = [
                        ("That makes sense.", "Happy to help! Any other questions?"),
                        ("Let me think about that.", "Take your time, no rush."),
                        ("Thanks for the advice.", "You're welcome! Feel free to ask more."),
                    ]
                    for filler in fillers:
                        if turn_count >= target_turns:
                            break
                        dialogue_parts.append(f"User: {filler[0]}")
                        dialogue_parts.append(f"Assistant: {filler[1]}")
                        turn_count += 1

            dialogue = "\n".join(dialogue_parts)

            self._samples.append(
                BenchmarkSample(
                    id=f"locomo_synthetic_{idx}",
                    dialogue=dialogue,
                    key_entities=theme["entities"],
                    key_facts=theme["facts"],
                    questions=theme["questions"],
                    metadata={
                        "synthetic": True,
                        "topic": theme["topic"],
                        "num_turns": len(dialogue_parts) // 2,
                    },
                )
            )

        logger.info(f"Generated {len(self._samples)} synthetic LoCoMo samples")

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Iterate over samples."""
        return iter(self._samples)

    def evaluate_qa(
        self,
        sample: BenchmarkSample,
        compressed_context: str,
        qa_model_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Evaluate QA performance on compressed context.

        Args:
            sample: Benchmark sample with questions.
            compressed_context: Compressed dialogue.
            qa_model_fn: Function to answer questions given context.

        Returns:
            QA evaluation results.
        """
        if not sample.questions:
            return {"accuracy": 1.0, "questions_answered": 0}

        if qa_model_fn is None:
            # Default: simple substring matching
            def qa_model_fn(context: str, question: str) -> str:
                return context[:100]  # Placeholder

        correct = 0
        results = []

        for q in sample.questions:
            question = q.get("question", "")
            expected = q.get("answer", "").lower()

            predicted = qa_model_fn(compressed_context, question).lower()

            # Simple matching
            is_correct = expected in predicted or predicted in expected

            results.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })

            if is_correct:
                correct += 1

        return {
            "accuracy": correct / len(sample.questions),
            "questions_answered": len(sample.questions),
            "details": results,
        }
