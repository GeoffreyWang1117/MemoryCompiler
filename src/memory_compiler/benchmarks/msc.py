"""MSC (Multi-Session Chat) benchmark implementation.

Multi-Session Chat is a dataset for training and evaluating models
on long-term conversation memory across multiple sessions.

Reference: https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/msc
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

from memory_compiler.benchmarks.base import BenchmarkDataset, BenchmarkSample


class MSCBenchmark(BenchmarkDataset):
    """Multi-Session Chat benchmark for long-term memory evaluation.

    MSC contains multi-session conversations where speakers build
    relationships over time, requiring memory of past interactions.

    Example:
        >>> benchmark = MSCBenchmark(data_dir="./data/msc")
        >>> benchmark.load()
        >>> for sample in benchmark:
        ...     print(sample.dialogue[:100])
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        split: str = "valid",
        num_sessions: int = 5,
    ) -> None:
        """Initialize MSC benchmark.

        Args:
            data_dir: Directory containing MSC data files.
            split: Data split to use ('train', 'valid', 'test').
            num_sessions: Number of sessions to include per conversation.
        """
        super().__init__(data_dir)
        self.split = split
        self.num_sessions = num_sessions
        self._samples: List[BenchmarkSample] = []

    @property
    def name(self) -> str:
        """Get benchmark name."""
        return f"MSC-{self.split}-{self.num_sessions}sessions"

    def load(self) -> None:
        """Load MSC dataset."""
        if self.data_dir and self.data_dir.exists():
            self._load_from_files()
        else:
            logger.warning("MSC data not found, using synthetic samples")
            self._load_synthetic()

    def _load_from_files(self) -> None:
        """Load from actual MSC data files."""
        data_file = self.data_dir / f"msc_{self.split}.json"

        if not data_file.exists():
            logger.warning(f"MSC file not found: {data_file}")
            self._load_synthetic()
            return

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, conv in enumerate(data.get("conversations", [])):
            sample = self._parse_conversation(conv, idx)
            if sample:
                self._samples.append(sample)

        logger.info(f"Loaded {len(self._samples)} MSC samples")

    def _parse_conversation(
        self, conv: Dict[str, Any], idx: int
    ) -> Optional[BenchmarkSample]:
        """Parse a conversation from MSC format."""
        sessions = conv.get("sessions", [])

        if len(sessions) < self.num_sessions:
            return None

        # Combine sessions into dialogue
        dialogue_parts = []
        all_personas = set()

        for session_idx, session in enumerate(sessions[: self.num_sessions]):
            dialogue_parts.append(f"[Session {session_idx + 1}]")

            for turn in session.get("turns", []):
                speaker = turn.get("speaker", "User")
                text = turn.get("text", "")
                dialogue_parts.append(f"{speaker}: {text}")

            # Collect personas mentioned
            for persona in session.get("personas", []):
                all_personas.add(persona)

        dialogue = "\n".join(dialogue_parts)

        # Extract key entities from personas
        key_entities = list(all_personas)[:10]

        # Extract key facts from conversation summary if available
        key_facts = conv.get("key_facts", [])

        return BenchmarkSample(
            id=f"msc_{idx}",
            dialogue=dialogue,
            reference_summary=conv.get("summary"),
            key_entities=key_entities,
            key_facts=key_facts,
            metadata={
                "num_sessions": min(len(sessions), self.num_sessions),
                "num_turns": sum(len(s.get("turns", [])) for s in sessions),
            },
        )

    def _load_synthetic(self) -> None:
        """Generate synthetic MSC-like samples for testing."""
        synthetic_conversations = [
            {
                "sessions": [
                    {
                        "turns": [
                            {"speaker": "User", "text": "Hi! I'm Alex and I love hiking."},
                            {"speaker": "Assistant", "text": "Nice to meet you Alex! Where do you like to hike?"},
                            {"speaker": "User", "text": "I often go to Yosemite. It's beautiful."},
                            {"speaker": "Assistant", "text": "Yosemite is amazing! The Half Dome is iconic."},
                        ],
                        "personas": ["Alex", "hiking enthusiast", "Yosemite visitor"],
                    },
                    {
                        "turns": [
                            {"speaker": "User", "text": "Remember I mentioned hiking? I went last weekend."},
                            {"speaker": "Assistant", "text": "Oh yes! Did you go to Yosemite again?"},
                            {"speaker": "User", "text": "No, this time I tried Joshua Tree. Different vibe."},
                            {"speaker": "Assistant", "text": "Joshua Tree is unique with all the rock formations."},
                        ],
                        "personas": ["Joshua Tree visitor"],
                    },
                    {
                        "turns": [
                            {"speaker": "User", "text": "I'm planning my next adventure - thinking Grand Canyon."},
                            {"speaker": "Assistant", "text": "The Grand Canyon would be incredible for a hiker like you!"},
                            {"speaker": "User", "text": "Yeah, I want to try the Rim-to-Rim trail."},
                            {"speaker": "Assistant", "text": "That's challenging! How are you preparing?"},
                        ],
                        "personas": ["Grand Canyon planner", "experienced hiker"],
                    },
                ],
                "key_facts": [
                    "Alex loves hiking",
                    "Alex has visited Yosemite",
                    "Alex went to Joshua Tree",
                    "Alex is planning to visit Grand Canyon",
                ],
                "summary": "Alex is an avid hiker who has explored Yosemite and Joshua Tree, and is now planning a challenging Rim-to-Rim trail hike at the Grand Canyon.",
            },
            {
                "sessions": [
                    {
                        "turns": [
                            {"speaker": "User", "text": "I just started a new job as a data scientist at Google."},
                            {"speaker": "Assistant", "text": "Congratulations! That's a great company. What team?"},
                            {"speaker": "User", "text": "I'm on the Search team, working on ranking algorithms."},
                            {"speaker": "Assistant", "text": "Search is Google's bread and butter. Exciting!"},
                        ],
                        "personas": ["data scientist", "Google employee", "Search team"],
                    },
                    {
                        "turns": [
                            {"speaker": "User", "text": "Work has been intense. We're launching a new feature."},
                            {"speaker": "Assistant", "text": "Is it related to the ranking work you mentioned?"},
                            {"speaker": "User", "text": "Yes! We're improving results for voice queries."},
                            {"speaker": "Assistant", "text": "Voice search is growing fast. Sounds important."},
                        ],
                        "personas": ["voice search developer"],
                    },
                    {
                        "turns": [
                            {"speaker": "User", "text": "The launch went well! Got promoted to senior."},
                            {"speaker": "Assistant", "text": "Amazing! Your voice search feature must have impressed."},
                            {"speaker": "User", "text": "Thanks! Now I'm leading a small team of three."},
                            {"speaker": "Assistant", "text": "Management responsibility already! Great progress."},
                        ],
                        "personas": ["senior data scientist", "team lead"],
                    },
                ],
                "key_facts": [
                    "User is a data scientist at Google",
                    "User works on the Search team",
                    "User worked on voice query ranking",
                    "User got promoted to senior",
                    "User now leads a team of three",
                ],
                "summary": "A data scientist at Google's Search team who worked on voice query ranking, recently got promoted to senior role and now leads a team of three.",
            },
            {
                "sessions": [
                    {
                        "turns": [
                            {"speaker": "User", "text": "I'm learning to play guitar. Just got my first acoustic."},
                            {"speaker": "Assistant", "text": "That's wonderful! Any particular style you want to learn?"},
                            {"speaker": "User", "text": "I love folk music, especially Bob Dylan."},
                            {"speaker": "Assistant", "text": "Dylan is a great inspiration. Start with 'Blowin in the Wind'."},
                        ],
                        "personas": ["guitar learner", "folk music fan", "Bob Dylan fan"],
                    },
                    {
                        "turns": [
                            {"speaker": "User", "text": "I can play three chords now! G, C, and D."},
                            {"speaker": "Assistant", "text": "Perfect! Those three chords open up so many songs."},
                            {"speaker": "User", "text": "I'm struggling with chord transitions though."},
                            {"speaker": "Assistant", "text": "Practice switching slowly, speed comes with time."},
                        ],
                        "personas": ["beginner guitarist"],
                    },
                    {
                        "turns": [
                            {"speaker": "User", "text": "Guess what? I played at an open mic last night!"},
                            {"speaker": "Assistant", "text": "That's huge! What did you perform?"},
                            {"speaker": "User", "text": "Knockin' on Heaven's Door. People actually clapped!"},
                            {"speaker": "Assistant", "text": "From struggling with chord changes to performing! Impressive growth."},
                        ],
                        "personas": ["performer", "open mic participant"],
                    },
                ],
                "key_facts": [
                    "User is learning guitar",
                    "User likes folk music and Bob Dylan",
                    "User learned G, C, D chords",
                    "User performed at open mic",
                    "User played Knockin' on Heaven's Door",
                ],
                "summary": "A beginner guitarist who loves folk music and Bob Dylan, progressed from learning basic chords to performing at an open mic night.",
            },
        ]

        for idx, conv in enumerate(synthetic_conversations):
            sessions = conv["sessions"]

            # Build dialogue
            dialogue_parts = []
            all_personas = set()

            for session_idx, session in enumerate(sessions):
                dialogue_parts.append(f"[Session {session_idx + 1}]")
                for turn in session["turns"]:
                    dialogue_parts.append(f"{turn['speaker']}: {turn['text']}")
                all_personas.update(session.get("personas", []))

            dialogue = "\n".join(dialogue_parts)

            self._samples.append(
                BenchmarkSample(
                    id=f"msc_synthetic_{idx}",
                    dialogue=dialogue,
                    reference_summary=conv.get("summary"),
                    key_entities=list(all_personas),
                    key_facts=conv.get("key_facts", []),
                    metadata={"synthetic": True, "num_sessions": len(sessions)},
                )
            )

        logger.info(f"Generated {len(self._samples)} synthetic MSC samples")

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Iterate over samples."""
        return iter(self._samples)

    def filter_by_length(
        self,
        min_turns: int = 0,
        max_turns: int = 1000,
    ) -> "MSCBenchmark":
        """Filter samples by dialogue length.

        Args:
            min_turns: Minimum number of turns.
            max_turns: Maximum number of turns.

        Returns:
            New benchmark with filtered samples.
        """
        filtered = MSCBenchmark(
            data_dir=self.data_dir,
            split=self.split,
            num_sessions=self.num_sessions,
        )

        filtered._samples = [
            s for s in self._samples
            if min_turns <= s.metadata.get("num_turns", 0) <= max_turns
        ]

        return filtered
