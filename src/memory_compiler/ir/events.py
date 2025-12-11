"""Event and temporal sequence representations for Memory IR.

Events capture actions and state changes over time, enabling temporal
compression of related event sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of events in dialogue."""

    ACTION = "action"  # User/system performed an action
    STATE_CHANGE = "state_change"  # Something changed state
    DECISION = "decision"  # A decision was made
    DISCOVERY = "discovery"  # Something was discovered/learned
    ERROR = "error"  # An error occurred
    RESOLUTION = "resolution"  # A problem was resolved
    QUESTION = "question"  # A question was asked
    ANSWER = "answer"  # An answer was provided
    REQUEST = "request"  # A request was made
    COMPLETION = "completion"  # A task was completed
    OTHER = "other"


@dataclass
class Event:
    """Represents a temporal event in the dialogue.

    Attributes:
        id: Unique identifier for the event.
        description: Natural language description of the event.
        event_type: Category of the event.
        turn_index: The turn index where this event occurred.
        timestamp: Optional actual timestamp if available.
        participants: Entities involved in this event.
        related_facts: Fact IDs related to this event.
        importance_score: Computed importance for pruning.
        metadata: Additional event metadata.
    """

    id: str
    description: str
    event_type: EventType
    turn_index: int
    timestamp: datetime | None = None
    participants: list[str] = field(default_factory=list)  # Entity IDs
    related_facts: list[str] = field(default_factory=list)  # Fact IDs
    importance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return False
        return self.id == other.id

    def __lt__(self, other: Event) -> bool:
        """Compare events by turn index for sorting."""
        return self.turn_index < other.turn_index

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "event_type": self.event_type.value,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "participants": self.participants,
            "related_facts": self.related_facts,
            "importance_score": self.importance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Create event from dictionary representation."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            id=data["id"],
            description=data["description"],
            event_type=EventType(data.get("event_type", "other")),
            turn_index=data["turn_index"],
            timestamp=timestamp,
            participants=data.get("participants", []),
            related_facts=data.get("related_facts", []),
            importance_score=data.get("importance_score", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EventSequence:
    """Represents a sequence of related events.

    Event sequences are candidates for temporal compression - they can
    be summarized into a single compressed description.
    """

    id: str
    events: list[Event] = field(default_factory=list)
    summary: str | None = None
    sequence_type: str = "general"  # e.g., "debugging", "discussion", "task"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def start_turn(self) -> int | None:
        """Get the starting turn index."""
        if not self.events:
            return None
        return min(e.turn_index for e in self.events)

    @property
    def end_turn(self) -> int | None:
        """Get the ending turn index."""
        if not self.events:
            return None
        return max(e.turn_index for e in self.events)

    @property
    def duration(self) -> int:
        """Get the number of turns spanned by this sequence."""
        if not self.events:
            return 0
        return self.end_turn - self.start_turn + 1

    @property
    def total_importance(self) -> float:
        """Sum of importance scores of all events."""
        return sum(e.importance_score for e in self.events)

    def add_event(self, event: Event) -> None:
        """Add an event to the sequence, maintaining order."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.turn_index)

    def get_participants(self) -> set[str]:
        """Get all unique participants across all events."""
        participants = set()
        for event in self.events:
            participants.update(event.participants)
        return participants

    def get_related_facts(self) -> set[str]:
        """Get all unique related facts across all events."""
        facts = set()
        for event in self.events:
            facts.update(event.related_facts)
        return facts

    def can_merge_with(self, other: EventSequence, max_gap: int = 3) -> bool:
        """Check if this sequence can be merged with another.

        Args:
            other: Another event sequence.
            max_gap: Maximum turn gap allowed between sequences.
        """
        if not self.events or not other.events:
            return False

        # Check if sequences are close enough in time
        gap = abs(self.end_turn - other.start_turn)
        if gap > max_gap:
            return False

        # Check if they share participants (related context)
        shared_participants = self.get_participants() & other.get_participants()
        return len(shared_participants) > 0

    def merge_with(self, other: EventSequence) -> EventSequence:
        """Merge with another event sequence."""
        merged = EventSequence(
            id=f"{self.id}_merged",
            events=sorted(self.events + other.events, key=lambda e: e.turn_index),
            sequence_type=self.sequence_type,
            metadata={**self.metadata, **other.metadata},
        )
        return merged

    def to_dict(self) -> dict[str, Any]:
        """Convert sequence to dictionary representation."""
        return {
            "id": self.id,
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary,
            "sequence_type": self.sequence_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventSequence:
        """Create sequence from dictionary representation."""
        return cls(
            id=data["id"],
            events=[Event.from_dict(e) for e in data.get("events", [])],
            summary=data.get("summary"),
            sequence_type=data.get("sequence_type", "general"),
            metadata=data.get("metadata", {}),
        )
