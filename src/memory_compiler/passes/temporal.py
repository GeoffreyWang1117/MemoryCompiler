"""Temporal Compression pass.

This pass compresses sequences of temporally related events into summaries,
reducing the number of memory units while preserving the essential narrative.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from loguru import logger

from memory_compiler.ir.events import Event, EventSequence, EventType
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import OptimizationPass, PassResult


class TemporalCompressionPass(OptimizationPass):
    """Compresses sequences of related events into summaries.

    This pass identifies groups of temporally adjacent events that share
    context (participants, topics) and compresses them into summary
    events or event sequences.

    Examples:
    - "Fixed bug A, then bug B, then bug C" -> "Fixed 3 bugs"
    - "Asked about X, clarified X, resolved X" -> "Discussed and resolved X"

    Configuration options:
        min_sequence_length: Minimum events to form a compressible sequence
            (default: 3).
        max_turn_gap: Maximum turn gap between events in a sequence
            (default: 5).
        similarity_threshold: Minimum similarity for events to be grouped
            (default: 0.5).
        use_llm_summarization: Whether to use LLM for generating summaries
            (default: False).
    """

    def __init__(
        self,
        min_sequence_length: int = 3,
        max_turn_gap: int = 5,
        similarity_threshold: float = 0.5,
        use_llm_summarization: bool = False,
        summarization_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_sequence_length = min_sequence_length
        self.max_turn_gap = max_turn_gap
        self.similarity_threshold = similarity_threshold
        self.use_llm_summarization = use_llm_summarization
        self.summarization_model = summarization_model

        self._summarizer = None
        if use_llm_summarization and summarization_model:
            self._load_summarizer()

    def _load_summarizer(self) -> None:
        """Load the summarization model."""
        try:
            from transformers import pipeline

            self._summarizer = pipeline(
                "summarization",
                model=self.summarization_model,
                device_map="auto",
            )
            logger.info(f"Loaded summarization model: {self.summarization_model}")
        except Exception as e:
            logger.warning(f"Failed to load summarizer: {e}")
            self.use_llm_summarization = False

    def run(self, ir: MemoryIR) -> PassResult:
        """Run temporal compression on the IR."""
        result = PassResult()

        # Step 1: Get all events sorted by turn
        events = sorted(ir.events.values(), key=lambda e: e.turn_index)

        if len(events) < self.min_sequence_length:
            return result

        # Step 2: Find compressible sequences
        sequences = self._find_compressible_sequences(events, ir)
        result.stats["sequences_found"] = len(sequences)

        if not sequences:
            return result

        # Step 3: Compress each sequence
        for sequence in sequences:
            if len(sequence) >= self.min_sequence_length:
                summary = self._compress_sequence(sequence, ir)

                # Create event sequence in IR
                event_seq = EventSequence(
                    id=f"seq_{uuid.uuid4().hex[:8]}",
                    events=sequence,
                    summary=summary,
                    sequence_type=self._determine_sequence_type(sequence),
                )
                ir.add_event_sequence(event_seq)

                # Remove individual events
                for event in sequence:
                    ir.remove_event(event.id)
                    result.events_removed += 1

                result.events_compressed += 1
                logger.debug(f"Compressed {len(sequence)} events into: {summary}")

        result.modified = result.events_compressed > 0

        logger.info(
            f"TemporalCompression: compressed {result.events_compressed} sequences, "
            f"removed {result.events_removed} individual events"
        )

        return result

    def _find_compressible_sequences(
        self, events: list[Event], ir: MemoryIR
    ) -> list[list[Event]]:
        """Find sequences of events that can be compressed together."""
        if not events:
            return []

        sequences = []
        current_sequence = [events[0]]

        for i in range(1, len(events)):
            prev_event = events[i - 1]
            curr_event = events[i]

            # Check if events should be in the same sequence
            if self._should_group_events(prev_event, curr_event):
                current_sequence.append(curr_event)
            else:
                # Save current sequence if long enough
                if len(current_sequence) >= self.min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = [curr_event]

        # Don't forget the last sequence
        if len(current_sequence) >= self.min_sequence_length:
            sequences.append(current_sequence)

        return sequences

    def _should_group_events(self, event1: Event, event2: Event) -> bool:
        """Determine if two events should be grouped in a sequence."""
        # Check turn gap
        turn_gap = event2.turn_index - event1.turn_index
        if turn_gap > self.max_turn_gap:
            return False

        # Check for shared participants
        shared_participants = set(event1.participants) & set(event2.participants)
        if shared_participants:
            return True

        # Check for same event type
        if event1.event_type == event2.event_type:
            # Compute description similarity
            similarity = self._compute_event_similarity(event1, event2)
            if similarity >= self.similarity_threshold:
                return True

        # Check for related event types
        related_types = {
            (EventType.QUESTION, EventType.ANSWER),
            (EventType.ERROR, EventType.RESOLUTION),
            (EventType.REQUEST, EventType.COMPLETION),
            (EventType.ACTION, EventType.STATE_CHANGE),
        }
        if (event1.event_type, event2.event_type) in related_types:
            return True
        if (event2.event_type, event1.event_type) in related_types:
            return True

        return False

    def _compute_event_similarity(self, event1: Event, event2: Event) -> float:
        """Compute similarity between two events."""
        # Simple word overlap similarity
        words1 = set(event1.description.lower().split())
        words2 = set(event2.description.lower().split())

        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "was", "were", "are", "been", "be", "to"}
        words1 -= stopwords
        words2 -= stopwords

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _compress_sequence(self, sequence: list[Event], ir: MemoryIR) -> str:
        """Generate a summary for a sequence of events."""
        if self.use_llm_summarization and self._summarizer:
            return self._llm_summarize(sequence)
        else:
            return self._heuristic_summarize(sequence)

    def _heuristic_summarize(self, sequence: list[Event]) -> str:
        """Generate a summary using heuristics."""
        n = len(sequence)
        first = sequence[0]
        last = sequence[-1]

        # Group by event type
        type_counts: dict[EventType, int] = defaultdict(int)
        for event in sequence:
            type_counts[event.event_type] += 1

        # Get the dominant type
        dominant_type = max(type_counts, key=type_counts.get)
        dominant_count = type_counts[dominant_type]

        # Get shared participants
        all_participants: set[str] = set()
        for event in sequence:
            all_participants.update(event.participants)
        participants_str = ", ".join(list(all_participants)[:3])

        # Generate summary based on pattern
        if dominant_type == EventType.QUESTION:
            if EventType.ANSWER in type_counts:
                return f"Q&A exchange with {dominant_count} questions answered"
            return f"Discussion with {dominant_count} questions"

        elif dominant_type == EventType.ERROR:
            if EventType.RESOLUTION in type_counts:
                resolved = type_counts[EventType.RESOLUTION]
                return f"Encountered and resolved {resolved} of {dominant_count} issues"
            return f"Encountered {dominant_count} issues"

        elif dominant_type == EventType.ACTION:
            if participants_str:
                return f"Series of {n} actions involving {participants_str}"
            return f"Series of {n} actions from turn {first.turn_index} to {last.turn_index}"

        elif dominant_type == EventType.STATE_CHANGE:
            return f"{n} state changes over turns {first.turn_index}-{last.turn_index}"

        elif dominant_type == EventType.COMPLETION:
            return f"Completed {n} tasks"

        else:
            # Generic summary
            type_name = dominant_type.value.replace("_", " ")
            return f"Sequence of {n} {type_name} events (turns {first.turn_index}-{last.turn_index})"

    def _llm_summarize(self, sequence: list[Event]) -> str:
        """Generate a summary using LLM."""
        # Combine event descriptions
        descriptions = [e.description for e in sequence]
        text = " ".join(descriptions)

        try:
            result = self._summarizer(
                text,
                max_length=50,
                min_length=10,
                do_sample=False,
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._heuristic_summarize(sequence)

    def _determine_sequence_type(self, sequence: list[Event]) -> str:
        """Determine the type of an event sequence."""
        type_counts: dict[EventType, int] = defaultdict(int)
        for event in sequence:
            type_counts[event.event_type] += 1

        dominant_type = max(type_counts, key=type_counts.get)

        type_to_sequence_type = {
            EventType.QUESTION: "qa_exchange",
            EventType.ERROR: "debugging",
            EventType.ACTION: "action_sequence",
            EventType.STATE_CHANGE: "state_transitions",
            EventType.COMPLETION: "task_completion",
            EventType.DECISION: "decision_making",
        }

        return type_to_sequence_type.get(dominant_type, "general")


class DialoguePhaseCompressionPass(OptimizationPass):
    """Compresses entire dialogue phases into summaries.

    This is a more aggressive compression that identifies distinct
    phases in the dialogue (e.g., introduction, problem solving, wrap-up)
    and compresses less important phases.
    """

    def __init__(
        self,
        min_phase_length: int = 5,
        compress_ratio: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_phase_length = min_phase_length
        self.compress_ratio = compress_ratio

    def run(self, ir: MemoryIR) -> PassResult:
        """Run dialogue phase compression."""
        result = PassResult()

        if len(ir.dialogue_turns) < self.min_phase_length * 2:
            return result

        # Identify dialogue phases
        phases = self._identify_phases(ir)
        result.stats["phases_found"] = len(phases)

        # Score phases by importance
        phase_scores = self._score_phases(phases, ir)

        # Compress low-importance phases
        total_events_before = len(ir.events)
        target_events = int(total_events_before * (1 - self.compress_ratio))

        for phase_idx, score in sorted(phase_scores.items(), key=lambda x: x[1]):
            if len(ir.events) <= target_events:
                break

            phase = phases[phase_idx]
            events_in_phase = ir.get_events_in_range(phase[0], phase[1])

            if len(events_in_phase) >= 2:
                # Create summary for this phase
                summary = self._summarize_phase(phase, events_in_phase, ir)

                # Create a single summary event
                summary_event = Event(
                    id=f"evt_{uuid.uuid4().hex[:8]}",
                    description=summary,
                    event_type=EventType.OTHER,
                    turn_index=phase[0],
                    metadata={"compressed_from": [e.id for e in events_in_phase]},
                )

                # Remove old events
                for event in events_in_phase:
                    ir.remove_event(event.id)
                    result.events_removed += 1

                # Add summary event
                ir.add_event(summary_event)
                result.events_compressed += 1

        result.modified = result.events_compressed > 0
        return result

    def _identify_phases(self, ir: MemoryIR) -> list[tuple[int, int]]:
        """Identify distinct phases in the dialogue."""
        phases = []
        n_turns = len(ir.dialogue_turns)

        if n_turns == 0:
            return phases

        # Simple segmentation based on turn count
        phase_size = max(self.min_phase_length, n_turns // 5)

        for start in range(0, n_turns, phase_size):
            end = min(start + phase_size - 1, n_turns - 1)
            phases.append((start, end))

        return phases

    def _score_phases(
        self, phases: list[tuple[int, int]], ir: MemoryIR
    ) -> dict[int, float]:
        """Score each phase by importance."""
        scores = {}

        for i, (start, end) in enumerate(phases):
            # Get events in this phase
            events = ir.get_events_in_range(start, end)

            # Score based on:
            # 1. Number of important events
            event_importance = sum(e.importance_score for e in events)

            # 2. Recency (later phases more important)
            recency_score = (i + 1) / len(phases)

            # 3. Information density (facts per turn)
            facts_in_phase = [
                f
                for f in ir.iter_facts()
                if any(start <= t <= end for t in f.source_turns)
            ]
            density = len(facts_in_phase) / max(1, end - start + 1)

            scores[i] = event_importance * 0.4 + recency_score * 0.4 + density * 0.2

        return scores

    def _summarize_phase(
        self, phase: tuple[int, int], events: list[Event], ir: MemoryIR
    ) -> str:
        """Create a summary for a dialogue phase."""
        start, end = phase

        # Get facts from this phase
        facts = [
            f
            for f in ir.iter_facts()
            if any(start <= t <= end for t in f.source_turns)
        ]

        n_events = len(events)
        n_facts = len(facts)
        n_turns = end - start + 1

        return f"Phase (turns {start}-{end}): {n_turns} turns with {n_events} events and {n_facts} facts"
