"""Context generation from Memory IR.

This module converts optimized Memory IR back into natural language
compressed context that can be used as input for language models.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from loguru import logger

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.events import Event, EventSequence
from memory_compiler.ir.facts import Fact, FactType
from memory_compiler.ir.memory_ir import MemoryIR


class ContextGenerator:
    """Generates natural language context from Memory IR.

    This class converts the structured Memory IR representation back into
    readable text that preserves the key information while being concise.

    The generator supports multiple output formats:
    - Narrative: A flowing narrative summarizing the dialogue
    - Structured: Organized sections for entities, facts, events
    - Bullet points: Concise bullet-point summary
    - JSON: Structured JSON output for programmatic use

    Configuration options:
        output_format: Output format ("narrative", "structured", "bullets", "json")
            (default: "structured").
        max_tokens: Target maximum tokens in output (soft limit, default: 2048).
        include_entities: Whether to include entity section (default: True).
        include_facts: Whether to include facts section (default: True).
        include_events: Whether to include events section (default: True).
        group_by_entity: Group facts by their subject entity (default: True).
        temporal_ordering: Order content chronologically (default: True).
    """

    def __init__(
        self,
        output_format: str = "structured",
        max_tokens: int = 2048,
        include_entities: bool = True,
        include_facts: bool = True,
        include_events: bool = True,
        group_by_entity: bool = True,
        temporal_ordering: bool = True,
        **kwargs: Any,
    ) -> None:
        self.output_format = output_format
        self.max_tokens = max_tokens
        self.include_entities = include_entities
        self.include_facts = include_facts
        self.include_events = include_events
        self.group_by_entity = group_by_entity
        self.temporal_ordering = temporal_ordering
        self.config = kwargs

    def generate(self, ir: MemoryIR) -> str:
        """Generate compressed context from Memory IR.

        Args:
            ir: The Memory IR to convert to text.

        Returns:
            Generated context string.
        """
        if self.output_format == "narrative":
            return self._generate_narrative(ir)
        elif self.output_format == "structured":
            return self._generate_structured(ir)
        elif self.output_format == "bullets":
            return self._generate_bullets(ir)
        elif self.output_format == "json":
            return self._generate_json(ir)
        else:
            logger.warning(f"Unknown format {self.output_format}, using structured")
            return self._generate_structured(ir)

    def _generate_narrative(self, ir: MemoryIR) -> str:
        """Generate a flowing narrative summary."""
        sections = []

        # Introduction with key entities
        entities = list(ir.iter_entities())
        if entities:
            key_entities = sorted(entities, key=lambda e: e.importance_score, reverse=True)[:5]
            entity_names = [e.name for e in key_entities]
            if entity_names:
                sections.append(f"This conversation involves {self._format_list(entity_names)}.")

        # Main content organized by importance
        facts = sorted(ir.iter_facts(), key=lambda f: f.importance_score, reverse=True)

        if facts:
            # Group into high and medium importance
            high_importance = [f for f in facts if f.importance_score >= 0.7]
            medium_importance = [f for f in facts if 0.4 <= f.importance_score < 0.7]

            if high_importance:
                sentences = [self._fact_to_sentence(f) for f in high_importance]
                sections.append(" ".join(sentences))

            if medium_importance:
                sentences = [self._fact_to_sentence(f) for f in medium_importance[:10]]
                sections.append("Additionally, " + " ".join(sentences))

        # Events summary
        events = list(ir.iter_events())
        sequences = ir.event_sequences

        if sequences:
            seq_summaries = [seq.summary or self._summarize_sequence(seq) for seq in sequences]
            if seq_summaries:
                sections.append("Key events: " + "; ".join(seq_summaries) + ".")

        if events:
            important_events = sorted(events, key=lambda e: e.importance_score, reverse=True)[:5]
            event_descs = [e.description for e in important_events]
            if event_descs:
                sections.append("Notable occurrences: " + "; ".join(event_descs) + ".")

        return "\n\n".join(sections)

    def _generate_structured(self, ir: MemoryIR) -> str:
        """Generate structured output with clear sections."""
        sections = []

        # Entities section
        if self.include_entities:
            entity_section = self._generate_entity_section(ir)
            if entity_section:
                sections.append(entity_section)

        # Facts section
        if self.include_facts:
            facts_section = self._generate_facts_section(ir)
            if facts_section:
                sections.append(facts_section)

        # Events section
        if self.include_events:
            events_section = self._generate_events_section(ir)
            if events_section:
                sections.append(events_section)

        return "\n\n".join(sections)

    def _generate_entity_section(self, ir: MemoryIR) -> str:
        """Generate the entities section."""
        entities = sorted(ir.iter_entities(), key=lambda e: e.importance_score, reverse=True)

        if not entities:
            return ""

        lines = ["## Key Entities"]

        # Group by type
        by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for entity in entities:
            by_type[entity.type].append(entity)

        for entity_type in [
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.PROJECT,
            EntityType.CONCEPT,
            EntityType.LOCATION,
            EntityType.PRODUCT,
            EntityType.OTHER,
        ]:
            type_entities = by_type.get(entity_type, [])
            if type_entities:
                type_name = entity_type.value.capitalize() + "s"
                lines.append(f"\n### {type_name}")
                for entity in type_entities[:10]:  # Limit per type
                    desc = self._describe_entity(entity, ir)
                    lines.append(f"- **{entity.name}**: {desc}")

        return "\n".join(lines)

    def _generate_facts_section(self, ir: MemoryIR) -> str:
        """Generate the facts section."""
        facts = sorted(ir.iter_facts(), key=lambda f: f.importance_score, reverse=True)

        if not facts:
            return ""

        lines = ["## Key Information"]

        if self.group_by_entity:
            # Group facts by subject
            by_subject: dict[str, list[Fact]] = defaultdict(list)
            for fact in facts:
                by_subject[fact.subject].append(fact)

            # Sort subjects by total importance of their facts
            sorted_subjects = sorted(
                by_subject.keys(),
                key=lambda s: sum(f.importance_score for f in by_subject[s]),
                reverse=True,
            )

            for subject in sorted_subjects[:15]:  # Limit subjects
                subject_facts = by_subject[subject][:5]  # Limit facts per subject
                lines.append(f"\n**{subject}:**")
                for fact in subject_facts:
                    lines.append(f"- {fact.predicate} {fact.object}")

        else:
            # Flat list of facts
            for fact in facts[:30]:  # Limit total facts
                lines.append(f"- {fact.natural_language}")

        return "\n".join(lines)

    def _generate_events_section(self, ir: MemoryIR) -> str:
        """Generate the events section."""
        events = list(ir.iter_events())
        sequences = ir.event_sequences

        if not events and not sequences:
            return ""

        lines = ["## Timeline"]

        # Add compressed sequences
        if sequences:
            lines.append("\n### Event Sequences")
            for seq in sequences:
                summary = seq.summary or self._summarize_sequence(seq)
                if seq.start_turn is not None and seq.end_turn is not None:
                    lines.append(f"- Turns {seq.start_turn}-{seq.end_turn}: {summary}")
                else:
                    lines.append(f"- {summary}")

        # Add individual events
        if events:
            if self.temporal_ordering:
                events = sorted(events, key=lambda e: e.turn_index)
            else:
                events = sorted(events, key=lambda e: e.importance_score, reverse=True)

            lines.append("\n### Key Events")
            for event in events[:15]:  # Limit events
                lines.append(f"- [Turn {event.turn_index}] {event.description}")

        return "\n".join(lines)

    def _generate_bullets(self, ir: MemoryIR) -> str:
        """Generate concise bullet-point summary."""
        bullets = []

        # Most important entities
        entities = sorted(ir.iter_entities(), key=lambda e: e.importance_score, reverse=True)
        if entities:
            names = [e.name for e in entities[:5]]
            bullets.append(f"• Participants: {', '.join(names)}")

        # Most important facts
        facts = sorted(ir.iter_facts(), key=lambda f: f.importance_score, reverse=True)
        for fact in facts[:15]:
            bullets.append(f"• {fact.natural_language}")

        # Event summaries
        for seq in ir.event_sequences:
            summary = seq.summary or self._summarize_sequence(seq)
            bullets.append(f"• {summary}")

        # Individual important events
        events = sorted(ir.iter_events(), key=lambda e: e.importance_score, reverse=True)
        for event in events[:5]:
            bullets.append(f"• {event.description}")

        return "\n".join(bullets)

    def _generate_json(self, ir: MemoryIR) -> str:
        """Generate JSON output."""
        import json

        output = {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type.value,
                    "importance": round(e.importance_score, 3),
                }
                for e in sorted(
                    ir.iter_entities(), key=lambda x: x.importance_score, reverse=True
                )
            ],
            "facts": [
                {
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "importance": round(f.importance_score, 3),
                }
                for f in sorted(
                    ir.iter_facts(), key=lambda x: x.importance_score, reverse=True
                )
            ],
            "event_sequences": [
                {
                    "summary": seq.summary or self._summarize_sequence(seq),
                    "turns": f"{seq.start_turn}-{seq.end_turn}" if seq.start_turn else "N/A",
                }
                for seq in ir.event_sequences
            ],
            "events": [
                {
                    "description": e.description,
                    "turn": e.turn_index,
                    "type": e.event_type.value,
                }
                for e in sorted(ir.iter_events(), key=lambda x: x.turn_index)
            ],
        }

        return json.dumps(output, indent=2, ensure_ascii=False)

    def _describe_entity(self, entity: Entity, ir: MemoryIR) -> str:
        """Generate a brief description of an entity."""
        parts = []

        # Type-based description
        if entity.type != EntityType.OTHER:
            parts.append(entity.type.value)

        # Key attributes
        if entity.attributes:
            attr_strs = [f"{k}: {v}" for k, v in list(entity.attributes.items())[:3]]
            if attr_strs:
                parts.append(", ".join(attr_strs))

        # Associated facts
        facts = ir.get_facts_about(entity.name)
        if facts:
            top_fact = max(facts, key=lambda f: f.importance_score)
            parts.append(f"{top_fact.predicate} {top_fact.object}")

        if not parts:
            parts.append("mentioned in conversation")

        return "; ".join(parts)

    def _fact_to_sentence(self, fact: Fact) -> str:
        """Convert a fact to a natural sentence."""
        if fact.negated:
            return f"{fact.subject} does not {fact.predicate} {fact.object}."
        else:
            # Handle different fact types
            if fact.fact_type == FactType.ATTRIBUTE:
                if fact.predicate.lower() in {"is", "are", "was", "were"}:
                    return f"{fact.subject} {fact.predicate} {fact.object}."
                return f"{fact.subject} has {fact.predicate} of {fact.object}."

            elif fact.fact_type == FactType.PREFERENCE:
                return f"{fact.subject} {fact.predicate} {fact.object}."

            elif fact.fact_type == FactType.ACTION:
                return f"{fact.subject} {fact.predicate} {fact.object}."

            else:
                return f"{fact.subject} {fact.predicate} {fact.object}."

    def _summarize_sequence(self, seq: EventSequence) -> str:
        """Generate a summary for an event sequence."""
        if seq.summary:
            return seq.summary

        n = len(seq.events)
        if n == 0:
            return "Empty sequence"

        # Get dominant event type
        types = [e.event_type for e in seq.events]
        type_counts = defaultdict(int)
        for t in types:
            type_counts[t] += 1
        dominant = max(type_counts, key=type_counts.get)

        type_name = dominant.value.replace("_", " ")
        return f"Sequence of {n} {type_name} events"

    def _format_list(self, items: list[str]) -> str:
        """Format a list of items as natural language."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text."""
        # Simple approximation: ~4 characters per token for English
        return len(text) // 4


class AdaptiveContextGenerator(ContextGenerator):
    """Context generator that adapts output to fit token budget exactly.

    This generator iteratively adjusts the output until it fits within
    the specified token budget while maximizing information retention.
    """

    def __init__(
        self,
        token_budget: int = 2048,
        tolerance: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_tokens=token_budget, **kwargs)
        self.token_budget = token_budget
        self.tolerance = tolerance

        # Try to use tiktoken for accurate counting
        self._tokenizer = None
        try:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.debug("tiktoken not available, using estimation")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return self.estimate_tokens(text)

    def generate(self, ir: MemoryIR) -> str:
        """Generate context that fits within token budget."""
        # Start with full output
        text = super().generate(ir)
        tokens = self.count_tokens(text)

        if tokens <= self.token_budget:
            return text

        # Iteratively reduce until within budget
        max_iterations = 5
        for _ in range(max_iterations):
            # Calculate how much to reduce
            ratio = self.token_budget / tokens

            # Reduce limits
            self.max_tokens = int(self.max_tokens * ratio * 0.9)

            # Regenerate with stricter limits
            text = self._generate_with_limits(ir, ratio)
            tokens = self.count_tokens(text)

            if tokens <= self.token_budget * (1 + self.tolerance):
                break

        return text

    def _generate_with_limits(self, ir: MemoryIR, ratio: float) -> str:
        """Generate with reduced content limits."""
        sections = []

        # Reduce entity limit
        entity_limit = max(3, int(10 * ratio))
        entities = sorted(ir.iter_entities(), key=lambda e: e.importance_score, reverse=True)
        if entities and self.include_entities:
            lines = ["## Key Entities"]
            for entity in entities[:entity_limit]:
                lines.append(f"- **{entity.name}**")
            sections.append("\n".join(lines))

        # Reduce fact limit
        fact_limit = max(5, int(20 * ratio))
        facts = sorted(ir.iter_facts(), key=lambda f: f.importance_score, reverse=True)
        if facts and self.include_facts:
            lines = ["## Key Information"]
            for fact in facts[:fact_limit]:
                lines.append(f"- {fact.natural_language}")
            sections.append("\n".join(lines))

        # Reduce event limit
        event_limit = max(3, int(10 * ratio))
        events = sorted(ir.iter_events(), key=lambda e: e.importance_score, reverse=True)
        if events and self.include_events:
            lines = ["## Timeline"]
            for event in events[:event_limit]:
                lines.append(f"- {event.description}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
