"""Main Memory IR class that combines all IR components.

The MemoryIR class represents the complete intermediate representation
of a dialogue's memory, combining entities, facts, relations, and events.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.events import Event, EventSequence, EventType
from memory_compiler.ir.facts import Fact, FactType
from memory_compiler.ir.relations import Relation, RelationGraph, RelationType


@dataclass
class DialogueTurn:
    """Represents a single turn in the dialogue."""

    index: int
    role: str  # "user" or "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryIRStats:
    """Statistics about a MemoryIR instance."""

    num_entities: int
    num_facts: int
    num_relations: int
    num_events: int
    num_event_sequences: int
    entity_types: dict[str, int]
    fact_types: dict[str, int]
    relation_types: dict[str, int]


class MemoryIR:
    """Memory Intermediate Representation for dialogue compression.

    This class provides a structured representation of dialogue memory,
    analogous to compiler IRs. It enables semantic-preserving transformations
    for memory compression.

    Attributes:
        entities: Dictionary mapping entity IDs to Entity objects.
        facts: Dictionary mapping fact IDs to Fact objects.
        relations: RelationGraph managing fact relationships.
        events: Dictionary mapping event IDs to Event objects.
        event_sequences: List of grouped event sequences.
        dialogue_turns: Original dialogue turns for reference.
        metadata: Additional metadata about the IR.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, Fact] = {}
        self.relations: RelationGraph = RelationGraph()
        self.events: dict[str, Event] = {}
        self.event_sequences: list[EventSequence] = []
        self.dialogue_turns: list[DialogueTurn] = []
        self.metadata: dict[str, Any] = {}
        self._entity_name_index: dict[str, str] = {}  # name -> entity_id

    # ==================== Entity Operations ====================

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the IR.

        Returns:
            The entity ID.
        """
        self.entities[entity.id] = entity
        self._entity_name_index[entity.name.lower()] = entity.id
        for alias in entity.aliases:
            self._entity_name_index[alias.lower()] = entity.id
        return entity.id

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Get an entity by name or alias."""
        entity_id = self._entity_name_index.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and its associated facts."""
        if entity_id not in self.entities:
            return

        entity = self.entities[entity_id]

        # Remove from name index
        if entity.name.lower() in self._entity_name_index:
            del self._entity_name_index[entity.name.lower()]
        for alias in entity.aliases:
            if alias.lower() in self._entity_name_index:
                del self._entity_name_index[alias.lower()]

        # Remove facts referencing this entity
        facts_to_remove = [
            f.id
            for f in self.facts.values()
            if f.subject == entity.name or f.object == entity.name
        ]
        for fact_id in facts_to_remove:
            self.remove_fact(fact_id)

        del self.entities[entity_id]

    def find_or_create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.OTHER,
        **kwargs: Any,
    ) -> Entity:
        """Find an existing entity by name or create a new one."""
        existing = self.get_entity_by_name(name)
        if existing:
            return existing

        entity = Entity(
            id=f"ent_{uuid.uuid4().hex[:8]}",
            name=name,
            type=entity_type,
            **kwargs,
        )
        self.add_entity(entity)
        return entity

    # ==================== Fact Operations ====================

    def add_fact(self, fact: Fact) -> str:
        """Add a fact to the IR.

        Returns:
            The fact ID.
        """
        self.facts[fact.id] = fact
        return fact.id

    def get_fact(self, fact_id: str) -> Fact | None:
        """Get a fact by ID."""
        return self.facts.get(fact_id)

    def remove_fact(self, fact_id: str) -> None:
        """Remove a fact and its associated relations."""
        if fact_id not in self.facts:
            return

        # Remove relations involving this fact
        relations_to_remove = []
        for relation in self.relations.all_relations():
            if relation.source_fact_id == fact_id or relation.target_fact_id == fact_id:
                relations_to_remove.append(relation.id)

        for rel_id in relations_to_remove:
            self.relations.remove_relation(rel_id)

        del self.facts[fact_id]

    def get_facts_about(self, entity_name: str) -> list[Fact]:
        """Get all facts where the entity is subject or object."""
        return [
            f
            for f in self.facts.values()
            if f.subject.lower() == entity_name.lower()
            or f.object.lower() == entity_name.lower()
        ]

    def get_facts_by_type(self, fact_type: FactType) -> list[Fact]:
        """Get all facts of a specific type."""
        return [f for f in self.facts.values() if f.fact_type == fact_type]

    def create_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        fact_type: FactType = FactType.OTHER,
        **kwargs: Any,
    ) -> Fact:
        """Create and add a new fact."""
        fact = Fact(
            id=f"fact_{uuid.uuid4().hex[:8]}",
            subject=subject,
            predicate=predicate,
            object=obj,
            fact_type=fact_type,
            **kwargs,
        )
        self.add_fact(fact)
        return fact

    # ==================== Relation Operations ====================

    def add_relation(self, relation: Relation) -> str:
        """Add a relation between facts.

        Returns:
            The relation ID.
        """
        self.relations.add_relation(relation)
        return relation.id

    def create_relation(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: RelationType,
        **kwargs: Any,
    ) -> Relation:
        """Create and add a new relation."""
        relation = Relation(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            **kwargs,
        )
        self.add_relation(relation)
        return relation

    def get_related_facts(self, fact_id: str) -> set[str]:
        """Get all facts related to a given fact."""
        return self.relations.get_related_facts(fact_id)

    # ==================== Event Operations ====================

    def add_event(self, event: Event) -> str:
        """Add an event to the IR.

        Returns:
            The event ID.
        """
        self.events[event.id] = event
        return event.id

    def get_event(self, event_id: str) -> Event | None:
        """Get an event by ID."""
        return self.events.get(event_id)

    def remove_event(self, event_id: str) -> None:
        """Remove an event."""
        if event_id in self.events:
            del self.events[event_id]

    def create_event(
        self,
        description: str,
        event_type: EventType,
        turn_index: int,
        **kwargs: Any,
    ) -> Event:
        """Create and add a new event."""
        event = Event(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            description=description,
            event_type=event_type,
            turn_index=turn_index,
            **kwargs,
        )
        self.add_event(event)
        return event

    def get_events_in_range(self, start_turn: int, end_turn: int) -> list[Event]:
        """Get events within a turn range."""
        return [
            e
            for e in self.events.values()
            if start_turn <= e.turn_index <= end_turn
        ]

    # ==================== Event Sequence Operations ====================

    def add_event_sequence(self, sequence: EventSequence) -> None:
        """Add an event sequence."""
        self.event_sequences.append(sequence)

    def create_event_sequence(
        self,
        events: list[Event],
        sequence_type: str = "general",
        **kwargs: Any,
    ) -> EventSequence:
        """Create and add a new event sequence."""
        sequence = EventSequence(
            id=f"seq_{uuid.uuid4().hex[:8]}",
            events=events,
            sequence_type=sequence_type,
            **kwargs,
        )
        self.add_event_sequence(sequence)
        return sequence

    # ==================== Dialogue Operations ====================

    def add_dialogue_turn(self, turn: DialogueTurn) -> None:
        """Add a dialogue turn."""
        self.dialogue_turns.append(turn)

    def set_dialogue(self, dialogue: list[dict[str, str]]) -> None:
        """Set the dialogue from a list of role/content dictionaries."""
        self.dialogue_turns = []
        for i, turn in enumerate(dialogue):
            self.add_dialogue_turn(
                DialogueTurn(
                    index=i,
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                    metadata=turn.get("metadata", {}),
                )
            )

    # ==================== Analysis Methods ====================

    def build_entity_graph(self) -> nx.Graph:
        """Build a graph of entity co-occurrences.

        Entities are connected if they appear in the same fact.
        """
        G = nx.Graph()

        for entity in self.entities.values():
            G.add_node(entity.id, name=entity.name, type=entity.type.value)

        for fact in self.facts.values():
            subj_entity = self.get_entity_by_name(fact.subject)
            obj_entity = self.get_entity_by_name(fact.object)

            if subj_entity and obj_entity:
                if G.has_edge(subj_entity.id, obj_entity.id):
                    G[subj_entity.id][obj_entity.id]["weight"] += 1
                else:
                    G.add_edge(subj_entity.id, obj_entity.id, weight=1)

        return G

    def compute_entity_centrality(self) -> dict[str, float]:
        """Compute centrality scores for entities."""
        G = self.build_entity_graph()
        if len(G) == 0:
            return {}

        try:
            centrality = nx.pagerank(G)
        except nx.PowerIterationFailedConvergence:
            centrality = nx.degree_centrality(G)

        return centrality

    def get_stats(self) -> MemoryIRStats:
        """Get statistics about this IR."""
        entity_types: dict[str, int] = {}
        for entity in self.entities.values():
            t = entity.type.value
            entity_types[t] = entity_types.get(t, 0) + 1

        fact_types: dict[str, int] = {}
        for fact in self.facts.values():
            t = fact.fact_type.value
            fact_types[t] = fact_types.get(t, 0) + 1

        relation_types: dict[str, int] = {}
        for relation in self.relations.all_relations():
            t = relation.relation_type.value
            relation_types[t] = relation_types.get(t, 0) + 1

        return MemoryIRStats(
            num_entities=len(self.entities),
            num_facts=len(self.facts),
            num_relations=len(self.relations),
            num_events=len(self.events),
            num_event_sequences=len(self.event_sequences),
            entity_types=entity_types,
            fact_types=fact_types,
            relation_types=relation_types,
        )

    # ==================== Iteration Methods ====================

    def iter_entities(self) -> Iterator[Entity]:
        """Iterate over all entities."""
        return iter(self.entities.values())

    def iter_facts(self) -> Iterator[Fact]:
        """Iterate over all facts."""
        return iter(self.facts.values())

    def iter_events(self) -> Iterator[Event]:
        """Iterate over all events in chronological order."""
        return iter(sorted(self.events.values(), key=lambda e: e.turn_index))

    # ==================== Serialization ====================

    def to_dict(self) -> dict[str, Any]:
        """Convert IR to dictionary representation."""
        return {
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "facts": {fid: f.to_dict() for fid, f in self.facts.items()},
            "relations": [r.to_dict() for r in self.relations.all_relations()],
            "events": {eid: e.to_dict() for eid, e in self.events.items()},
            "event_sequences": [s.to_dict() for s in self.event_sequences],
            "dialogue_turns": [
                {
                    "index": t.index,
                    "role": t.role,
                    "content": t.content,
                    "metadata": t.metadata,
                }
                for t in self.dialogue_turns
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryIR:
        """Create IR from dictionary representation."""
        ir = cls()

        # Load entities
        for entity_data in data.get("entities", {}).values():
            entity = Entity.from_dict(entity_data)
            ir.add_entity(entity)

        # Load facts
        for fact_data in data.get("facts", {}).values():
            fact = Fact.from_dict(fact_data)
            ir.add_fact(fact)

        # Load relations
        for relation_data in data.get("relations", []):
            relation = Relation.from_dict(relation_data)
            ir.add_relation(relation)

        # Load events
        for event_data in data.get("events", {}).values():
            event = Event.from_dict(event_data)
            ir.add_event(event)

        # Load event sequences
        for seq_data in data.get("event_sequences", []):
            sequence = EventSequence.from_dict(seq_data)
            ir.add_event_sequence(sequence)

        # Load dialogue turns
        for turn_data in data.get("dialogue_turns", []):
            ir.add_dialogue_turn(
                DialogueTurn(
                    index=turn_data["index"],
                    role=turn_data["role"],
                    content=turn_data["content"],
                    metadata=turn_data.get("metadata", {}),
                )
            )

        ir.metadata = data.get("metadata", {})
        return ir

    def save(self, path: str | Path) -> None:
        """Save IR to a JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> MemoryIR:
        """Load IR from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def clone(self) -> MemoryIR:
        """Create a deep copy of this IR."""
        return MemoryIR.from_dict(self.to_dict())

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MemoryIR(entities={stats.num_entities}, facts={stats.num_facts}, "
            f"relations={stats.num_relations}, events={stats.num_events})"
        )
