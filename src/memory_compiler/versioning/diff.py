"""Memory IR diffing and patch operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from memory_compiler.ir.entities import Entity
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.events import Event
from memory_compiler.ir.relations import Relation
from memory_compiler.ir.memory_ir import MemoryIR


class DiffOperation(Enum):
    """Types of diff operations."""

    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


@dataclass
class DiffItem:
    """A single diff item representing a change.

    Attributes:
        operation: Type of operation (add/remove/modify).
        item_type: Type of item (entity/fact/event/relation).
        item_id: ID of the affected item.
        old_value: Previous value (for modify/remove).
        new_value: New value (for add/modify).
    """

    operation: DiffOperation
    item_type: str  # "entity", "fact", "event", "relation", "summary"
    item_id: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation.value,
            "item_type": self.item_type,
            "item_id": self.item_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiffItem":
        """Create from dictionary."""
        return cls(
            operation=DiffOperation(data["operation"]),
            item_type=data["item_type"],
            item_id=data["item_id"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
        )


@dataclass
class MemoryDiff:
    """Diff between two Memory IR versions.

    Attributes:
        from_version: Source version identifier.
        to_version: Target version identifier.
        items: List of diff items.
        metadata: Additional diff metadata.
    """

    from_version: str
    to_version: str
    items: List[DiffItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_additions(self) -> int:
        """Count additions."""
        return sum(1 for i in self.items if i.operation == DiffOperation.ADD)

    @property
    def num_removals(self) -> int:
        """Count removals."""
        return sum(1 for i in self.items if i.operation == DiffOperation.REMOVE)

    @property
    def num_modifications(self) -> int:
        """Count modifications."""
        return sum(1 for i in self.items if i.operation == DiffOperation.MODIFY)

    @property
    def is_empty(self) -> bool:
        """Check if diff is empty."""
        return len(self.items) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryDiff":
        """Create from dictionary."""
        return cls(
            from_version=data["from_version"],
            to_version=data["to_version"],
            items=[DiffItem.from_dict(i) for i in data.get("items", [])],
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryDiff":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Get human-readable summary."""
        parts = [f"Diff from {self.from_version} to {self.to_version}:"]
        parts.append(f"  Additions: {self.num_additions}")
        parts.append(f"  Removals: {self.num_removals}")
        parts.append(f"  Modifications: {self.num_modifications}")

        # Group by type
        by_type: Dict[str, List[DiffItem]] = {}
        for item in self.items:
            by_type.setdefault(item.item_type, []).append(item)

        for item_type, items in by_type.items():
            adds = sum(1 for i in items if i.operation == DiffOperation.ADD)
            removes = sum(1 for i in items if i.operation == DiffOperation.REMOVE)
            mods = sum(1 for i in items if i.operation == DiffOperation.MODIFY)
            parts.append(f"  {item_type}: +{adds} -{removes} ~{mods}")

        return "\n".join(parts)


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute hash of dictionary for comparison."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()[:8]


def _entity_to_dict(entity: Entity) -> Dict[str, Any]:
    """Convert entity to comparable dict."""
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": sorted(entity.aliases),
        "attributes": entity.attributes,
        "importance_score": entity.importance_score,
    }


def _fact_to_dict(fact: Fact) -> Dict[str, Any]:
    """Convert fact to comparable dict."""
    return {
        "id": fact.id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "confidence": fact.confidence,
        "importance_score": fact.importance_score,
    }


def _event_to_dict(event: Event) -> Dict[str, Any]:
    """Convert event to comparable dict."""
    return {
        "id": event.id,
        "description": event.description,
        "participants": sorted(event.participants),
        "importance_score": event.importance_score,
    }


def _relation_to_dict(relation: Relation) -> Dict[str, Any]:
    """Convert relation to comparable dict."""
    return {
        "source_id": relation.source_id,
        "target_id": relation.target_id,
        "relation_type": relation.relation_type.value,
        "weight": relation.weight,
    }


def compute_diff(
    old_ir: MemoryIR,
    new_ir: MemoryIR,
    old_version: str = "v1",
    new_version: str = "v2",
) -> MemoryDiff:
    """Compute diff between two Memory IRs.

    Args:
        old_ir: Original Memory IR.
        new_ir: New Memory IR.
        old_version: Version identifier for old IR.
        new_version: Version identifier for new IR.

    Returns:
        MemoryDiff representing changes.
    """
    diff = MemoryDiff(from_version=old_version, to_version=new_version)

    # Diff entities
    old_entities = {e.id: _entity_to_dict(e) for e in old_ir.iter_entities()}
    new_entities = {e.id: _entity_to_dict(e) for e in new_ir.iter_entities()}

    diff.items.extend(_diff_dicts(old_entities, new_entities, "entity"))

    # Diff facts
    old_facts = {f.id: _fact_to_dict(f) for f in old_ir.iter_facts()}
    new_facts = {f.id: _fact_to_dict(f) for f in new_ir.iter_facts()}

    diff.items.extend(_diff_dicts(old_facts, new_facts, "fact"))

    # Diff events
    old_events = {e.id: _event_to_dict(e) for e in old_ir.iter_events()}
    new_events = {e.id: _event_to_dict(e) for e in new_ir.iter_events()}

    diff.items.extend(_diff_dicts(old_events, new_events, "event"))

    # Diff relations (use composite key)
    old_relations = {}
    for r in old_ir.relations:
        key = f"{r.source_id}->{r.target_id}:{r.relation_type.value}"
        old_relations[key] = _relation_to_dict(r)

    new_relations = {}
    for r in new_ir.relations:
        key = f"{r.source_id}->{r.target_id}:{r.relation_type.value}"
        new_relations[key] = _relation_to_dict(r)

    diff.items.extend(_diff_dicts(old_relations, new_relations, "relation"))

    # Diff summaries
    old_summaries = {f"summary_{i}": {"text": s} for i, s in enumerate(old_ir.summaries)}
    new_summaries = {f"summary_{i}": {"text": s} for i, s in enumerate(new_ir.summaries)}

    diff.items.extend(_diff_dicts(old_summaries, new_summaries, "summary"))

    # Add metadata
    diff.metadata["old_entity_count"] = len(old_entities)
    diff.metadata["new_entity_count"] = len(new_entities)
    diff.metadata["old_fact_count"] = len(old_facts)
    diff.metadata["new_fact_count"] = len(new_facts)

    return diff


def _diff_dicts(
    old: Dict[str, Dict[str, Any]],
    new: Dict[str, Dict[str, Any]],
    item_type: str,
) -> List[DiffItem]:
    """Compute diff between two dictionaries."""
    items = []

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    # Removed items
    for key in old_keys - new_keys:
        items.append(
            DiffItem(
                operation=DiffOperation.REMOVE,
                item_type=item_type,
                item_id=key,
                old_value=old[key],
            )
        )

    # Added items
    for key in new_keys - old_keys:
        items.append(
            DiffItem(
                operation=DiffOperation.ADD,
                item_type=item_type,
                item_id=key,
                new_value=new[key],
            )
        )

    # Modified items
    for key in old_keys & new_keys:
        if _compute_hash(old[key]) != _compute_hash(new[key]):
            items.append(
                DiffItem(
                    operation=DiffOperation.MODIFY,
                    item_type=item_type,
                    item_id=key,
                    old_value=old[key],
                    new_value=new[key],
                )
            )

    return items


def apply_diff(ir: MemoryIR, diff: MemoryDiff) -> MemoryIR:
    """Apply a diff to a Memory IR to produce new version.

    Args:
        ir: Base Memory IR.
        diff: Diff to apply.

    Returns:
        New Memory IR with diff applied.
    """
    from memory_compiler.ir.entities import EntityType
    from memory_compiler.ir.relations import RelationType

    # Create a copy
    new_ir = MemoryIR(session_id=ir.session_id)

    # Copy existing items
    for entity in ir.iter_entities():
        new_ir.add_entity(entity)

    for fact in ir.iter_facts():
        new_ir.add_fact(fact)

    for event in ir.iter_events():
        new_ir.add_event(event)

    new_ir.relations = list(ir.relations)
    new_ir.summaries = list(ir.summaries)

    # Apply diff items
    for item in diff.items:
        if item.item_type == "entity":
            _apply_entity_diff(new_ir, item)
        elif item.item_type == "fact":
            _apply_fact_diff(new_ir, item)
        elif item.item_type == "event":
            _apply_event_diff(new_ir, item)
        elif item.item_type == "relation":
            _apply_relation_diff(new_ir, item)
        elif item.item_type == "summary":
            _apply_summary_diff(new_ir, item)

    return new_ir


def _apply_entity_diff(ir: MemoryIR, item: DiffItem) -> None:
    """Apply entity diff item."""
    from memory_compiler.ir.entities import EntityType

    if item.operation == DiffOperation.REMOVE:
        # Remove entity
        ir.entities = {
            eid: e for eid, e in ir.entities.items() if eid != item.item_id
        }

    elif item.operation == DiffOperation.ADD:
        # Add new entity
        data = item.new_value
        entity = Entity(
            id=data["id"],
            name=data["name"],
            type=EntityType(data["type"]),
            aliases=set(data.get("aliases", [])),
            attributes=data.get("attributes", {}),
            importance_score=data.get("importance_score", 0.5),
        )
        ir.add_entity(entity)

    elif item.operation == DiffOperation.MODIFY:
        # Update entity
        if item.item_id in ir.entities:
            data = item.new_value
            ir.entities[item.item_id] = Entity(
                id=data["id"],
                name=data["name"],
                type=EntityType(data["type"]),
                aliases=set(data.get("aliases", [])),
                attributes=data.get("attributes", {}),
                importance_score=data.get("importance_score", 0.5),
            )


def _apply_fact_diff(ir: MemoryIR, item: DiffItem) -> None:
    """Apply fact diff item."""
    if item.operation == DiffOperation.REMOVE:
        ir.facts = {fid: f for fid, f in ir.facts.items() if fid != item.item_id}

    elif item.operation == DiffOperation.ADD:
        data = item.new_value
        fact = Fact(
            id=data["id"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            importance_score=data.get("importance_score", 0.5),
        )
        ir.add_fact(fact)

    elif item.operation == DiffOperation.MODIFY:
        if item.item_id in ir.facts:
            data = item.new_value
            ir.facts[item.item_id] = Fact(
                id=data["id"],
                subject=data["subject"],
                predicate=data["predicate"],
                object=data["object"],
                confidence=data.get("confidence", 1.0),
                importance_score=data.get("importance_score", 0.5),
            )


def _apply_event_diff(ir: MemoryIR, item: DiffItem) -> None:
    """Apply event diff item."""
    if item.operation == DiffOperation.REMOVE:
        ir.events = {eid: e for eid, e in ir.events.items() if eid != item.item_id}

    elif item.operation == DiffOperation.ADD:
        data = item.new_value
        event = Event(
            id=data["id"],
            description=data["description"],
            participants=data.get("participants", []),
            importance_score=data.get("importance_score", 0.5),
        )
        ir.add_event(event)

    elif item.operation == DiffOperation.MODIFY:
        if item.item_id in ir.events:
            data = item.new_value
            ir.events[item.item_id] = Event(
                id=data["id"],
                description=data["description"],
                participants=data.get("participants", []),
                importance_score=data.get("importance_score", 0.5),
            )


def _apply_relation_diff(ir: MemoryIR, item: DiffItem) -> None:
    """Apply relation diff item."""
    from memory_compiler.ir.relations import RelationType

    if item.operation == DiffOperation.REMOVE:
        data = item.old_value
        ir.relations = [
            r for r in ir.relations
            if not (
                r.source_id == data["source_id"]
                and r.target_id == data["target_id"]
                and r.relation_type.value == data["relation_type"]
            )
        ]

    elif item.operation == DiffOperation.ADD:
        data = item.new_value
        relation = Relation(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data.get("weight", 1.0),
        )
        ir.add_relation(relation)


def _apply_summary_diff(ir: MemoryIR, item: DiffItem) -> None:
    """Apply summary diff item."""
    # For summaries, we handle by index
    idx = int(item.item_id.replace("summary_", ""))

    if item.operation == DiffOperation.REMOVE:
        if idx < len(ir.summaries):
            ir.summaries.pop(idx)

    elif item.operation == DiffOperation.ADD:
        text = item.new_value.get("text", "")
        ir.summaries.append(text)

    elif item.operation == DiffOperation.MODIFY:
        if idx < len(ir.summaries):
            ir.summaries[idx] = item.new_value.get("text", "")
