"""Memory extraction from dialogues to Memory IR.

This module handles the conversion of raw dialogue text into structured
Memory IR representations using LLM-based information extraction.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from loguru import logger

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.events import Event, EventType
from memory_compiler.ir.facts import Fact, FactType
from memory_compiler.ir.memory_ir import DialogueTurn, MemoryIR
from memory_compiler.ir.relations import Relation, RelationType


# Mapping from string types to enums
ENTITY_TYPE_MAP = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "location": EntityType.LOCATION,
    "project": EntityType.PROJECT,
    "concept": EntityType.CONCEPT,
    "event": EntityType.EVENT,
    "product": EntityType.PRODUCT,
    "time": EntityType.TIME,
    "quantity": EntityType.QUANTITY,
    "other": EntityType.OTHER,
}

FACT_TYPE_MAP = {
    "attribute": FactType.ATTRIBUTE,
    "relation": FactType.RELATION,
    "action": FactType.ACTION,
    "state": FactType.STATE,
    "preference": FactType.PREFERENCE,
    "belief": FactType.BELIEF,
    "intention": FactType.INTENTION,
    "experience": FactType.EXPERIENCE,
    "possession": FactType.POSSESSION,
    "membership": FactType.MEMBERSHIP,
    "other": FactType.OTHER,
}

EVENT_TYPE_MAP = {
    "action": EventType.ACTION,
    "state_change": EventType.STATE_CHANGE,
    "decision": EventType.DECISION,
    "discovery": EventType.DISCOVERY,
    "error": EventType.ERROR,
    "resolution": EventType.RESOLUTION,
    "question": EventType.QUESTION,
    "answer": EventType.ANSWER,
    "request": EventType.REQUEST,
    "completion": EventType.COMPLETION,
    "other": EventType.OTHER,
}

RELATION_TYPE_MAP = {
    "before": RelationType.BEFORE,
    "after": RelationType.AFTER,
    "during": RelationType.DURING,
    "simultaneous": RelationType.SIMULTANEOUS,
    "causes": RelationType.CAUSES,
    "caused_by": RelationType.CAUSED_BY,
    "enables": RelationType.ENABLES,
    "prevents": RelationType.PREVENTS,
    "entails": RelationType.ENTAILS,
    "contradicts": RelationType.CONTRADICTS,
    "supports": RelationType.SUPPORTS,
    "refutes": RelationType.REFUTES,
    "elaborates": RelationType.ELABORATES,
    "summarizes": RelationType.SUMMARIZES,
    "equivalent": RelationType.EQUIVALENT,
    "similar": RelationType.SIMILAR,
    "updates": RelationType.UPDATES,
    "corrects": RelationType.CORRECTS,
    "related": RelationType.RELATED,
    "other": RelationType.OTHER,
}


EXTRACTION_PROMPT = '''You are an information extraction system. Analyze the following dialogue and extract structured information.

## Dialogue:
{dialogue}

## Instructions:
Extract the following from the dialogue:

1. **Entities**: People, organizations, locations, projects, concepts, products mentioned.
2. **Facts**: Information stated as subject-predicate-object triples.
3. **Events**: Key actions, decisions, or state changes that occurred.
4. **Relations**: Relationships between facts (causal, temporal, logical).

## Output Format (JSON):
```json
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "person|organization|location|project|concept|product|time|other",
      "aliases": ["alternative names"],
      "attributes": {{"key": "value"}}
    }}
  ],
  "facts": [
    {{
      "subject": "subject entity or noun phrase",
      "predicate": "verb or relationship",
      "object": "object entity or value",
      "type": "attribute|relation|action|state|preference|belief|intention|experience|possession|membership|other",
      "confidence": 0.9,
      "source_turn": 0,
      "negated": false
    }}
  ],
  "events": [
    {{
      "description": "what happened",
      "type": "action|state_change|decision|discovery|error|resolution|question|answer|request|completion|other",
      "turn_index": 0,
      "participants": ["entity names"]
    }}
  ],
  "relations": [
    {{
      "source_fact_index": 0,
      "target_fact_index": 1,
      "type": "causes|before|after|entails|elaborates|updates|similar|other"
    }}
  ]
}}
```

Be thorough but precise. Only extract information that is explicitly stated or strongly implied.
Output ONLY the JSON, no other text.'''


class MemoryExtractor:
    """Extracts Memory IR from dialogue using LLM-based information extraction.

    This class uses a language model to convert unstructured dialogue into
    structured Memory IR representations.

    Attributes:
        model: The language model to use for extraction.
        tokenizer: Tokenizer for the model.
        batch_size: Number of turns to process together.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        batch_size: int = 10,
        use_mock: bool = False,
    ) -> None:
        """Initialize the memory extractor.

        Args:
            model_name: HuggingFace model name or path.
            device: Device for inference ("auto", "cuda", "cpu").
            batch_size: Number of dialogue turns to process together.
            use_mock: If True, use mock extraction (for testing without GPU).
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_mock = use_mock

        self.model = None
        self.tokenizer = None

        if not use_mock:
            self._load_model()

    def _load_model(self) -> None:
        """Load the language model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            device_map = self.device if self.device != "auto" else "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )

            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning("transformers not installed, using mock extraction")
            self.use_mock = True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using mock extraction")
            self.use_mock = True

    def extract(self, dialogue: list[dict[str, str]]) -> MemoryIR:
        """Extract Memory IR from a dialogue.

        Args:
            dialogue: List of dialogue turns, each with 'role' and 'content'.

        Returns:
            MemoryIR instance containing extracted information.
        """
        ir = MemoryIR()
        ir.set_dialogue(dialogue)

        # Process dialogue in batches
        for i in range(0, len(dialogue), self.batch_size):
            batch = dialogue[i : i + self.batch_size]
            batch_start_idx = i

            if self.use_mock:
                extracted = self._mock_extract(batch, batch_start_idx)
            else:
                extracted = self._llm_extract(batch, batch_start_idx)

            self._add_extracted_to_ir(ir, extracted, batch_start_idx)

        # Post-process: resolve coreferences, compute embeddings
        self._post_process(ir)

        return ir

    def _format_dialogue_for_prompt(
        self, dialogue: list[dict[str, str]], start_idx: int = 0
    ) -> str:
        """Format dialogue turns for the extraction prompt."""
        lines = []
        for i, turn in enumerate(dialogue):
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            lines.append(f"[Turn {start_idx + i}] {role}: {content}")
        return "\n".join(lines)

    def _llm_extract(
        self, dialogue: list[dict[str, str]], start_idx: int
    ) -> dict[str, Any]:
        """Use LLM to extract information from dialogue."""
        dialogue_text = self._format_dialogue_for_prompt(dialogue, start_idx)
        prompt = EXTRACTION_PROMPT.format(dialogue=dialogue_text)

        # Format for chat model
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Parse JSON from response
        return self._parse_extraction_response(response)

    def _parse_extraction_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("No JSON found in response")
                return {"entities": [], "facts": [], "events": [], "relations": []}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {"entities": [], "facts": [], "events": [], "relations": []}

    def _mock_extract(
        self, dialogue: list[dict[str, str]], start_idx: int
    ) -> dict[str, Any]:
        """Mock extraction for testing without a model.

        Uses simple heuristics to extract basic information.
        """
        entities = []
        facts = []
        events = []

        seen_entities = set()

        for i, turn in enumerate(dialogue):
            content = turn.get("content", "")
            role = turn.get("role", "user")
            turn_idx = start_idx + i

            # Simple entity extraction using capitalized words
            words = content.split()
            for j, word in enumerate(words):
                # Look for capitalized words that might be names
                clean_word = re.sub(r"[^\w]", "", word)
                if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                    if clean_word.lower() not in {"i", "the", "a", "an", "my", "your"}:
                        if clean_word not in seen_entities:
                            seen_entities.add(clean_word)
                            entities.append(
                                {
                                    "name": clean_word,
                                    "type": "person" if role == "user" else "other",
                                    "aliases": [],
                                    "attributes": {},
                                }
                            )

            # Simple fact extraction using pattern matching
            # Pattern: "X is/am/are Y"
            is_patterns = re.findall(
                r"(\w+(?:\s+\w+)?)\s+(?:is|am|are)\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
                content,
                re.IGNORECASE,
            )
            for subj, obj in is_patterns:
                facts.append(
                    {
                        "subject": subj.strip(),
                        "predicate": "is",
                        "object": obj.strip(),
                        "type": "attribute",
                        "confidence": 0.8,
                        "source_turn": turn_idx,
                        "negated": False,
                    }
                )

            # Pattern: "X works at/for Y"
            work_patterns = re.findall(
                r"(\w+)\s+works?\s+(?:at|for)\s+(\w+(?:\s+\w+)?)", content, re.IGNORECASE
            )
            for subj, obj in work_patterns:
                facts.append(
                    {
                        "subject": subj.strip(),
                        "predicate": "works at",
                        "object": obj.strip(),
                        "type": "membership",
                        "confidence": 0.9,
                        "source_turn": turn_idx,
                        "negated": False,
                    }
                )

            # Pattern: "My name is X"
            name_match = re.search(
                r"(?:my|the)\s+name\s+is\s+(\w+)", content, re.IGNORECASE
            )
            if name_match:
                facts.append(
                    {
                        "subject": "User" if role == "user" else "Assistant",
                        "predicate": "has name",
                        "object": name_match.group(1),
                        "type": "attribute",
                        "confidence": 1.0,
                        "source_turn": turn_idx,
                        "negated": False,
                    }
                )

            # Create event for questions and answers
            if "?" in content:
                events.append(
                    {
                        "description": f"Question asked: {content[:100]}",
                        "type": "question",
                        "turn_index": turn_idx,
                        "participants": [role],
                    }
                )
            elif role == "assistant" and turn_idx > 0:
                events.append(
                    {
                        "description": f"Response provided about: {content[:50]}...",
                        "type": "answer",
                        "turn_index": turn_idx,
                        "participants": ["assistant"],
                    }
                )

        return {
            "entities": entities,
            "facts": facts,
            "events": events,
            "relations": [],
        }

    def _add_extracted_to_ir(
        self, ir: MemoryIR, extracted: dict[str, Any], batch_start_idx: int
    ) -> None:
        """Add extracted information to the Memory IR."""
        fact_id_map = {}  # Map from index to fact_id for relation creation

        # Add entities
        for entity_data in extracted.get("entities", []):
            entity = Entity(
                id=f"ent_{uuid.uuid4().hex[:8]}",
                name=entity_data.get("name", "Unknown"),
                type=ENTITY_TYPE_MAP.get(
                    entity_data.get("type", "other").lower(), EntityType.OTHER
                ),
                aliases=entity_data.get("aliases", []),
                attributes=entity_data.get("attributes", {}),
            )
            ir.add_entity(entity)

        # Add facts
        for i, fact_data in enumerate(extracted.get("facts", [])):
            source_turns = fact_data.get("source_turns", [])
            if not source_turns and "source_turn" in fact_data:
                source_turns = [fact_data["source_turn"]]

            fact = Fact(
                id=f"fact_{uuid.uuid4().hex[:8]}",
                subject=fact_data.get("subject", ""),
                predicate=fact_data.get("predicate", ""),
                object=fact_data.get("object", ""),
                fact_type=FACT_TYPE_MAP.get(
                    fact_data.get("type", "other").lower(), FactType.OTHER
                ),
                confidence=fact_data.get("confidence", 1.0),
                source_turns=source_turns,
                negated=fact_data.get("negated", False),
            )
            ir.add_fact(fact)
            fact_id_map[i] = fact.id

        # Add events
        for event_data in extracted.get("events", []):
            event = Event(
                id=f"evt_{uuid.uuid4().hex[:8]}",
                description=event_data.get("description", ""),
                event_type=EVENT_TYPE_MAP.get(
                    event_data.get("type", "other").lower(), EventType.OTHER
                ),
                turn_index=event_data.get("turn_index", batch_start_idx),
                participants=event_data.get("participants", []),
            )
            ir.add_event(event)

        # Add relations
        for rel_data in extracted.get("relations", []):
            source_idx = rel_data.get("source_fact_index")
            target_idx = rel_data.get("target_fact_index")

            if source_idx in fact_id_map and target_idx in fact_id_map:
                relation = Relation(
                    id=f"rel_{uuid.uuid4().hex[:8]}",
                    source_fact_id=fact_id_map[source_idx],
                    target_fact_id=fact_id_map[target_idx],
                    relation_type=RELATION_TYPE_MAP.get(
                        rel_data.get("type", "related").lower(), RelationType.RELATED
                    ),
                )
                ir.add_relation(relation)

    def _post_process(self, ir: MemoryIR) -> None:
        """Post-process the IR after extraction.

        - Resolve entity coreferences
        - Link facts to entities
        - Compute initial importance scores
        """
        # Link facts to entities by updating entity mentions
        for fact in ir.iter_facts():
            for turn_idx in fact.source_turns:
                # Check if subject is an entity
                subj_entity = ir.get_entity_by_name(fact.subject)
                if subj_entity:
                    subj_entity.add_mention(turn_idx, (0, len(fact.subject)))

                # Check if object is an entity
                obj_entity = ir.get_entity_by_name(fact.object)
                if obj_entity:
                    obj_entity.add_mention(turn_idx, (0, len(fact.object)))

        # Compute initial importance scores based on reference counts
        for entity in ir.iter_entities():
            entity.importance_score = min(1.0, entity.reference_count * 0.1)

        for fact in ir.iter_facts():
            # Base importance from confidence
            fact.importance_score = fact.confidence * 0.5

            # Boost for facts about important entities
            subj_entity = ir.get_entity_by_name(fact.subject)
            if subj_entity:
                fact.importance_score += subj_entity.importance_score * 0.3

            # Boost for facts with more source turns (repeated information)
            fact.importance_score += min(0.2, len(fact.source_turns) * 0.05)

        for event in ir.iter_events():
            # Base importance
            event.importance_score = 0.5

            # Boost for events with more participants
            event.importance_score += min(0.3, len(event.participants) * 0.1)

    def extract_incremental(
        self, ir: MemoryIR, new_turns: list[dict[str, str]]
    ) -> MemoryIR:
        """Incrementally add new turns to an existing IR.

        Args:
            ir: Existing Memory IR.
            new_turns: New dialogue turns to add.

        Returns:
            Updated Memory IR.
        """
        start_idx = len(ir.dialogue_turns)

        # Add new turns to dialogue
        for i, turn in enumerate(new_turns):
            ir.add_dialogue_turn(
                DialogueTurn(
                    index=start_idx + i,
                    role=turn.get("role", "user"),
                    content=turn.get("content", ""),
                )
            )

        # Extract from new turns
        if self.use_mock:
            extracted = self._mock_extract(new_turns, start_idx)
        else:
            extracted = self._llm_extract(new_turns, start_idx)

        self._add_extracted_to_ir(ir, extracted, start_idx)
        self._post_process(ir)

        return ir
