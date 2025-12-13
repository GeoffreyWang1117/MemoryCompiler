"""Multi-lingual entity and fact extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger

from memory_compiler.ir.entities import Entity, EntityType
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.multilang.detector import (
    Language,
    LanguageDetector,
    DetectionResult,
)
from memory_compiler.multilang.tokenizer import (
    MultilingualTokenizer,
    SpacyTokenizer,
    Token,
)


@dataclass
class LanguageConfig:
    """Configuration for language-specific extraction.

    Attributes:
        language: Language code.
        person_patterns: Patterns for person names.
        location_patterns: Patterns for locations.
        org_patterns: Patterns for organizations.
        date_patterns: Patterns for dates.
        relation_patterns: Patterns for extracting relations.
        stopwords: Words to ignore.
    """

    language: str
    person_patterns: List[str] = field(default_factory=list)
    location_patterns: List[str] = field(default_factory=list)
    org_patterns: List[str] = field(default_factory=list)
    date_patterns: List[str] = field(default_factory=list)
    relation_patterns: Dict[str, str] = field(default_factory=dict)
    stopwords: Set[str] = field(default_factory=set)

    @classmethod
    def for_english(cls) -> "LanguageConfig":
        """Get English configuration."""
        return cls(
            language="en",
            person_patterns=[
                r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b",
            ],
            location_patterns=[
                r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            ],
            date_patterns=[
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s*\d{4})?\b",
            ],
            relation_patterns={
                "works_at": r"(\w+)\s+works?\s+at\s+(\w+)",
                "lives_in": r"(\w+)\s+lives?\s+in\s+(\w+)",
                "is_a": r"(\w+)\s+is\s+(?:a|an)\s+(\w+)",
                "has": r"(\w+)\s+has\s+(?:a|an)?\s*(\w+)",
            },
            stopwords={
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "must", "shall",
            },
        )

    @classmethod
    def for_chinese(cls) -> "LanguageConfig":
        """Get Chinese configuration."""
        return cls(
            language="zh",
            person_patterns=[
                r"[李王张刘陈杨黄赵周吴][一-龥]{1,2}",
                r"(?:先生|女士|小姐)\s*[一-龥]{2,3}",
            ],
            location_patterns=[
                r"在\s*([一-龥]{2,6}(?:省|市|区|县|镇|村))",
                r"来自\s*([一-龥]{2,6})",
            ],
            date_patterns=[
                r"\d{4}年\d{1,2}月\d{1,2}日",
                r"\d{1,2}月\d{1,2}日",
            ],
            relation_patterns={
                "works_at": r"([一-龥]+)\s*在\s*([一-龥]+)\s*工作",
                "lives_in": r"([一-龥]+)\s*住在\s*([一-龥]+)",
                "is_a": r"([一-龥]+)\s*是\s*(?:一个|一名)?\s*([一-龥]+)",
            },
            stopwords={
                "的", "是", "在", "了", "和", "与", "或", "等", "这", "那",
                "个", "些", "不", "也", "都", "就", "还", "很", "但", "而",
            },
        )

    @classmethod
    def for_spanish(cls) -> "LanguageConfig":
        """Get Spanish configuration."""
        return cls(
            language="es",
            person_patterns=[
                r"\b(?:Sr|Sra|Srta|Dr|Dra)\.?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*",
                r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,2}\b",
            ],
            location_patterns=[
                r"\ben\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)",
                r"\bde\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)",
            ],
            date_patterns=[
                r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b",
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            ],
            relation_patterns={
                "works_at": r"(\w+)\s+trabaja\s+en\s+(\w+)",
                "lives_in": r"(\w+)\s+vive\s+en\s+(\w+)",
                "is_a": r"(\w+)\s+es\s+(?:un|una)\s+(\w+)",
            },
            stopwords={
                "el", "la", "los", "las", "un", "una", "de", "del", "en",
                "y", "o", "que", "es", "son", "por", "para", "con", "como",
            },
        )

    @classmethod
    def for_japanese(cls) -> "LanguageConfig":
        """Get Japanese configuration."""
        return cls(
            language="ja",
            person_patterns=[
                r"[一-龥]{1,4}(?:さん|様|氏|君)",
                r"[ァ-ヴー]+(?:さん|様)",
            ],
            location_patterns=[
                r"([一-龥]{2,6}(?:都|道|府|県|市|区|町|村))",
            ],
            date_patterns=[
                r"\d{4}年\d{1,2}月\d{1,2}日",
                r"(?:令和|平成|昭和)\d{1,2}年",
            ],
            relation_patterns={
                "works_at": r"([一-龥ァ-ヴー]+)は\s*([一-龥ァ-ヴー]+)で働",
                "lives_in": r"([一-龥ァ-ヴー]+)は\s*([一-龥ァ-ヴー]+)に住",
            },
            stopwords={
                "の", "は", "が", "を", "に", "で", "と", "も", "や", "など",
                "です", "ます", "だ", "った", "ている", "される",
            },
        )

    @classmethod
    def get_config(cls, language: str) -> "LanguageConfig":
        """Get configuration for language.

        Args:
            language: Language code.

        Returns:
            Language configuration.
        """
        configs = {
            "en": cls.for_english,
            "zh": cls.for_chinese,
            "es": cls.for_spanish,
            "ja": cls.for_japanese,
        }

        factory = configs.get(language, cls.for_english)
        return factory()


class MultilingualExtractor:
    """Extract entities and facts from multi-lingual text.

    Supports automatic language detection and language-specific
    extraction patterns.

    Example:
        >>> extractor = MultilingualExtractor()
        >>> ir = extractor.extract("张三在北京工作。他喜欢编程。")
        >>> for entity in ir.iter_entities():
        ...     print(f"{entity.name}: {entity.type}")
    """

    def __init__(
        self,
        default_language: str = "en",
        auto_detect: bool = True,
        use_spacy: bool = True,
        custom_configs: Optional[Dict[str, LanguageConfig]] = None,
    ) -> None:
        """Initialize extractor.

        Args:
            default_language: Default language.
            auto_detect: Auto-detect language.
            use_spacy: Use spaCy for NER.
            custom_configs: Custom language configurations.
        """
        self.default_language = default_language
        self.auto_detect = auto_detect
        self.use_spacy = use_spacy

        self._detector = LanguageDetector()
        self._tokenizer = MultilingualTokenizer(
            default_language=default_language,
            use_spacy=use_spacy,
        )

        # Language configs
        self._configs: Dict[str, LanguageConfig] = {}
        if custom_configs:
            self._configs.update(custom_configs)

        # spaCy extractors per language
        self._spacy_extractors: Dict[str, SpacyTokenizer] = {}

    def _get_config(self, language: str) -> LanguageConfig:
        """Get configuration for language."""
        if language not in self._configs:
            self._configs[language] = LanguageConfig.get_config(language)
        return self._configs[language]

    def _get_spacy_extractor(self, language: str) -> Optional[SpacyTokenizer]:
        """Get spaCy extractor for language."""
        if not self.use_spacy:
            return None

        if language not in self._spacy_extractors:
            try:
                self._spacy_extractors[language] = SpacyTokenizer(language)
            except Exception as e:
                logger.warning(f"Could not initialize spaCy for {language}: {e}")
                self._spacy_extractors[language] = None

        return self._spacy_extractors[language]

    def extract(
        self,
        text: str,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> MemoryIR:
        """Extract entities and facts from text.

        Args:
            text: Text to extract from.
            language: Override language detection.
            session_id: Session identifier.

        Returns:
            Memory IR with extracted data.
        """
        # Detect language if needed
        if language is None and self.auto_detect:
            detection = self._detector.detect(text)
            language = detection.language.value
        elif language is None:
            language = self.default_language

        # Create IR
        ir = MemoryIR(session_id=session_id or f"multilang_{language}")
        ir.metadata["source_language"] = language

        # Extract entities
        entities = self._extract_entities(text, language)
        for entity in entities:
            ir.add_entity(entity)

        # Extract facts
        facts = self._extract_facts(text, language, entities)
        for fact in facts:
            ir.add_fact(fact)

        return ir

    def _extract_entities(
        self,
        text: str,
        language: str,
    ) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        seen_names: Set[str] = set()

        # Try spaCy first
        spacy_ext = self._get_spacy_extractor(language)
        if spacy_ext:
            spacy_entities = spacy_ext.extract_entities(text)
            for ent_text, ent_type, start, end in spacy_entities:
                if ent_text.lower() not in seen_names:
                    entity = self._create_entity(ent_text, ent_type, start)
                    if entity:
                        entities.append(entity)
                        seen_names.add(ent_text.lower())

        # Supplement with pattern-based extraction
        config = self._get_config(language)
        pattern_entities = self._extract_pattern_entities(text, config)

        for entity in pattern_entities:
            if entity.name.lower() not in seen_names:
                entities.append(entity)
                seen_names.add(entity.name.lower())

        return entities

    def _create_entity(
        self,
        name: str,
        ner_type: str,
        mention_position: int,
    ) -> Optional[Entity]:
        """Create entity from NER result."""
        # Map NER types to EntityType
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "DATE": EntityType.TIME,
            "TIME": EntityType.TIME,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.OBJECT,
            "WORK_OF_ART": EntityType.CONCEPT,
        }

        entity_type = type_mapping.get(ner_type, EntityType.OTHER)

        return Entity(
            name=name,
            type=entity_type,
            mentions=[mention_position],
            importance_score=0.6,
        )

    def _extract_pattern_entities(
        self,
        text: str,
        config: LanguageConfig,
    ) -> List[Entity]:
        """Extract entities using patterns."""
        entities = []

        # Person patterns
        for pattern in config.person_patterns:
            for match in re.finditer(pattern, text):
                name = match.group().strip()
                if len(name) > 1:
                    entities.append(Entity(
                        name=name,
                        type=EntityType.PERSON,
                        mentions=[match.start()],
                        importance_score=0.7,
                    ))

        # Location patterns
        for pattern in config.location_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1) if match.groups() else match.group()
                name = name.strip()
                if len(name) > 1:
                    entities.append(Entity(
                        name=name,
                        type=EntityType.LOCATION,
                        mentions=[match.start()],
                        importance_score=0.5,
                    ))

        # Date patterns
        for pattern in config.date_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    name=match.group(),
                    type=EntityType.TIME,
                    mentions=[match.start()],
                    importance_score=0.4,
                ))

        return entities

    def _extract_facts(
        self,
        text: str,
        language: str,
        entities: List[Entity],
    ) -> List[Fact]:
        """Extract facts from text."""
        facts = []
        config = self._get_config(language)

        # Entity name lookup
        entity_names = {e.name.lower() for e in entities}

        # Pattern-based fact extraction
        for predicate, pattern in config.relation_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) >= 2:
                    subject = match.group(1).strip()
                    obj = match.group(2).strip()

                    # Validate against entities
                    if subject.lower() in entity_names or obj.lower() in entity_names:
                        facts.append(Fact(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            confidence=0.7,
                            source_turn=0,
                        ))

        # Extract simple attribute facts
        attr_facts = self._extract_attribute_facts(text, entities, config)
        facts.extend(attr_facts)

        return facts

    def _extract_attribute_facts(
        self,
        text: str,
        entities: List[Entity],
        config: LanguageConfig,
    ) -> List[Fact]:
        """Extract attribute facts about entities."""
        facts = []

        # Simple patterns for attributes
        attr_patterns = {
            "en": [
                (r"(\w+)'s\s+(\w+)\s+is\s+(\w+)", 1, 2, 3),  # X's Y is Z
                (r"(\w+)\s+has\s+(\w+)\s+(\w+)", 1, "has_" + "2", 3),
            ],
            "zh": [
                (r"([一-龥]+)的([一-龥]+)是([一-龥]+)", 1, 2, 3),
            ],
        }

        patterns = attr_patterns.get(config.language, attr_patterns["en"])
        entity_names = {e.name.lower(): e.name for e in entities}

        for pattern_tuple in patterns:
            pattern = pattern_tuple[0]
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    subject = match.group(1).strip()
                    predicate = match.group(2).strip()
                    obj = match.group(3).strip()

                    # Check if subject is a known entity
                    if subject.lower() in entity_names:
                        facts.append(Fact(
                            subject=entity_names[subject.lower()],
                            predicate=f"has_{predicate}",
                            object=obj,
                            confidence=0.6,
                        ))
                except Exception:
                    continue

        return facts

    def extract_multilingual(
        self,
        text: str,
        session_id: Optional[str] = None,
    ) -> MemoryIR:
        """Extract from mixed-language text.

        Detects and processes each language segment separately.

        Args:
            text: Mixed-language text.
            session_id: Session identifier.

        Returns:
            Merged Memory IR.
        """
        # Detect language segments
        segments = self._detector.detect_segments(text)

        if len(segments) <= 1:
            # Single language
            return self.extract(text, session_id=session_id)

        # Process each segment
        irs = []
        for segment_text, detection in segments:
            ir = self.extract(
                segment_text,
                language=detection.language.value,
            )
            irs.append(ir)

        # Merge IRs
        if not irs:
            return MemoryIR(session_id=session_id or "multilang_mixed")

        merged = irs[0]
        for ir in irs[1:]:
            for entity in ir.iter_entities():
                merged.add_entity(entity)
            for fact in ir.iter_facts():
                merged.add_fact(fact)

        merged.session_id = session_id or "multilang_mixed"
        merged.metadata["is_multilingual"] = True
        merged.metadata["languages"] = list(set(
            d.language.value for _, d in segments
        ))

        return merged


class CrossLingualResolver:
    """Resolve entities across languages.

    Identifies when entities in different languages
    refer to the same real-world entity.

    Example:
        >>> resolver = CrossLingualResolver()
        >>> matches = resolver.find_matches("Beijing", "北京")
        >>> print(matches)  # [(('Beijing', 'en'), ('北京', 'zh'), 0.95)]
    """

    # Known cross-lingual entity mappings
    KNOWN_MAPPINGS = {
        # Cities
        ("beijing", "en"): [("北京", "zh"), ("ペキン", "ja"), ("베이징", "ko")],
        ("shanghai", "en"): [("上海", "zh"), ("シャンハイ", "ja")],
        ("tokyo", "en"): [("東京", "ja"), ("东京", "zh")],
        ("paris", "en"): [("巴黎", "zh"), ("パリ", "ja")],

        # Countries
        ("china", "en"): [("中国", "zh"), ("中國", "zh"), ("中国", "ja")],
        ("japan", "en"): [("日本", "zh"), ("日本", "ja")],
        ("america", "en"): [("美国", "zh"), ("アメリカ", "ja")],
    }

    def __init__(
        self,
        use_translation: bool = False,
        similarity_threshold: float = 0.8,
    ) -> None:
        """Initialize resolver.

        Args:
            use_translation: Use translation API for matching.
            similarity_threshold: Minimum similarity for match.
        """
        self.use_translation = use_translation
        self.similarity_threshold = similarity_threshold

        # Build reverse lookup
        self._reverse_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for key, values in self.KNOWN_MAPPINGS.items():
            for val in values:
                if val not in self._reverse_mappings:
                    self._reverse_mappings[val] = []
                self._reverse_mappings[val].append(key)

    def find_matches(
        self,
        entity1: str,
        entity2: str,
        lang1: str = "en",
        lang2: str = "zh",
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str], float]]:
        """Find cross-lingual matches between entities.

        Args:
            entity1: First entity name.
            entity2: Second entity name.
            lang1: Language of first entity.
            lang2: Language of second entity.

        Returns:
            List of ((entity1, lang1), (entity2, lang2), score) matches.
        """
        matches = []

        # Check known mappings
        key1 = (entity1.lower(), lang1)
        key2 = (entity2.lower(), lang2)

        if key1 in self.KNOWN_MAPPINGS:
            for mapped in self.KNOWN_MAPPINGS[key1]:
                if mapped[0].lower() == entity2.lower() and mapped[1] == lang2:
                    matches.append((
                        (entity1, lang1),
                        (entity2, lang2),
                        0.95,
                    ))

        if key2 in self._reverse_mappings:
            for mapped in self._reverse_mappings[key2]:
                if mapped[0].lower() == entity1.lower() and mapped[1] == lang1:
                    matches.append((
                        (entity1, lang1),
                        (entity2, lang2),
                        0.95,
                    ))

        return matches

    def resolve_entities(
        self,
        entities: List[Tuple[str, str]],
    ) -> List[Set[Tuple[str, str]]]:
        """Cluster entities that refer to the same thing.

        Args:
            entities: List of (entity_name, language) tuples.

        Returns:
            List of entity clusters.
        """
        clusters: List[Set[Tuple[str, str]]] = []
        assigned: Set[Tuple[str, str]] = set()

        for i, ent1 in enumerate(entities):
            if ent1 in assigned:
                continue

            cluster = {ent1}
            assigned.add(ent1)

            for ent2 in entities[i + 1:]:
                if ent2 in assigned:
                    continue

                matches = self.find_matches(
                    ent1[0], ent2[0],
                    ent1[1], ent2[1],
                )

                if matches:
                    cluster.add(ent2)
                    assigned.add(ent2)

            clusters.append(cluster)

        return clusters
