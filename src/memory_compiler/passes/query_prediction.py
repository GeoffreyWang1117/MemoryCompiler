"""Query prediction for intelligent memory pruning.

This module provides query prediction capabilities to guide memory
retention decisions. By predicting likely future queries, the system
can prioritize keeping information most likely to be needed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from loguru import logger

from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import OptimizationPass, PassResult


class QueryPredictor:
    """Predicts likely future queries based on dialogue context.

    This predictor uses multiple signals to estimate the probability
    distribution over potential future queries:
    - Entity salience (frequently mentioned entities)
    - Recent topic focus
    - Question patterns in dialogue
    - Task/goal inference

    Can be used with or without LLM for prediction generation.
    """

    def __init__(
        self,
        use_llm: bool = False,
        model_name: str | None = None,
        num_predictions: int = 10,
        embedding_manager: Any = None,
    ) -> None:
        self.use_llm = use_llm
        self.model_name = model_name
        self.num_predictions = num_predictions
        self.embedding_manager = embedding_manager

        self._model = None
        self._tokenizer = None

        if use_llm and model_name:
            self._load_model()

    def _load_model(self) -> None:
        """Load LLM for query generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info(f"Loaded query prediction model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load LLM for query prediction: {e}")
            self.use_llm = False

    def predict(
        self,
        ir: MemoryIR,
        num_queries: int | None = None,
    ) -> list[PredictedQuery]:
        """Predict likely future queries.

        Args:
            ir: Memory IR representing the conversation.
            num_queries: Number of queries to predict.

        Returns:
            List of predicted queries with relevance scores.
        """
        num_queries = num_queries or self.num_predictions

        if self.use_llm and self._model is not None:
            return self._predict_with_llm(ir, num_queries)
        else:
            return self._predict_heuristic(ir, num_queries)

    def _predict_heuristic(
        self, ir: MemoryIR, num_queries: int
    ) -> list[PredictedQuery]:
        """Predict queries using heuristics."""
        queries = []

        # Extract salient entities
        entities = sorted(
            ir.iter_entities(),
            key=lambda e: e.importance_score,
            reverse=True,
        )[:10]

        # Generate entity-focused queries
        for entity in entities[:5]:
            # What queries about this entity
            queries.append(PredictedQuery(
                query=f"What is {entity.name}?",
                relevance=entity.importance_score * 0.8,
                query_type="entity_info",
                target_entities=[entity.name],
            ))

            # Facts about entity
            facts = ir.get_facts_about(entity.name)
            if facts:
                queries.append(PredictedQuery(
                    query=f"What do we know about {entity.name}?",
                    relevance=entity.importance_score * 0.7,
                    query_type="entity_facts",
                    target_entities=[entity.name],
                ))

        # Extract recent topics from dialogue
        recent_turns = ir.dialogue_turns[-5:] if ir.dialogue_turns else []
        topic_words = self._extract_topic_words(recent_turns)

        for word, count in topic_words.most_common(5):
            queries.append(PredictedQuery(
                query=f"What about {word}?",
                relevance=min(1.0, count * 0.15),
                query_type="topic",
                keywords=[word],
            ))

        # Look for unresolved questions
        for turn in ir.dialogue_turns:
            if "?" in turn.content and turn.role == "user":
                # Extract question
                questions = re.findall(r"[^.!?]*\?", turn.content)
                for q in questions[:2]:
                    queries.append(PredictedQuery(
                        query=q.strip(),
                        relevance=0.6,
                        query_type="follow_up",
                        source_turn=turn.index,
                    ))

        # Deduplicate and sort
        seen = set()
        unique_queries = []
        for q in queries:
            if q.query.lower() not in seen:
                seen.add(q.query.lower())
                unique_queries.append(q)

        unique_queries.sort(key=lambda x: x.relevance, reverse=True)
        return unique_queries[:num_queries]

    def _predict_with_llm(
        self, ir: MemoryIR, num_queries: int
    ) -> list[PredictedQuery]:
        """Predict queries using LLM."""
        # Build context
        context = self._build_context(ir)

        prompt = f"""Based on the following conversation context, predict {num_queries} questions that the user might ask next.

Context:
{context}

Generate {num_queries} likely follow-up questions, one per line. Only output the questions, nothing else.
"""

        # Generate with LLM
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
        )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse generated questions
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and "?" in line:
                # Clean up numbering
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                queries.append(PredictedQuery(
                    query=line,
                    relevance=0.7,
                    query_type="llm_predicted",
                ))

        return queries[:num_queries]

    def _build_context(self, ir: MemoryIR) -> str:
        """Build context string for LLM."""
        lines = []

        # Recent dialogue
        recent = ir.dialogue_turns[-10:]
        for turn in recent:
            lines.append(f"{turn.role.upper()}: {turn.content}")

        # Key entities
        entities = sorted(
            ir.iter_entities(),
            key=lambda e: e.importance_score,
            reverse=True,
        )[:5]

        if entities:
            lines.append("\nKey entities mentioned:")
            for e in entities:
                lines.append(f"- {e.name} ({e.type.value})")

        return "\n".join(lines)

    def _extract_topic_words(self, turns: list) -> Counter:
        """Extract topic words from dialogue turns."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "whom", "whose",
            "where", "when", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "but", "and", "or", "if", "then", "else", "for",
            "of", "to", "from", "by", "on", "at", "in", "out", "up",
            "down", "with", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "further",
            "once", "here", "there", "where", "when", "why", "how", "any",
            "hi", "hello", "thanks", "thank", "please", "yes", "no", "ok",
        }

        word_counts: Counter = Counter()

        for turn in turns:
            words = re.findall(r"\b\w+\b", turn.content.lower())
            for word in words:
                if word not in stopwords and len(word) > 2:
                    word_counts[word] += 1

        return word_counts


class PredictedQuery:
    """Represents a predicted future query."""

    def __init__(
        self,
        query: str,
        relevance: float = 0.5,
        query_type: str = "general",
        target_entities: list[str] | None = None,
        keywords: list[str] | None = None,
        source_turn: int | None = None,
    ) -> None:
        self.query = query
        self.relevance = relevance
        self.query_type = query_type
        self.target_entities = target_entities or []
        self.keywords = keywords or []
        self.source_turn = source_turn

    def __repr__(self) -> str:
        return f"PredictedQuery('{self.query[:50]}...', relevance={self.relevance:.2f})"


class QueryPredictionPruningPass(OptimizationPass):
    """Prune memory based on predicted query relevance.

    This pass uses query prediction to estimate which memory units
    are most likely to be needed for future queries, and prunes
    those with low predicted relevance.

    Configuration options:
        relevance_threshold: Minimum relevance score to keep.
        num_predicted_queries: Number of queries to predict.
        use_llm: Whether to use LLM for prediction.
    """

    def __init__(
        self,
        relevance_threshold: float = 0.3,
        num_predicted_queries: int = 10,
        use_llm: bool = False,
        model_name: str | None = None,
        embedding_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.relevance_threshold = relevance_threshold
        self.num_predicted_queries = num_predicted_queries

        self.predictor = QueryPredictor(
            use_llm=use_llm,
            model_name=model_name,
            num_predictions=num_predicted_queries,
            embedding_manager=embedding_manager,
        )
        self.embedding_manager = embedding_manager

    def run(self, ir: MemoryIR) -> PassResult:
        """Run query-prediction-based pruning."""
        result = PassResult()

        # Predict queries
        predicted_queries = self.predictor.predict(ir, self.num_predicted_queries)
        result.stats["predicted_queries"] = len(predicted_queries)

        if not predicted_queries:
            logger.warning("No queries predicted, skipping pass")
            return result

        # Compute relevance scores for each fact
        fact_relevance = self._compute_fact_relevance(ir, predicted_queries)
        result.stats["facts_scored"] = len(fact_relevance)

        # Identify facts to remove
        facts_to_remove = [
            fact_id
            for fact_id, relevance in fact_relevance.items()
            if relevance < self.relevance_threshold
        ]

        # Also consider entity relevance
        entity_relevance = self._compute_entity_relevance(ir, predicted_queries)
        entities_to_remove = [
            ent_id
            for ent_id, relevance in entity_relevance.items()
            if relevance < self.relevance_threshold
        ]

        # Remove low-relevance items
        for entity_id in entities_to_remove:
            ir.remove_entity(entity_id)
            result.entities_removed += 1

        for fact_id in facts_to_remove:
            if fact_id in ir.facts:
                ir.remove_fact(fact_id)
                result.facts_removed += 1

        result.modified = result.entities_removed > 0 or result.facts_removed > 0

        logger.info(
            f"QueryPredictionPruning: removed {result.entities_removed} entities, "
            f"{result.facts_removed} facts based on {len(predicted_queries)} predicted queries"
        )

        return result

    def _compute_fact_relevance(
        self,
        ir: MemoryIR,
        queries: list[PredictedQuery],
    ) -> dict[str, float]:
        """Compute relevance scores for facts based on predicted queries."""
        relevance = {}

        for fact in ir.iter_facts():
            max_score = 0.0

            for query in queries:
                score = self._fact_query_relevance(fact, query)
                max_score = max(max_score, score * query.relevance)

            relevance[fact.id] = max_score

        return relevance

    def _compute_entity_relevance(
        self,
        ir: MemoryIR,
        queries: list[PredictedQuery],
    ) -> dict[str, float]:
        """Compute relevance scores for entities based on predicted queries."""
        relevance = {}

        for entity in ir.iter_entities():
            max_score = 0.0

            for query in queries:
                score = self._entity_query_relevance(entity, query)
                max_score = max(max_score, score * query.relevance)

            relevance[entity.id] = max_score

        return relevance

    def _fact_query_relevance(self, fact: Fact, query: PredictedQuery) -> float:
        """Compute relevance of a fact to a predicted query."""
        score = 0.0

        # Check entity overlap
        for entity in query.target_entities:
            if entity.lower() in fact.subject.lower():
                score += 0.5
            if entity.lower() in fact.object.lower():
                score += 0.3

        # Check keyword overlap
        fact_words = set(fact.natural_language.lower().split())
        query_words = set(query.query.lower().split())

        overlap = len(fact_words & query_words)
        if overlap > 0:
            score += min(0.4, overlap * 0.1)

        # Use embeddings if available
        if self.embedding_manager:
            try:
                similarity = self.embedding_manager.similarity(
                    fact.natural_language, query.query
                )
                score = max(score, similarity)
            except Exception:
                pass

        return min(1.0, score)

    def _entity_query_relevance(self, entity: Any, query: PredictedQuery) -> float:
        """Compute relevance of an entity to a predicted query."""
        score = 0.0

        # Direct mention in query target entities
        if entity.name in query.target_entities:
            return 1.0

        # Name appears in query
        if entity.name.lower() in query.query.lower():
            score += 0.8

        # Alias appears in query
        for alias in entity.aliases:
            if alias.lower() in query.query.lower():
                score += 0.6

        # Keyword overlap
        entity_words = {entity.name.lower()} | {a.lower() for a in entity.aliases}
        query_words = set(query.query.lower().split())

        if entity_words & query_words:
            score += 0.5

        return min(1.0, score)
