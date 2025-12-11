"""Fact Folding pass.

This pass merges semantically equivalent or redundant facts, similar to
constant folding and common subexpression elimination in compilers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger

from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.relations import RelationType
from memory_compiler.passes.base import OptimizationPass, PassResult


class FactFoldingPass(OptimizationPass):
    """Folds semantically equivalent or redundant facts.

    This pass identifies facts that:
    1. Are semantically equivalent (same meaning, different wording)
    2. Have entailment relationships (one implies the other)
    3. Are redundant given other facts

    Configuration options:
        similarity_threshold: Minimum similarity for facts to be considered
            equivalent (default: 0.85).
        use_embeddings: Whether to use embedding-based similarity
            (default: True).
        fold_entailed: Whether to fold facts with entailment relations
            (default: True).
        merge_strategy: How to merge facts - "keep_first", "keep_highest_confidence",
            or "combine" (default: "keep_highest_confidence").
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_embeddings: bool = True,
        fold_entailed: bool = True,
        merge_strategy: str = "keep_highest_confidence",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        self.fold_entailed = fold_entailed
        self.merge_strategy = merge_strategy
        self.embedding_model_name = embedding_model

        self._embedder = None
        if use_embeddings:
            self._load_embedder()

    def _load_embedder(self) -> None:
        """Load the sentence embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using string matching")
            self.use_embeddings = False
        except Exception as e:
            logger.warning(f"Failed to load embedder: {e}, using string matching")
            self.use_embeddings = False

    def run(self, ir: MemoryIR) -> PassResult:
        """Run fact folding on the IR."""
        result = PassResult()

        # Step 1: Compute fact embeddings if using embedding-based similarity
        if self.use_embeddings and self._embedder is not None:
            self._compute_embeddings(ir)

        # Step 2: Find equivalent fact groups
        equivalence_groups = self._find_equivalent_facts(ir)
        result.stats["equivalence_groups"] = len(equivalence_groups)

        # Step 3: Fold entailed facts
        if self.fold_entailed:
            entailment_pairs = self._find_entailment_pairs(ir)
            result.stats["entailment_pairs"] = len(entailment_pairs)
        else:
            entailment_pairs = []

        # Step 4: Merge equivalent facts
        for group in equivalence_groups:
            if len(group) > 1:
                kept_fact = self._merge_fact_group(ir, group)
                result.facts_merged += len(group) - 1
                logger.debug(f"Merged {len(group)} facts into: {kept_fact.natural_language}")

        # Step 5: Handle entailment (remove entailed facts if they add no info)
        for entailing_id, entailed_id in entailment_pairs:
            if entailed_id in ir.facts and entailing_id in ir.facts:
                entailed_fact = ir.facts[entailed_id]
                entailing_fact = ir.facts[entailing_id]

                # Keep the more specific (entailing) fact
                if self._should_remove_entailed(entailed_fact, entailing_fact):
                    ir.remove_fact(entailed_id)
                    result.facts_removed += 1
                    logger.debug(
                        f"Removed entailed fact: {entailed_fact.natural_language} "
                        f"(entailed by {entailing_fact.natural_language})"
                    )

        result.modified = result.facts_merged > 0 or result.facts_removed > 0

        logger.info(
            f"FactFolding: merged {result.facts_merged} facts, "
            f"removed {result.facts_removed} entailed facts"
        )

        return result

    def _compute_embeddings(self, ir: MemoryIR) -> None:
        """Compute embeddings for all facts."""
        facts = list(ir.iter_facts())
        if not facts:
            return

        texts = [f.natural_language for f in facts]
        embeddings = self._embedder.encode(texts, convert_to_numpy=True)

        for fact, embedding in zip(facts, embeddings):
            fact.embedding = embedding.tolist()

    def _find_equivalent_facts(self, ir: MemoryIR) -> list[list[str]]:
        """Find groups of semantically equivalent facts."""
        facts = list(ir.iter_facts())
        n = len(facts)

        if n == 0:
            return []

        # Build similarity matrix
        if self.use_embeddings and facts[0].embedding is not None:
            similarity_matrix = self._compute_embedding_similarity(facts)
        else:
            similarity_matrix = self._compute_string_similarity(facts)

        # Find connected components where similarity > threshold
        # Using Union-Find for efficiency
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    union(i, j)

        # Group facts by their root
        groups: dict[int, list[str]] = defaultdict(list)
        for i, fact in enumerate(facts):
            root = find(i)
            groups[root].append(fact.id)

        # Return only groups with multiple facts
        return [group for group in groups.values() if len(group) > 1]

    def _compute_embedding_similarity(self, facts: list[Fact]) -> np.ndarray:
        """Compute pairwise cosine similarity using embeddings."""
        embeddings = np.array([f.embedding for f in facts])

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Compute cosine similarity matrix
        similarity = np.dot(normalized, normalized.T)
        return similarity

    def _compute_string_similarity(self, facts: list[Fact]) -> np.ndarray:
        """Compute pairwise similarity using string matching."""
        n = len(facts)
        similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    sim = self._string_similarity(facts[i], facts[j])
                    similarity[i, j] = sim
                    similarity[j, i] = sim

        return similarity

    def _string_similarity(self, fact1: Fact, fact2: Fact) -> float:
        """Compute string-based similarity between two facts."""
        # Check for exact triple match
        if fact1.triple == fact2.triple:
            return 1.0

        # Compute Jaccard similarity on words
        words1 = set(fact1.natural_language.lower().split())
        words2 = set(fact2.natural_language.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Boost similarity if subject and predicate match
        if fact1.subject.lower() == fact2.subject.lower():
            jaccard += 0.2
        if fact1.predicate.lower() == fact2.predicate.lower():
            jaccard += 0.1

        return min(1.0, jaccard)

    def _find_entailment_pairs(self, ir: MemoryIR) -> list[tuple[str, str]]:
        """Find pairs of facts where one entails the other."""
        pairs = []

        # Check existing entailment relations
        for relation in ir.relations.all_relations():
            if relation.relation_type == RelationType.ENTAILS:
                pairs.append((relation.source_fact_id, relation.target_fact_id))

        # Check for structural entailment
        facts = list(ir.iter_facts())
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i + 1 :]:
                if fact1.entails(fact2):
                    pairs.append((fact1.id, fact2.id))
                elif fact2.entails(fact1):
                    pairs.append((fact2.id, fact1.id))

        return pairs

    def _merge_fact_group(self, ir: MemoryIR, fact_ids: list[str]) -> Fact:
        """Merge a group of equivalent facts into one."""
        facts = [ir.facts[fid] for fid in fact_ids if fid in ir.facts]

        if not facts:
            raise ValueError("No facts to merge")

        if self.merge_strategy == "keep_first":
            kept = facts[0]
        elif self.merge_strategy == "keep_highest_confidence":
            kept = max(facts, key=lambda f: f.confidence)
        else:  # combine
            kept = facts[0]
            for other in facts[1:]:
                kept = kept.merge_with(other)

        # Remove all other facts
        for fact in facts:
            if fact.id != kept.id:
                # Update any relations pointing to removed facts
                for relation in ir.relations.all_relations():
                    if relation.source_fact_id == fact.id:
                        relation.source_fact_id = kept.id
                    if relation.target_fact_id == fact.id:
                        relation.target_fact_id = kept.id

                ir.remove_fact(fact.id)

        return kept

    def _should_remove_entailed(
        self, entailed: Fact, entailing: Fact
    ) -> bool:
        """Determine if an entailed fact should be removed."""
        # Keep if entailed fact has much higher confidence
        if entailed.confidence > entailing.confidence + 0.3:
            return False

        # Keep if entailed fact provides unique source information
        entailed_sources = set(entailed.source_turns)
        entailing_sources = set(entailing.source_turns)
        if entailed_sources - entailing_sources:
            return False

        return True


class SemanticDeduplicationPass(OptimizationPass):
    """Deduplicate facts using semantic clustering.

    This pass uses clustering to group similar facts and keeps
    only representative facts from each cluster.
    """

    def __init__(
        self,
        num_clusters: int | None = None,
        min_cluster_distance: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.min_cluster_distance = min_cluster_distance

    def run(self, ir: MemoryIR) -> PassResult:
        """Run semantic deduplication."""
        result = PassResult()

        facts = list(ir.iter_facts())
        if len(facts) < 2:
            return result

        # Get embeddings
        embeddings = []
        fact_ids = []
        for fact in facts:
            if fact.embedding is not None:
                embeddings.append(fact.embedding)
                fact_ids.append(fact.id)

        if len(embeddings) < 2:
            return result

        try:
            from sklearn.cluster import AgglomerativeClustering

            X = np.array(embeddings)

            # Determine number of clusters
            if self.num_clusters is None:
                n_clusters = max(1, len(X) // 3)
            else:
                n_clusters = min(self.num_clusters, len(X))

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(X)

            # For each cluster, keep the fact with highest importance
            clusters: dict[int, list[str]] = defaultdict(list)
            for fact_id, label in zip(fact_ids, labels):
                clusters[label].append(fact_id)

            for cluster_facts in clusters.values():
                if len(cluster_facts) > 1:
                    # Keep fact with highest importance
                    sorted_facts = sorted(
                        cluster_facts,
                        key=lambda fid: ir.facts[fid].importance_score,
                        reverse=True,
                    )

                    # Remove all but the most important
                    for fact_id in sorted_facts[1:]:
                        ir.remove_fact(fact_id)
                        result.facts_removed += 1

            result.modified = result.facts_removed > 0
            result.stats["num_clusters"] = len(clusters)

        except ImportError:
            logger.warning("sklearn not installed, skipping semantic deduplication")

        return result
