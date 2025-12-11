"""Main compression pipeline for MemoryCompiler.

This module provides the high-level interface for dialogue memory compression,
orchestrating extraction, optimization, and generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from memory_compiler.extraction.extractor import MemoryExtractor
from memory_compiler.generation.generator import AdaptiveContextGenerator, ContextGenerator
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.passes.base import PassManager, PassResult
from memory_compiler.passes.dead_memory import DeadMemoryEliminationPass
from memory_compiler.passes.fact_folding import FactFoldingPass
from memory_compiler.passes.importance import ImportancePruningPass
from memory_compiler.passes.temporal import TemporalCompressionPass


@dataclass
class CompressionResult:
    """Result of dialogue compression.

    Attributes:
        text: The compressed context text.
        ir: The optimized Memory IR.
        original_tokens: Estimated tokens in original dialogue.
        compressed_tokens: Tokens in compressed output.
        compression_ratio: Ratio of compressed to original size.
        pass_results: Results from each optimization pass.
        metadata: Additional metadata.
    """

    text: str
    ir: MemoryIR
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    pass_results: dict[str, PassResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryCompiler:
    """Main class for dialogue memory compression.

    The MemoryCompiler orchestrates the full compression pipeline:
    1. Extract structured information from dialogue into Memory IR
    2. Apply optimization passes to reduce memory size
    3. Generate compressed natural language context

    Example:
        >>> compiler = MemoryCompiler(token_budget=2048)
        >>> dialogue = [
        ...     {"role": "user", "content": "Hi, I'm Alice."},
        ...     {"role": "assistant", "content": "Hello Alice!"},
        ... ]
        >>> result = compiler.compress(dialogue)
        >>> print(result.text)
        >>> print(f"Compression ratio: {result.compression_ratio:.2%}")

    Configuration options:
        model_name: LLM for extraction (default: mock extraction).
        token_budget: Target token budget for output (default: 2048).
        output_format: Format of generated context (default: "structured").
        enable_dead_memory: Run dead memory elimination (default: True).
        enable_fact_folding: Run fact folding (default: True).
        enable_temporal: Run temporal compression (default: True).
        enable_importance: Run importance pruning (default: True).
        use_mock_extraction: Use mock extraction for testing (default: True).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        token_budget: int = 2048,
        output_format: str = "structured",
        enable_dead_memory: bool = True,
        enable_fact_folding: bool = True,
        enable_temporal: bool = True,
        enable_importance: bool = True,
        use_mock_extraction: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.token_budget = token_budget
        self.output_format = output_format
        self.config = kwargs

        # Initialize extractor
        self.extractor = MemoryExtractor(
            model_name=model_name,
            use_mock=use_mock_extraction,
        )

        # Initialize pass manager with optimization passes
        self.pass_manager = PassManager()

        if enable_dead_memory:
            self.pass_manager.add_pass(
                DeadMemoryEliminationPass(
                    recency_threshold=kwargs.get("recency_threshold", 10),
                    min_fact_confidence=kwargs.get("min_fact_confidence", 0.5),
                )
            )

        if enable_fact_folding:
            self.pass_manager.add_pass(
                FactFoldingPass(
                    similarity_threshold=kwargs.get("similarity_threshold", 0.85),
                    use_embeddings=kwargs.get("use_embeddings", False),
                )
            )

        if enable_temporal:
            self.pass_manager.add_pass(
                TemporalCompressionPass(
                    min_sequence_length=kwargs.get("min_sequence_length", 3),
                    max_turn_gap=kwargs.get("max_turn_gap", 5),
                )
            )

        if enable_importance:
            self.pass_manager.add_pass(
                ImportancePruningPass(
                    token_budget=token_budget,
                    min_importance=kwargs.get("min_importance", 0.1),
                )
            )

        # Initialize generator
        self.generator = AdaptiveContextGenerator(
            token_budget=token_budget,
            output_format=output_format,
        )

    def compress(self, dialogue: list[dict[str, str]]) -> CompressionResult:
        """Compress a dialogue into condensed context.

        Args:
            dialogue: List of dialogue turns with 'role' and 'content'.

        Returns:
            CompressionResult with compressed text and statistics.
        """
        logger.info(f"Compressing dialogue with {len(dialogue)} turns")

        # Step 1: Estimate original token count
        original_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in dialogue
        )
        original_tokens = self.generator.count_tokens(original_text)
        logger.info(f"Original dialogue: ~{original_tokens} tokens")

        # Step 2: Extract Memory IR from dialogue
        logger.info("Extracting Memory IR from dialogue...")
        ir = self.extractor.extract(dialogue)
        logger.info(f"Extracted IR: {ir}")

        # Step 3: Run optimization passes
        logger.info("Running optimization passes...")
        pass_results: dict[str, PassResult] = {}

        for pass_ in self.pass_manager.passes:
            result = pass_.run(ir)
            pass_results[pass_.name] = result
            logger.info(f"  {pass_.name}: {result}")

        # Step 4: Generate compressed context
        logger.info("Generating compressed context...")
        compressed_text = self.generator.generate(ir)
        compressed_tokens = self.generator.count_tokens(compressed_text)

        # Compute compression ratio
        if original_tokens > 0:
            compression_ratio = compressed_tokens / original_tokens
        else:
            compression_ratio = 1.0

        logger.info(
            f"Compression complete: {original_tokens} -> {compressed_tokens} tokens "
            f"({compression_ratio:.1%})"
        )

        return CompressionResult(
            text=compressed_text,
            ir=ir,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            pass_results=pass_results,
            metadata={
                "model_name": self.model_name,
                "token_budget": self.token_budget,
                "output_format": self.output_format,
                "num_turns": len(dialogue),
            },
        )

    def compress_incremental(
        self,
        ir: MemoryIR,
        new_turns: list[dict[str, str]],
    ) -> CompressionResult:
        """Incrementally compress new dialogue turns.

        This is more efficient for streaming scenarios where new turns
        are added to an existing conversation.

        Args:
            ir: Existing Memory IR from previous compression.
            new_turns: New dialogue turns to add.

        Returns:
            CompressionResult with updated compressed text.
        """
        logger.info(f"Incrementally compressing {len(new_turns)} new turns")

        # Extract from new turns
        ir = self.extractor.extract_incremental(ir, new_turns)

        # Run optimization passes
        pass_results: dict[str, PassResult] = {}
        for pass_ in self.pass_manager.passes:
            result = pass_.run(ir)
            pass_results[pass_.name] = result

        # Generate compressed context
        compressed_text = self.generator.generate(ir)
        compressed_tokens = self.generator.count_tokens(compressed_text)

        # Estimate original tokens (all dialogue)
        original_text = "\n".join(
            f"{turn.role}: {turn.content}" for turn in ir.dialogue_turns
        )
        original_tokens = self.generator.count_tokens(original_text)

        compression_ratio = compressed_tokens / max(original_tokens, 1)

        return CompressionResult(
            text=compressed_text,
            ir=ir,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            pass_results=pass_results,
        )

    def save_ir(self, ir: MemoryIR, path: str | Path) -> None:
        """Save Memory IR to file for later use."""
        ir.save(path)

    def load_ir(self, path: str | Path) -> MemoryIR:
        """Load Memory IR from file."""
        return MemoryIR.load(path)


class MemoryCompilerConfig:
    """Configuration class for MemoryCompiler.

    Provides named configurations for common use cases.
    """

    @staticmethod
    def default() -> dict[str, Any]:
        """Default balanced configuration."""
        return {
            "token_budget": 2048,
            "output_format": "structured",
            "enable_dead_memory": True,
            "enable_fact_folding": True,
            "enable_temporal": True,
            "enable_importance": True,
            "use_mock_extraction": True,
        }

    @staticmethod
    def aggressive() -> dict[str, Any]:
        """Aggressive compression for tight token budgets."""
        return {
            "token_budget": 1024,
            "output_format": "bullets",
            "enable_dead_memory": True,
            "enable_fact_folding": True,
            "enable_temporal": True,
            "enable_importance": True,
            "recency_threshold": 5,
            "min_fact_confidence": 0.6,
            "min_importance": 0.2,
            "use_mock_extraction": True,
        }

    @staticmethod
    def conservative() -> dict[str, Any]:
        """Conservative compression preserving more information."""
        return {
            "token_budget": 4096,
            "output_format": "structured",
            "enable_dead_memory": True,
            "enable_fact_folding": True,
            "enable_temporal": False,
            "enable_importance": True,
            "recency_threshold": 20,
            "min_fact_confidence": 0.3,
            "min_importance": 0.05,
            "use_mock_extraction": True,
        }

    @staticmethod
    def with_llm(model_name: str) -> dict[str, Any]:
        """Configuration using actual LLM for extraction."""
        return {
            "model_name": model_name,
            "token_budget": 2048,
            "output_format": "structured",
            "enable_dead_memory": True,
            "enable_fact_folding": True,
            "enable_temporal": True,
            "enable_importance": True,
            "use_mock_extraction": False,
            "use_embeddings": True,
        }


def create_compiler(config: str | dict[str, Any] = "default") -> MemoryCompiler:
    """Factory function to create a MemoryCompiler with named configuration.

    Args:
        config: Configuration name ("default", "aggressive", "conservative")
               or custom configuration dict.

    Returns:
        Configured MemoryCompiler instance.
    """
    if isinstance(config, str):
        configs = {
            "default": MemoryCompilerConfig.default,
            "aggressive": MemoryCompilerConfig.aggressive,
            "conservative": MemoryCompilerConfig.conservative,
        }
        if config not in configs:
            raise ValueError(f"Unknown config: {config}. Use one of {list(configs.keys())}")
        config_dict = configs[config]()
    else:
        config_dict = config

    return MemoryCompiler(**config_dict)
