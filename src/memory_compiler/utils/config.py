"""Configuration management for MemoryCompiler.

This module provides YAML-based configuration loading and validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "auto"
    dtype: str = "float16"
    max_length: int = 4096


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_enabled: bool = True
    cache_dir: str = ".cache/embeddings"
    batch_size: int = 32


@dataclass
class CompressionConfig:
    """Configuration for compression settings."""
    token_budget: int = 2048
    output_format: str = "structured"


@dataclass
class ExtractionConfig:
    """Configuration for extraction settings."""
    batch_size: int = 10
    use_mock: bool = False
    confidence_threshold: float = 0.5


@dataclass
class DeadMemoryPassConfig:
    """Configuration for dead memory elimination pass."""
    enabled: bool = True
    recency_threshold: int = 10
    min_entity_mentions: int = 1
    min_fact_confidence: float = 0.5
    remove_superseded: bool = True


@dataclass
class FactFoldingPassConfig:
    """Configuration for fact folding pass."""
    enabled: bool = True
    similarity_threshold: float = 0.85
    use_embeddings: bool = True
    fold_entailed: bool = True
    merge_strategy: str = "keep_highest_confidence"


@dataclass
class TemporalCompressionPassConfig:
    """Configuration for temporal compression pass."""
    enabled: bool = True
    min_sequence_length: int = 3
    max_turn_gap: int = 5
    similarity_threshold: float = 0.5
    use_llm_summarization: bool = False


@dataclass
class ImportancePruningPassConfig:
    """Configuration for importance pruning pass."""
    enabled: bool = True
    min_importance: float = 0.1
    recency_weight: float = 0.3
    frequency_weight: float = 0.3
    centrality_weight: float = 0.2
    type_weight: float = 0.2
    use_knapsack: bool = True


@dataclass
class PassesConfig:
    """Configuration for all optimization passes."""
    dead_memory: DeadMemoryPassConfig = field(default_factory=DeadMemoryPassConfig)
    fact_folding: FactFoldingPassConfig = field(default_factory=FactFoldingPassConfig)
    temporal_compression: TemporalCompressionPassConfig = field(
        default_factory=TemporalCompressionPassConfig
    )
    importance_pruning: ImportancePruningPassConfig = field(
        default_factory=ImportancePruningPassConfig
    )


@dataclass
class GenerationConfig:
    """Configuration for context generation."""
    include_entities: bool = True
    include_facts: bool = True
    include_events: bool = True
    group_by_entity: bool = True
    temporal_ordering: bool = True
    max_entities_per_type: int = 10
    max_facts_per_subject: int = 5
    max_events: int = 15


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    compute_fact_recall: bool = True
    compute_entity_recall: bool = True
    compute_qa_accuracy: bool = True
    compute_semantic_similarity: bool = True
    similarity_threshold: float = 0.8


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: str | None = None
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    directory: str = ".cache/memory_compiler"
    ttl: int = 3600
    max_size: int = 1000


@dataclass
class Config:
    """Main configuration class for MemoryCompiler."""

    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    passes: PassesConfig = field(default_factory=PassesConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create configuration from dictionary."""
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "compression" in data:
            config.compression = CompressionConfig(**data["compression"])
        if "extraction" in data:
            config.extraction = ExtractionConfig(**data["extraction"])
        if "generation" in data:
            config.generation = GenerationConfig(**data["generation"])
        if "evaluation" in data:
            config.evaluation = EvaluationConfig(**data["evaluation"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "cache" in data:
            config.cache = CacheConfig(**data["cache"])

        if "passes" in data:
            passes_data = data["passes"]
            config.passes = PassesConfig(
                dead_memory=DeadMemoryPassConfig(**passes_data.get("dead_memory", {})),
                fact_folding=FactFoldingPassConfig(**passes_data.get("fact_folding", {})),
                temporal_compression=TemporalCompressionPassConfig(
                    **passes_data.get("temporal_compression", {})
                ),
                importance_pruning=ImportancePruningPassConfig(
                    **passes_data.get("importance_pruning", {})
                ),
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses

        def to_dict_recursive(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: to_dict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return to_dict_recursive(self)

    def merge_with(self, overrides: dict[str, Any]) -> Config:
        """Create new config with overrides applied."""
        base = self.to_dict()
        _deep_merge(base, overrides)
        return Config.from_dict(base)


def _deep_merge(base: dict, updates: dict) -> None:
    """Deep merge updates into base dictionary."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_config(
    config_path: str | Path | None = None,
    config_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to a YAML config file.
        config_name: Name of a preset config (default, aggressive, conservative).
        overrides: Dictionary of values to override.

    Returns:
        Loaded and validated configuration.
    """
    # Start with default config
    config = Config()

    # Determine config file to load
    if config_path:
        path = Path(config_path)
    elif config_name:
        # Look for preset configs
        package_dir = Path(__file__).parent.parent.parent.parent
        path = package_dir / "configs" / f"{config_name}.yaml"
        if not path.exists():
            # Try current directory
            path = Path("configs") / f"{config_name}.yaml"
    else:
        path = None

    # Load from file if available
    if path and path.exists():
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data:
                config = Config.from_dict(data)
                logger.info(f"Loaded configuration from: {path}")
        except ImportError:
            logger.warning("PyYAML not installed, using default configuration")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    # Apply overrides
    if overrides:
        config = config.merge_with(overrides)

    # Setup logging based on config
    _setup_logging(config.logging)

    return config


def _setup_logging(log_config: LoggingConfig) -> None:
    """Setup logging based on configuration."""
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_config.level,
        format=log_config.format,
    )

    if log_config.file:
        logger.add(
            log_config.file,
            level=log_config.level,
            format=log_config.format,
            rotation="10 MB",
        )


def get_config_path(name: str) -> Path | None:
    """Get path to a named configuration file."""
    # Check package configs
    package_dir = Path(__file__).parent.parent.parent.parent
    path = package_dir / "configs" / f"{name}.yaml"
    if path.exists():
        return path

    # Check current directory
    path = Path("configs") / f"{name}.yaml"
    if path.exists():
        return path

    # Check environment variable
    env_path = os.environ.get("MEMORY_COMPILER_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    return None
