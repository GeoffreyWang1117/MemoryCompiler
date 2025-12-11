"""Tests for the main compression pipeline."""

import pytest

from memory_compiler.pipeline import (
    MemoryCompiler,
    CompressionResult,
    MemoryCompilerConfig,
    create_compiler,
)
from memory_compiler.extraction.extractor import MemoryExtractor
from memory_compiler.generation.generator import ContextGenerator


class TestMemoryExtractor:
    """Tests for MemoryExtractor."""

    def test_mock_extraction(self):
        """Test mock extraction mode."""
        extractor = MemoryExtractor(use_mock=True)

        dialogue = [
            {"role": "user", "content": "My name is Alice and I work at TechCorp."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]

        ir = extractor.extract(dialogue)

        # Should have extracted some facts
        assert len(ir.facts) > 0
        assert len(ir.dialogue_turns) == 2

    def test_extract_entities(self):
        """Test entity extraction."""
        extractor = MemoryExtractor(use_mock=True)

        dialogue = [
            {"role": "user", "content": "I'm Alice from TechCorp in New York."},
        ]

        ir = extractor.extract(dialogue)

        # Should have some entities (mock extraction uses capitalized words)
        entity_names = [e.name for e in ir.iter_entities()]
        # Check that at least Alice was found
        assert any("Alice" in name for name in entity_names)

    def test_incremental_extraction(self):
        """Test incremental extraction."""
        extractor = MemoryExtractor(use_mock=True)

        dialogue1 = [
            {"role": "user", "content": "My name is Alice."},
        ]

        ir = extractor.extract(dialogue1)
        initial_facts = len(ir.facts)

        dialogue2 = [
            {"role": "user", "content": "I work at TechCorp."},
        ]

        ir = extractor.extract_incremental(ir, dialogue2)

        # Should have more content now
        assert len(ir.dialogue_turns) == 2


class TestContextGenerator:
    """Tests for ContextGenerator."""

    def test_generate_structured(self):
        """Test structured output generation."""
        from memory_compiler.ir.memory_ir import MemoryIR
        from memory_compiler.ir.entities import EntityType

        ir = MemoryIR()
        ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.create_fact("Alice", "works at", "TechCorp")

        generator = ContextGenerator(output_format="structured")
        text = generator.generate(ir)

        assert "Alice" in text
        assert "TechCorp" in text

    def test_generate_bullets(self):
        """Test bullet point output."""
        from memory_compiler.ir.memory_ir import MemoryIR
        from memory_compiler.ir.entities import EntityType

        ir = MemoryIR()
        ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.create_fact("Alice", "is", "engineer")

        generator = ContextGenerator(output_format="bullets")
        text = generator.generate(ir)

        assert "•" in text or "-" in text

    def test_generate_json(self):
        """Test JSON output."""
        import json
        from memory_compiler.ir.memory_ir import MemoryIR
        from memory_compiler.ir.entities import EntityType

        ir = MemoryIR()
        ir.find_or_create_entity("Alice", EntityType.PERSON)
        ir.create_fact("Alice", "is", "engineer")

        generator = ContextGenerator(output_format="json")
        text = generator.generate(ir)

        # Should be valid JSON
        data = json.loads(text)
        assert "entities" in data
        assert "facts" in data


class TestMemoryCompiler:
    """Tests for MemoryCompiler."""

    def test_compress_dialogue(self):
        """Test basic dialogue compression."""
        compiler = MemoryCompiler(
            use_mock_extraction=True,
            token_budget=1000,
        )

        dialogue = [
            {"role": "user", "content": "Hi, my name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help?"},
            {"role": "user", "content": "I work at TechCorp as an engineer."},
            {"role": "assistant", "content": "That's great! What do you need help with?"},
        ]

        result = compiler.compress(dialogue)

        assert isinstance(result, CompressionResult)
        assert result.text != ""
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0
        assert 0 < result.compression_ratio <= 1.0

    def test_compression_result_metadata(self):
        """Test compression result contains metadata."""
        compiler = MemoryCompiler(use_mock_extraction=True)

        dialogue = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = compiler.compress(dialogue)

        assert "num_turns" in result.metadata
        assert result.metadata["num_turns"] == 2

    def test_pass_results(self):
        """Test that pass results are recorded."""
        compiler = MemoryCompiler(
            use_mock_extraction=True,
            enable_dead_memory=True,
            enable_fact_folding=True,
        )

        dialogue = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
        ]

        result = compiler.compress(dialogue)

        # Should have pass results
        assert len(result.pass_results) > 0


class TestMemoryCompilerConfig:
    """Tests for configuration presets."""

    def test_default_config(self):
        """Test default configuration."""
        config = MemoryCompilerConfig.default()

        assert "token_budget" in config
        assert config["token_budget"] == 2048

    def test_aggressive_config(self):
        """Test aggressive configuration."""
        config = MemoryCompilerConfig.aggressive()

        assert config["token_budget"] == 1024
        assert config["output_format"] == "bullets"

    def test_conservative_config(self):
        """Test conservative configuration."""
        config = MemoryCompilerConfig.conservative()

        assert config["token_budget"] == 4096
        assert config["enable_temporal"] is False

    def test_create_compiler_factory(self):
        """Test create_compiler factory function."""
        compiler = create_compiler("default")
        assert compiler is not None

        compiler = create_compiler("aggressive")
        assert compiler.token_budget == 1024

        with pytest.raises(ValueError):
            create_compiler("invalid_config")


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_long_dialogue_compression(self):
        """Test compression of a longer dialogue."""
        compiler = MemoryCompiler(
            use_mock_extraction=True,
            token_budget=500,
        )

        # Create a longer dialogue
        dialogue = []
        for i in range(20):
            if i % 2 == 0:
                dialogue.append({
                    "role": "user",
                    "content": f"This is message {i} from the user about topic {i//4}."
                })
            else:
                dialogue.append({
                    "role": "assistant",
                    "content": f"This is response {i} from the assistant."
                })

        result = compiler.compress(dialogue)

        # Should achieve some compression
        assert result.compression_ratio < 1.0
        assert result.compressed_tokens < result.original_tokens

    def test_ir_persistence(self):
        """Test saving and loading Memory IR."""
        import tempfile
        import os

        compiler = MemoryCompiler(use_mock_extraction=True)

        dialogue = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
        ]

        result = compiler.compress(dialogue)

        # Save IR
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            compiler.save_ir(result.ir, path)

            # Load IR
            loaded_ir = compiler.load_ir(path)

            assert len(loaded_ir.facts) == len(result.ir.facts)
            assert len(loaded_ir.entities) == len(result.ir.entities)
        finally:
            os.unlink(path)
