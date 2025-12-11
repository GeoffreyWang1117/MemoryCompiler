#!/usr/bin/env python3
"""Basic usage example for MemoryCompiler.

This script demonstrates how to use MemoryCompiler to compress
dialogue history for use with language models.
"""

from memory_compiler import MemoryCompiler, MemoryIR
from memory_compiler.pipeline import create_compiler


def basic_compression():
    """Demonstrate basic dialogue compression."""
    print("=" * 60)
    print("Basic Dialogue Compression")
    print("=" * 60)

    # Create a sample dialogue
    dialogue = [
        {"role": "user", "content": "Hi, my name is Alice and I work at TechCorp."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you. What do you do at TechCorp?"},
        {"role": "user", "content": "I'm a machine learning engineer working on NLP systems."},
        {"role": "assistant", "content": "That's exciting! NLP is a fascinating field. What projects are you working on?"},
        {"role": "user", "content": "Currently I'm building a dialogue system for customer support."},
        {"role": "assistant", "content": "Customer support is a great application. Are you using any specific frameworks?"},
        {"role": "user", "content": "Yes, we're using PyTorch and Hugging Face Transformers."},
        {"role": "assistant", "content": "Great choices! Those are widely used in the industry."},
    ]

    # Create compiler with default settings
    compiler = MemoryCompiler(
        use_mock_extraction=True,  # Use mock for demo (no GPU needed)
        token_budget=500,
        output_format="structured",
    )

    # Compress the dialogue
    result = compiler.compress(dialogue)

    # Display results
    print("\nOriginal dialogue tokens:", result.original_tokens)
    print("Compressed tokens:", result.compressed_tokens)
    print(f"Compression ratio: {result.compression_ratio:.1%}")

    print("\n--- Compressed Context ---")
    print(result.text)
    print("-" * 40)

    return result


def different_formats():
    """Demonstrate different output formats."""
    print("\n" + "=" * 60)
    print("Different Output Formats")
    print("=" * 60)

    dialogue = [
        {"role": "user", "content": "I'm Bob, a data scientist at DataCorp."},
        {"role": "assistant", "content": "Hello Bob! How can I assist you?"},
        {"role": "user", "content": "I need help with feature engineering for a classification task."},
    ]

    formats = ["structured", "bullets", "narrative", "json"]

    for fmt in formats:
        compiler = MemoryCompiler(
            use_mock_extraction=True,
            token_budget=300,
            output_format=fmt,
        )

        result = compiler.compress(dialogue)

        print(f"\n--- Format: {fmt} ---")
        print(result.text[:500] + "..." if len(result.text) > 500 else result.text)


def configuration_presets():
    """Demonstrate different configuration presets."""
    print("\n" + "=" * 60)
    print("Configuration Presets")
    print("=" * 60)

    dialogue = [
        {"role": "user", "content": "Hi, I'm Carol from FinTech Inc."},
        {"role": "assistant", "content": "Hello Carol!"},
        {"role": "user", "content": "I work on fraud detection systems."},
        {"role": "assistant", "content": "That's important work!"},
    ]

    presets = ["default", "aggressive", "conservative"]

    for preset in presets:
        compiler = create_compiler(preset)
        result = compiler.compress(dialogue)

        print(f"\n{preset.upper()} preset:")
        print(f"  Token budget: {compiler.token_budget}")
        print(f"  Compression ratio: {result.compression_ratio:.1%}")
        print(f"  Compressed tokens: {result.compressed_tokens}")


def inspect_memory_ir():
    """Demonstrate inspecting the Memory IR."""
    print("\n" + "=" * 60)
    print("Inspecting Memory IR")
    print("=" * 60)

    dialogue = [
        {"role": "user", "content": "My name is Diana and I'm from Seattle."},
        {"role": "assistant", "content": "Hi Diana! Seattle is a great city."},
        {"role": "user", "content": "Yes, I love the tech scene here. I work at CloudTech."},
    ]

    compiler = MemoryCompiler(use_mock_extraction=True)
    result = compiler.compress(dialogue)

    ir = result.ir

    print("\n--- Memory IR Statistics ---")
    stats = ir.get_stats()
    print(f"Entities: {stats.num_entities}")
    print(f"Facts: {stats.num_facts}")
    print(f"Events: {stats.num_events}")
    print(f"Relations: {stats.num_relations}")

    print("\n--- Entities ---")
    for entity in ir.iter_entities():
        print(f"  {entity.name} ({entity.type.value})")

    print("\n--- Facts ---")
    for fact in ir.iter_facts():
        print(f"  {fact.natural_language}")

    print("\n--- Events ---")
    for event in ir.iter_events():
        print(f"  Turn {event.turn_index}: {event.description[:50]}...")


def incremental_compression():
    """Demonstrate incremental compression for streaming scenarios."""
    print("\n" + "=" * 60)
    print("Incremental Compression")
    print("=" * 60)

    compiler = MemoryCompiler(use_mock_extraction=True, token_budget=400)

    # Initial dialogue
    initial_dialogue = [
        {"role": "user", "content": "Hi, I'm working on a Python project."},
        {"role": "assistant", "content": "I'd be happy to help! What kind of project?"},
    ]

    result = compiler.compress(initial_dialogue)
    print(f"\nAfter initial turns: {result.compressed_tokens} tokens")

    # Add more turns incrementally
    new_turns_1 = [
        {"role": "user", "content": "It's a web scraping tool using BeautifulSoup."},
        {"role": "assistant", "content": "BeautifulSoup is great for HTML parsing!"},
    ]

    result = compiler.compress_incremental(result.ir, new_turns_1)
    print(f"After adding 2 turns: {result.compressed_tokens} tokens")

    new_turns_2 = [
        {"role": "user", "content": "I'm having trouble with JavaScript-rendered content."},
        {"role": "assistant", "content": "You might need Selenium or Playwright for that."},
    ]

    result = compiler.compress_incremental(result.ir, new_turns_2)
    print(f"After adding 2 more turns: {result.compressed_tokens} tokens")


if __name__ == "__main__":
    basic_compression()
    different_formats()
    configuration_presets()
    inspect_memory_ir()
    incremental_compression()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
