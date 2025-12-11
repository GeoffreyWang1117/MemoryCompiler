# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/MemoryCompiler/MemoryCompiler.git
cd MemoryCompiler

# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Basic Usage

### Simple Compression

```python
from memory_compiler import MemoryCompiler

# Create compiler with default settings
compiler = MemoryCompiler(
    token_budget=2048,
    use_mock_extraction=True,  # Set False to use real LLM
)

# Sample dialogue
dialogue = [
    {"role": "user", "content": "Hi, my name is Alice and I work at TechCorp."},
    {"role": "assistant", "content": "Hello Alice! Nice to meet you. What do you do at TechCorp?"},
    {"role": "user", "content": "I'm a machine learning engineer working on NLP systems."},
    {"role": "assistant", "content": "That's exciting! What projects are you working on?"},
    {"role": "user", "content": "Currently building a dialogue summarization system."},
]

# Compress
result = compiler.compress(dialogue)

# View results
print("Compressed Context:")
print(result.text)
print(f"\nOriginal tokens: {result.original_tokens}")
print(f"Compressed tokens: {result.compressed_tokens}")
print(f"Compression ratio: {result.compression_ratio:.1%}")
```

### Using Configuration Presets

```python
from memory_compiler.pipeline import create_compiler

# Aggressive compression (smaller output)
compiler = create_compiler("aggressive")

# Conservative compression (preserve more info)
compiler = create_compiler("conservative")

# Default balanced compression
compiler = create_compiler("default")
```

### Different Output Formats

```python
# Structured output (default)
compiler = MemoryCompiler(output_format="structured")

# Bullet points
compiler = MemoryCompiler(output_format="bullets")

# Flowing narrative
compiler = MemoryCompiler(output_format="narrative")

# JSON output
compiler = MemoryCompiler(output_format="json")
```

### Inspecting Memory IR

```python
result = compiler.compress(dialogue)
ir = result.ir

# View statistics
stats = ir.get_stats()
print(f"Entities: {stats.num_entities}")
print(f"Facts: {stats.num_facts}")
print(f"Events: {stats.num_events}")

# Iterate over extracted information
print("\nEntities:")
for entity in ir.iter_entities():
    print(f"  - {entity.name} ({entity.type.value})")

print("\nFacts:")
for fact in ir.iter_facts():
    print(f"  - {fact.natural_language}")
```

### Streaming/Incremental Compression

```python
from memory_compiler.async_pipeline import StreamingDialogueProcessor
import asyncio

async def process_stream():
    processor = StreamingDialogueProcessor(
        token_budget=500,
        auto_compress_threshold=5,
    )

    # Add turns one by one
    turns = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        # ... more turns
    ]

    for turn in turns:
        await processor.add_turn(turn)

    # Get compressed context
    context = await processor.get_compressed_context()
    print(context)

asyncio.run(process_stream())
```

### Using LLM Summarization

```python
from memory_compiler.summarization import ConversationSummarizer

summarizer = ConversationSummarizer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    style="detailed",  # or "executive", "bullet", "narrative"
    use_mock=True,     # Set False for real LLM
)

result = summarizer.summarize(dialogue)
print(result.summary)
print("Key points:", result.key_points)
```

### Evaluation

```python
from memory_compiler.evaluation import evaluate_compression

# Define QA pairs for evaluation
qa_pairs = [
    ("What is the user's name?", "Alice"),
    ("Where does Alice work?", "TechCorp"),
]

# Evaluate compression quality
metrics = evaluate_compression(
    result,
    qa_pairs=qa_pairs,
    original_text="\n".join(t['content'] for t in dialogue),
)

print(f"Fact recall: {metrics['retention']['fact_recall']:.2f}")
print(f"QA accuracy: {metrics['retention']['qa_accuracy']:.2f}")
```

### Running Benchmarks

```python
from memory_compiler.evaluation import Benchmark

benchmark = Benchmark()
results = benchmark.run(
    datasets=["synthetic"],
    configs=["default", "aggressive"],
    num_samples=50,
)

print(benchmark.generate_report())
benchmark.save_results("benchmark.json")
```

## Command Line Interface

```bash
# Run demo
memory-compiler demo

# Compress a dialogue file
memory-compiler compress dialogue.json -o compressed.txt --budget 1024

# Evaluate compression
memory-compiler evaluate dialogue.json -o metrics.json
```

## YAML Configuration

Create a custom configuration file:

```yaml
# my_config.yaml
compression:
  token_budget: 1500
  output_format: bullets

passes:
  dead_memory:
    enabled: true
    recency_threshold: 8
  fact_folding:
    enabled: true
    similarity_threshold: 0.8
  importance_pruning:
    enabled: true
    min_importance: 0.15
```

Use it:

```python
from memory_compiler.utils import load_config
from memory_compiler import MemoryCompiler

config = load_config("my_config.yaml")
compiler = MemoryCompiler(**config.compression.__dict__)
```

## Next Steps

- Read the [Architecture Guide](architecture.md) for deeper understanding
- Check the [API Reference](api.md) for detailed documentation
- Explore [examples/](../examples/) for more use cases
