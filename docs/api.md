# MemoryCompiler API Reference

## Main Classes

### MemoryCompiler

The main interface for dialogue compression.

```python
from memory_compiler import MemoryCompiler

compiler = MemoryCompiler(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # LLM for extraction
    token_budget=2048,                       # Target token budget
    output_format="structured",               # Output format
    use_mock_extraction=True,                # Use mock for testing
)
```

#### Methods

**compress(dialogue) → CompressionResult**
Compress a dialogue into condensed context.

```python
dialogue = [
    {"role": "user", "content": "Hello, I'm Alice."},
    {"role": "assistant", "content": "Hi Alice!"},
]
result = compiler.compress(dialogue)
print(result.text)
print(f"Compression: {result.compression_ratio:.1%}")
```

**compress_incremental(ir, new_turns) → CompressionResult**
Add new turns to existing Memory IR.

```python
result = compiler.compress_incremental(result.ir, new_turns)
```

**save_ir(ir, path) / load_ir(path)**
Persist Memory IR to disk.

```python
compiler.save_ir(result.ir, "memory.json")
ir = compiler.load_ir("memory.json")
```

### CompressionResult

Result of dialogue compression.

```python
@dataclass
class CompressionResult:
    text: str                    # Compressed context
    ir: MemoryIR                 # Optimized Memory IR
    original_tokens: int         # Original dialogue tokens
    compressed_tokens: int       # Compressed output tokens
    compression_ratio: float     # Ratio (compressed/original)
    pass_results: dict          # Results from each pass
    metadata: dict              # Additional metadata
```

### MemoryIR

Intermediate representation of dialogue memory.

```python
from memory_compiler.ir import MemoryIR

ir = MemoryIR()

# Add entities
entity = ir.find_or_create_entity("Alice", EntityType.PERSON)

# Add facts
fact = ir.create_fact("Alice", "works at", "TechCorp")

# Add events
event = ir.create_event("User introduced themselves", EventType.ACTION, 0)

# Query
facts = ir.get_facts_about("Alice")
stats = ir.get_stats()
```

### Optimization Passes

All passes inherit from `OptimizationPass`:

```python
from memory_compiler.passes import (
    DeadMemoryEliminationPass,
    FactFoldingPass,
    TemporalCompressionPass,
    ImportancePruningPass,
    QueryPredictionPruningPass,
)

# Create custom pass pipeline
from memory_compiler.passes import PassManager

manager = PassManager([
    DeadMemoryEliminationPass(recency_threshold=10),
    FactFoldingPass(similarity_threshold=0.85),
    ImportancePruningPass(token_budget=2048),
])

result = manager.run(ir)
```

### ContextGenerator

Generate text from Memory IR.

```python
from memory_compiler.generation import ContextGenerator

generator = ContextGenerator(
    output_format="structured",  # structured, narrative, bullets, json
    include_entities=True,
    include_facts=True,
    include_events=True,
)

text = generator.generate(ir)
```

### EmbeddingManager

Manage embeddings with caching.

```python
from memory_compiler.utils import EmbeddingManager

embedder = EmbeddingManager(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_enabled=True,
)

# Single embedding
emb = embedder.embed("Hello world")

# Batch embedding
embs = embedder.embed_batch(["Hello", "World"])

# Similarity
sim = embedder.similarity("Hello", "Hi")

# Clustering
clusters = embedder.cluster_texts(texts, n_clusters=5)
```

### ConversationSummarizer

LLM-based summarization.

```python
from memory_compiler.summarization import ConversationSummarizer

summarizer = ConversationSummarizer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    style="detailed",  # executive, detailed, bullet, narrative, qa
)

result = summarizer.summarize(dialogue)
print(result.summary)
print(result.key_points)
```

### AsyncMemoryCompiler

Async and streaming support.

```python
from memory_compiler.async_pipeline import AsyncMemoryCompiler

compiler = AsyncMemoryCompiler(token_budget=2048)

# Async compression
result = await compiler.compress_async(dialogue)

# Batch processing
results = await compiler.compress_batch(dialogues, max_concurrent=5)

# Streaming
async for result in compiler.stream_compress(turn_stream):
    print(result.text)
```

## Configuration

### Load from YAML

```python
from memory_compiler.utils import load_config

config = load_config("configs/default.yaml")
# or
config = load_config(config_name="aggressive")
```

### Factory Function

```python
from memory_compiler import create_compiler

compiler = create_compiler("default")      # Balanced
compiler = create_compiler("aggressive")   # Maximum compression
compiler = create_compiler("conservative") # Preserve more info
```

## Evaluation

### Metrics

```python
from memory_compiler.evaluation import evaluate_compression

metrics = evaluate_compression(
    result,
    original_ir=original_ir,
    qa_pairs=[("What is Alice's job?", "engineer")],
    original_text=original_dialogue_text,
)

print(metrics["compression"]["compression_ratio"])
print(metrics["retention"]["fact_recall"])
```

### Benchmarking

```python
from memory_compiler.evaluation import Benchmark

benchmark = Benchmark()
results = benchmark.run(
    datasets=["synthetic"],
    configs=["default", "aggressive"],
    num_samples=100,
)
benchmark.save_results("results.json")
print(benchmark.generate_report())
```

## CLI Usage

```bash
# Compress a dialogue
memory-compiler compress input.json -o output.txt --budget 2048

# Run demo
memory-compiler demo

# Evaluate
memory-compiler evaluate input.json -o metrics.json
```

## Error Handling

```python
try:
    result = compiler.compress(dialogue)
except ExtractionError as e:
    print(f"Extraction failed: {e}")
except CompressionError as e:
    print(f"Compression failed: {e}")
```

## Logging

```python
from loguru import logger

# Configure logging level
import sys
logger.remove()
logger.add(sys.stderr, level="DEBUG")
```
