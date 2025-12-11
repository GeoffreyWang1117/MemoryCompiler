# MemoryCompiler Architecture

## Overview

MemoryCompiler is a framework for compressing long dialogue history using compiler optimization principles. It treats dialogue compression as a compilation problem where:

- **Source Code** = Original dialogue history
- **Intermediate Representation (IR)** = Structured memory representation
- **Optimization Passes** = Information-preserving transformations
- **Target Code** = Compressed context

## Core Components

### 1. Memory IR (Intermediate Representation)

The Memory IR is a structured representation of dialogue information:

```
MemoryIR
├── Entities          # People, organizations, concepts
├── Facts             # Subject-predicate-object triples
├── Relations         # Semantic connections between facts
├── Events            # Temporal event sequences
└── DialogueTurns     # Original dialogue reference
```

#### Entities
- Represent key objects mentioned in dialogue
- Types: Person, Organization, Location, Project, Concept, etc.
- Track mentions, aliases, and attributes

#### Facts
- Store information as (subject, predicate, object) triples
- Include confidence scores and source turn references
- Support negation and temporal scoping

#### Relations
- Connect facts with semantic relationships
- Types: Causal, Temporal, Logical, Semantic
- Enable reasoning about information structure

#### Events
- Capture temporal sequences
- Can be compressed into summaries
- Track participants and related facts

### 2. Extraction Module

Converts raw dialogue into Memory IR:

```python
extractor = MemoryExtractor(model_name="Qwen/Qwen2.5-7B-Instruct")
ir = extractor.extract(dialogue)
```

- Uses LLM-based information extraction
- Supports mock extraction for testing
- Handles incremental updates

### 3. Optimization Passes

Transform IR to reduce size while preserving information:

#### Dead Memory Elimination
- Removes unreferenced memory units
- Eliminates superseded facts
- Considers recency and importance

#### Fact Folding
- Merges semantically equivalent facts
- Handles entailment relationships
- Uses embedding-based similarity

#### Temporal Compression
- Groups related events into sequences
- Generates summary descriptions
- Preserves narrative structure

#### Importance Pruning
- Knapsack-based selection under budget
- Multi-factor importance scoring
- Configurable weights

#### Query Prediction Pruning
- Predicts likely future queries
- Keeps query-relevant information
- Uses LLM or heuristics

### 4. Generation Module

Converts optimized IR back to natural language:

```python
generator = ContextGenerator(output_format="structured")
text = generator.generate(ir)
```

Supported formats:
- `structured`: Organized sections
- `narrative`: Flowing prose
- `bullets`: Bullet points
- `json`: Structured JSON

### 5. Compression Pipeline

Orchestrates the full process:

```python
compiler = MemoryCompiler(
    token_budget=2048,
    output_format="structured",
)
result = compiler.compress(dialogue)
```

## Data Flow

```
Dialogue → Extraction → Memory IR → Passes → Optimized IR → Generation → Compressed Text
                           ↑                     |
                           └─────────────────────┘
                           (iterative optimization)
```

## Configuration

YAML-based configuration system:

```yaml
compression:
  token_budget: 2048
  output_format: structured

passes:
  dead_memory:
    enabled: true
    recency_threshold: 10
  fact_folding:
    enabled: true
    similarity_threshold: 0.85
  importance_pruning:
    enabled: true
    min_importance: 0.1
```

## Extension Points

### Custom Passes

```python
class MyPass(OptimizationPass):
    def run(self, ir: MemoryIR) -> PassResult:
        # Custom logic
        return PassResult(modified=True)

compiler.pass_manager.add_pass(MyPass())
```

### Custom Generators

```python
class MyGenerator(ContextGenerator):
    def generate(self, ir: MemoryIR) -> str:
        # Custom generation logic
        return custom_text
```

## Performance Considerations

1. **Caching**: Embeddings and extraction results are cached
2. **Batching**: Efficient batch processing for embeddings
3. **Lazy Loading**: Models loaded on first use
4. **Incremental Processing**: Support for streaming scenarios

## Evaluation

Built-in evaluation framework:

```python
from memory_compiler.evaluation import evaluate_compression

metrics = evaluate_compression(result, original_ir, qa_pairs)
```

Metrics:
- Compression ratio
- Token reduction
- Fact recall
- Entity recall
- QA accuracy
- Semantic similarity
