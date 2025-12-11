# MemoryCompiler

**基于编译优化理论的长对话记忆压缩框架**

A memory compression framework for long conversations based on compiler optimization theory.

## Overview

MemoryCompiler treats dialogue memory compression as a compiler optimization problem:
- **Source Code**: Original dialogue history
- **Target Code**: Compressed context
- **Optimization Goal**: Maximize information retention under token budget constraints

## Key Features

- **Memory IR (Intermediate Representation)**: Structured representation with entities, facts, relations, and temporal events
- **Optimization Passes**:
  - Dead Memory Elimination: Remove unreferenced memory units
  - Fact Folding: Merge semantically equivalent or entailed facts
  - Temporal Compression: Summarize related event sequences
  - Importance Pruning: Select optimal memory subset via knapsack optimization

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from memory_compiler import MemoryCompiler

# Initialize compiler
compiler = MemoryCompiler(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    token_budget=2048
)

# Compress dialogue history
dialogue = [
    {"role": "user", "content": "My name is Alice and I work at TechCorp."},
    {"role": "assistant", "content": "Nice to meet you, Alice! What do you do at TechCorp?"},
    {"role": "user", "content": "I'm a software engineer working on ML systems."},
    # ... more turns
]

compressed = compiler.compress(dialogue)
print(compressed.text)
print(f"Compression ratio: {compressed.compression_ratio:.2f}")
```

## Project Structure

```
src/memory_compiler/
├── ir/                     # Memory IR definitions
│   ├── entities.py         # Entity representations
│   ├── facts.py            # Fact triples
│   ├── relations.py        # Semantic relations
│   └── memory_ir.py        # Main IR class
├── extraction/             # Dialogue to IR conversion
│   └── extractor.py        # Information extraction
├── passes/                 # Optimization passes
│   ├── base.py             # Base pass interface
│   ├── dead_memory.py      # Dead memory elimination
│   ├── fact_folding.py     # Fact folding
│   ├── temporal.py         # Temporal compression
│   └── importance.py       # Importance pruning
├── generation/             # IR to text conversion
│   └── generator.py        # Text generation
├── pipeline.py             # Main compression pipeline
└── evaluation/             # Evaluation metrics
    └── metrics.py          # Evaluation utilities
```

## Evaluation

Supported datasets:
- Multi-Session Chat (MSC)
- LoCoMo
- LongBench
- MultiWOZ

Run evaluation:
```bash
python -m memory_compiler.evaluation.run --dataset msc --compression-ratio 0.3
```

## Citation

If you use MemoryCompiler in your research, please cite:

```bibtex
@article{memorycompiler2024,
  title={MemoryCompiler: A Compiler-Inspired Framework for Long Conversation Memory Compression},
  author={MemoryCompiler Team},
  year={2024}
}
```

## License

MIT License
