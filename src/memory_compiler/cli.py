"""Command-line interface for MemoryCompiler."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from memory_compiler.pipeline import MemoryCompiler, create_compiler


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MemoryCompiler: Compress dialogue memory using compiler optimization techniques"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a dialogue")
    compress_parser.add_argument(
        "input",
        type=str,
        help="Input file (JSON format with dialogue turns)",
    )
    compress_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for compressed context",
    )
    compress_parser.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="Token budget for compression (default: 2048)",
    )
    compress_parser.add_argument(
        "--format",
        choices=["structured", "narrative", "bullets", "json"],
        default="structured",
        help="Output format (default: structured)",
    )
    compress_parser.add_argument(
        "--config",
        choices=["default", "aggressive", "conservative"],
        default="default",
        help="Compression configuration preset",
    )
    compress_parser.add_argument(
        "--save-ir",
        type=str,
        help="Save intermediate Memory IR to file",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate compression quality")
    eval_parser.add_argument(
        "input",
        type=str,
        help="Input file with dialogue and optional QA pairs",
    )
    eval_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for evaluation results (JSON)",
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo compression")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "compress":
        return compress_command(args)
    elif args.command == "evaluate":
        return evaluate_command(args)
    elif args.command == "demo":
        return demo_command(args)

    return 0


def compress_command(args: argparse.Namespace) -> int:
    """Handle the compress command."""
    # Load input dialogue
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both list of turns and dict with "dialogue" key
    if isinstance(data, list):
        dialogue = data
    elif isinstance(data, dict) and "dialogue" in data:
        dialogue = data["dialogue"]
    else:
        logger.error("Input must be a list of turns or dict with 'dialogue' key")
        return 1

    # Create compiler
    compiler = create_compiler(args.config)
    compiler.token_budget = args.budget
    compiler.generator.output_format = args.format

    logger.info(f"Compressing dialogue with {len(dialogue)} turns...")

    # Compress
    result = compiler.compress(dialogue)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.text)
        logger.info(f"Compressed context saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("COMPRESSED CONTEXT")
        print("=" * 60)
        print(result.text)
        print("=" * 60)

    # Print stats
    print(f"\nCompression Statistics:")
    print(f"  Original tokens:   {result.original_tokens}")
    print(f"  Compressed tokens: {result.compressed_tokens}")
    print(f"  Compression ratio: {result.compression_ratio:.1%}")

    # Save IR if requested
    if args.save_ir:
        compiler.save_ir(result.ir, args.save_ir)
        logger.info(f"Memory IR saved to: {args.save_ir}")

    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Handle the evaluate command."""
    from memory_compiler.evaluation.metrics import evaluate_compression

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogue = data.get("dialogue", data if isinstance(data, list) else [])
    qa_pairs = data.get("qa_pairs", [])
    qa_pairs = [(q["question"], q["answer"]) for q in qa_pairs] if qa_pairs else None

    # Compress
    compiler = create_compiler("default")
    result = compiler.compress(dialogue)

    # Build original text
    original_text = "\n".join(f"{t['role']}: {t['content']}" for t in dialogue)

    # Evaluate
    metrics = evaluate_compression(
        result,
        qa_pairs=qa_pairs,
        original_text=original_text,
    )

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation results saved to: {args.output}")
    else:
        print("\nEvaluation Results:")
        print(json.dumps(metrics, indent=2))

    return 0


def demo_command(args: argparse.Namespace) -> int:
    """Run a demo compression."""
    print("MemoryCompiler Demo")
    print("=" * 60)

    # Sample dialogue
    dialogue = [
        {"role": "user", "content": "Hi, my name is Alice and I'm a software engineer at TechCorp."},
        {"role": "assistant", "content": "Hello Alice! It's nice to meet you. How can I help you today?"},
        {"role": "user", "content": "I'm working on a machine learning project and need some advice."},
        {"role": "assistant", "content": "I'd be happy to help with your ML project. What specific aspect are you working on?"},
        {"role": "user", "content": "I'm trying to optimize the training pipeline. It's taking too long."},
        {"role": "assistant", "content": "Training optimization is important. Have you tried techniques like mixed precision training or gradient accumulation?"},
        {"role": "user", "content": "I've tried mixed precision but haven't looked into gradient accumulation yet."},
        {"role": "assistant", "content": "Gradient accumulation can help when you're memory-constrained. It allows you to effectively use larger batch sizes."},
        {"role": "user", "content": "That sounds useful. Also, I'm having issues with data loading being a bottleneck."},
        {"role": "assistant", "content": "Data loading bottlenecks are common. Consider using multiple workers, prefetching, and caching preprocessed data."},
    ]

    print("\nOriginal Dialogue:")
    print("-" * 40)
    for turn in dialogue:
        print(f"{turn['role'].upper()}: {turn['content']}")

    # Compress
    print("\n" + "=" * 60)
    print("Compressing...")

    compiler = MemoryCompiler(
        use_mock_extraction=True,
        token_budget=500,
        output_format="structured",
    )

    result = compiler.compress(dialogue)

    print("\nCompressed Context:")
    print("-" * 40)
    print(result.text)

    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Original tokens:   {result.original_tokens}")
    print(f"  Compressed tokens: {result.compressed_tokens}")
    print(f"  Compression ratio: {result.compression_ratio:.1%}")
    print(f"  Token reduction:   {(1 - result.compression_ratio):.1%}")

    print("\nMemory IR Stats:")
    stats = result.ir.get_stats()
    print(f"  Entities: {stats.num_entities}")
    print(f"  Facts:    {stats.num_facts}")
    print(f"  Events:   {stats.num_events}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
