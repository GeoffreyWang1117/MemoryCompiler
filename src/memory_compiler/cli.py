"""Command-line interface for MemoryCompiler."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from memory_compiler import __version__


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MemoryCompiler: Compress dialogue memory using compiler optimization techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  memory-compiler compress input.json -o output.txt --budget 2000
  memory-compiler evaluate input.json --benchmark msc
  memory-compiler serve --port 8000
  memory-compiler merge session1.json session2.json -o merged.json
  memory-compiler visualize memory.json -o graph.html
        """,
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"MemoryCompiler {__version__}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Compress command
    _add_compress_parser(subparsers)

    # Extract command
    _add_extract_parser(subparsers)

    # Evaluate command
    _add_evaluate_parser(subparsers)

    # Benchmark command
    _add_benchmark_parser(subparsers)

    # Serve command
    _add_serve_parser(subparsers)

    # Merge command
    _add_merge_parser(subparsers)

    # Visualize command
    _add_visualize_parser(subparsers)

    # Export command
    _add_export_parser(subparsers)

    # Health command
    _add_health_parser(subparsers)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo compression")
    demo_parser.add_argument(
        "--style",
        choices=["simple", "multi-session", "streaming"],
        default="simple",
        help="Demo style",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if args.command is None:
        parser.print_help()
        return 0

    # Route to command handler
    command_handlers = {
        "compress": compress_command,
        "extract": extract_command,
        "evaluate": evaluate_command,
        "benchmark": benchmark_command,
        "serve": serve_command,
        "merge": merge_command,
        "visualize": visualize_command,
        "export": export_command,
        "health": health_command,
        "demo": demo_command,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)

    return 0


def _add_compress_parser(subparsers) -> None:
    """Add compress subparser."""
    parser = subparsers.add_parser("compress", help="Compress a dialogue")
    parser.add_argument(
        "input",
        type=str,
        help="Input file (JSON format with dialogue turns)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for compressed context",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="Token budget for compression (default: 2048)",
    )
    parser.add_argument(
        "--format",
        choices=["structured", "narrative", "bullets", "json"],
        default="structured",
        help="Output format (default: structured)",
    )
    parser.add_argument(
        "--config",
        choices=["default", "aggressive", "conservative"],
        default="default",
        help="Compression configuration preset",
    )
    parser.add_argument(
        "--passes",
        type=str,
        help="Comma-separated list of optimization passes",
    )
    parser.add_argument(
        "--save-ir",
        type=str,
        help="Save intermediate Memory IR to file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock extraction (no LLM required)",
    )


def _add_extract_parser(subparsers) -> None:
    """Add extract subparser."""
    parser = subparsers.add_parser("extract", help="Extract Memory IR from dialogue")
    parser.add_argument(
        "input",
        type=str,
        help="Input dialogue file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file for Memory IR (JSON)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock extraction",
    )


def _add_evaluate_parser(subparsers) -> None:
    """Add evaluate subparser."""
    parser = subparsers.add_parser("evaluate", help="Evaluate compression quality")
    parser.add_argument(
        "input",
        type=str,
        help="Input file with dialogue and optional QA pairs",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for evaluation results (JSON)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated list of metrics (compression,recall,rouge)",
    )


def _add_benchmark_parser(subparsers) -> None:
    """Add benchmark subparser."""
    parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation")
    parser.add_argument(
        "--dataset",
        choices=["msc", "locomo", "longbench"],
        default="msc",
        help="Benchmark dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing benchmark data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="Token budget for compression",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for results (JSON)",
    )


def _add_serve_parser(subparsers) -> None:
    """Add serve subparser."""
    parser = subparsers.add_parser("serve", help="Start the API server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--storage",
        choices=["sqlite", "file"],
        default="sqlite",
        help="Storage backend",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./memory_store.db",
        help="Path for storage",
    )


def _add_merge_parser(subparsers) -> None:
    """Add merge subparser."""
    parser = subparsers.add_parser("merge", help="Merge multiple memory IRs")
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input Memory IR files to merge",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file for merged IR",
    )
    parser.add_argument(
        "--strategy",
        choices=["union", "smart", "latest"],
        default="smart",
        help="Merge strategy",
    )


def _add_visualize_parser(subparsers) -> None:
    """Add visualize subparser."""
    parser = subparsers.add_parser("visualize", help="Visualize Memory IR")
    parser.add_argument(
        "input",
        type=str,
        help="Input Memory IR file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file for visualization",
    )
    parser.add_argument(
        "--format",
        choices=["html", "dot", "mermaid", "png"],
        default="html",
        help="Output format",
    )


def _add_export_parser(subparsers) -> None:
    """Add export subparser."""
    parser = subparsers.add_parser("export", help="Export memories to various formats")
    parser.add_argument(
        "input",
        type=str,
        help="Input (Memory IR file or storage path)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv", "parquet"],
        default="json",
        help="Export format",
    )


def _add_health_parser(subparsers) -> None:
    """Add health subparser."""
    parser = subparsers.add_parser("health", help="Check system health")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )


def compress_command(args: argparse.Namespace) -> int:
    """Handle the compress command."""
    from memory_compiler.pipeline import MemoryCompiler

    # Load input dialogue
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both list of turns and dict with "dialogue" key
    if isinstance(data, list):
        dialogue = "\n".join(
            f"{t.get('role', 'User')}: {t.get('content', t)}"
            for t in data
        )
    elif isinstance(data, dict) and "dialogue" in data:
        if isinstance(data["dialogue"], str):
            dialogue = data["dialogue"]
        else:
            dialogue = "\n".join(
                f"{t.get('role', 'User')}: {t.get('content', t)}"
                for t in data["dialogue"]
            )
    elif isinstance(data, str):
        dialogue = data
    else:
        logger.error("Input must be a list of turns, dict with 'dialogue' key, or string")
        return 1

    # Create compiler
    compiler = MemoryCompiler(use_mock=args.mock)

    logger.info(f"Compressing dialogue with budget {args.budget} tokens...")

    # Compress
    ir = compiler.compile(dialogue, token_budget=args.budget)
    compressed_text = compiler.generate(ir, style=args.format)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(compressed_text)
        logger.info(f"Compressed context saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("COMPRESSED CONTEXT")
        print("=" * 60)
        print(compressed_text)
        print("=" * 60)

    # Print stats
    original_tokens = len(dialogue.split()) * 1.3
    compressed_tokens = len(compressed_text.split()) * 1.3
    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

    print(f"\nCompression Statistics:")
    print(f"  Original tokens:   ~{int(original_tokens)}")
    print(f"  Compressed tokens: ~{int(compressed_tokens)}")
    print(f"  Compression ratio: {ratio:.1%}")
    print(f"  Entities: {len(list(ir.iter_entities()))}")
    print(f"  Facts: {len(list(ir.iter_facts()))}")

    # Save IR if requested
    if args.save_ir:
        with open(args.save_ir, "w", encoding="utf-8") as f:
            json.dump(ir.to_dict(), f, indent=2)
        logger.info(f"Memory IR saved to: {args.save_ir}")

    return 0


def extract_command(args: argparse.Namespace) -> int:
    """Handle the extract command."""
    from memory_compiler.pipeline import MemoryCompiler

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try to parse as JSON, otherwise treat as plain text
    try:
        data = json.loads(content)
        if isinstance(data, list):
            dialogue = "\n".join(
                f"{t.get('role', 'User')}: {t.get('content', t)}"
                for t in data
            )
        elif isinstance(data, dict) and "dialogue" in data:
            dialogue = data["dialogue"] if isinstance(data["dialogue"], str) else "\n".join(
                f"{t.get('role', 'User')}: {t.get('content', t)}"
                for t in data["dialogue"]
            )
        else:
            dialogue = content
    except json.JSONDecodeError:
        dialogue = content

    compiler = MemoryCompiler(use_mock=args.mock)
    ir = compiler.extract(dialogue)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ir.to_dict(), f, indent=2)

    logger.info(f"Memory IR saved to: {args.output}")
    print(f"Extracted: {len(list(ir.iter_entities()))} entities, {len(list(ir.iter_facts()))} facts")

    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    """Handle the evaluate command."""
    from memory_compiler.pipeline import MemoryCompiler

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract dialogue
    if isinstance(data, list):
        dialogue = "\n".join(f"{t.get('role', 'User')}: {t.get('content', t)}" for t in data)
    else:
        dialogue = data.get("dialogue", "")
        if isinstance(dialogue, list):
            dialogue = "\n".join(f"{t.get('role', 'User')}: {t.get('content', t)}" for t in dialogue)

    # Compress
    compiler = MemoryCompiler(use_mock=True)
    ir = compiler.compile(dialogue, token_budget=2048)
    compressed = compiler.generate(ir)

    # Calculate metrics
    original_tokens = len(dialogue.split()) * 1.3
    compressed_tokens = len(compressed.split()) * 1.3

    metrics = {
        "original_tokens": int(original_tokens),
        "compressed_tokens": int(compressed_tokens),
        "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 0,
        "entities_extracted": len(list(ir.iter_entities())),
        "facts_extracted": len(list(ir.iter_facts())),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation results saved to: {args.output}")
    else:
        print("\nEvaluation Results:")
        print(json.dumps(metrics, indent=2))

    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    """Handle the benchmark command."""
    from memory_compiler.benchmarks import (
        MSCBenchmark,
        LoCoMoBenchmark,
        LongBenchBenchmark,
        BenchmarkRunner,
    )

    # Select dataset
    dataset_map = {
        "msc": MSCBenchmark,
        "locomo": LoCoMoBenchmark,
        "longbench": LongBenchBenchmark,
    }

    dataset_class = dataset_map.get(args.dataset)
    if not dataset_class:
        logger.error(f"Unknown dataset: {args.dataset}")
        return 1

    # Create dataset
    dataset = dataset_class(data_dir=args.data_dir)
    dataset.load()

    logger.info(f"Running benchmark: {dataset.name} ({len(dataset)} samples)")

    # Run benchmark
    runner = BenchmarkRunner(token_budget=args.budget)

    def progress_callback(current: int, total: int):
        print(f"\rProgress: {current}/{total}", end="", flush=True)

    result = runner.run(
        dataset,
        max_samples=args.max_samples,
        progress_callback=progress_callback,
    )

    print()  # New line after progress

    # Output
    if args.output:
        result.save(args.output)
    else:
        print("\nBenchmark Results:")
        print(f"  Dataset: {result.benchmark_name}")
        print(f"  Samples: {result.num_samples}")
        print(f"  Compression Ratio: {result.metrics.compression_ratio:.2f}x")
        print(f"  Entity Recall: {result.metrics.entity_recall:.2%}")
        print(f"  Fact Recall: {result.metrics.fact_recall:.2%}")
        print(f"  Avg Latency: {result.metrics.latency_ms:.1f}ms")

    return 0


def serve_command(args: argparse.Namespace) -> int:
    """Handle the serve command."""
    from memory_compiler.api.app import run_server

    logger.info(f"Starting API server on {args.host}:{args.port}")

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

    return 0


def merge_command(args: argparse.Namespace) -> int:
    """Handle the merge command."""
    from memory_compiler.ir.memory_ir import MemoryIR
    from memory_compiler.merging import MemoryMerger, MergeStrategy, MergeConfig

    # Load input IRs
    memories = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            logger.error(f"Input file not found: {path}")
            return 1

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ir = MemoryIR.from_dict(data)
        memories.append(ir)
        logger.debug(f"Loaded IR from {path}")

    # Merge
    strategy_map = {
        "union": MergeStrategy.UNION,
        "smart": MergeStrategy.SMART,
        "latest": MergeStrategy.LATEST,
    }

    config = MergeConfig(strategy=strategy_map.get(args.strategy, MergeStrategy.SMART))
    merger = MemoryMerger(config=config)

    result = merger.merge(memories)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result.merged_ir.to_dict(), f, indent=2)

    logger.info(f"Merged {len(memories)} IRs to: {args.output}")
    print(f"Merged: {result.stats}")

    return 0


def visualize_command(args: argparse.Namespace) -> int:
    """Handle the visualize command."""
    from memory_compiler.ir.memory_ir import MemoryIR
    from memory_compiler.visualization import IRVisualizer

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ir = MemoryIR.from_dict(data)
    visualizer = IRVisualizer(ir)

    output_path = Path(args.output)

    if args.format == "html":
        visualizer.save_html(str(output_path))
    elif args.format == "dot":
        content = visualizer.to_graphviz()
        output_path.write_text(content)
    elif args.format == "mermaid":
        content = visualizer.to_mermaid()
        output_path.write_text(content)
    elif args.format == "png":
        # Requires graphviz installed
        dot_content = visualizer.to_graphviz()
        dot_path = output_path.with_suffix(".dot")
        dot_path.write_text(dot_content)
        import subprocess
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(output_path)])
        dot_path.unlink()

    logger.info(f"Visualization saved to: {output_path}")
    return 0


def export_command(args: argparse.Namespace) -> int:
    """Handle the export command."""
    from memory_compiler.ir.memory_ir import MemoryIR

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ir = MemoryIR.from_dict(data)
    output_path = Path(args.output)

    if args.format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ir.to_dict(), f, indent=2)

    elif args.format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for entity in ir.iter_entities():
                f.write(json.dumps({"type": "entity", **entity.to_dict()}) + "\n")
            for fact in ir.iter_facts():
                f.write(json.dumps({"type": "fact", **fact.to_dict()}) + "\n")

    elif args.format == "csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "id", "content", "importance"])
            for entity in ir.iter_entities():
                writer.writerow(["entity", entity.id, entity.name, entity.importance_score])
            for fact in ir.iter_facts():
                content = f"{fact.subject} {fact.predicate} {fact.object}"
                writer.writerow(["fact", fact.id, content, fact.importance_score])

    elif args.format == "parquet":
        try:
            import pandas as pd
            rows = []
            for entity in ir.iter_entities():
                rows.append({
                    "type": "entity",
                    "id": entity.id,
                    "content": entity.name,
                    "importance": entity.importance_score,
                })
            for fact in ir.iter_facts():
                rows.append({
                    "type": "fact",
                    "id": fact.id,
                    "content": f"{fact.subject} {fact.predicate} {fact.object}",
                    "importance": fact.importance_score,
                })
            df = pd.DataFrame(rows)
            df.to_parquet(output_path)
        except ImportError:
            logger.error("pandas and pyarrow required for parquet export")
            return 1

    logger.info(f"Exported to: {output_path}")
    return 0


def health_command(args: argparse.Namespace) -> int:
    """Handle the health command."""
    from memory_compiler.monitoring.health import quick_health_check

    report = quick_health_check()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Status: {report['status']}")
        print(f"Uptime: {report['uptime_seconds']:.1f}s")
        print("\nComponents:")
        for component in report.get("components", []):
            status_icon = "✓" if component["status"] == "healthy" else "✗"
            print(f"  {status_icon} {component['name']}: {component['status']}")
            if component.get("message"):
                print(f"    {component['message']}")

    return 0 if report["status"] == "healthy" else 1


def demo_command(args: argparse.Namespace) -> int:
    """Run a demo compression."""
    from memory_compiler.pipeline import MemoryCompiler

    print("MemoryCompiler Demo")
    print("=" * 60)

    if args.style == "streaming":
        return _demo_streaming()
    elif args.style == "multi-session":
        return _demo_multi_session()
    else:
        return _demo_simple()


def _demo_simple() -> int:
    """Simple demo."""
    from memory_compiler.pipeline import MemoryCompiler

    dialogue = """
User: Hi, my name is Alice and I'm a software engineer at TechCorp.
Assistant: Hello Alice! It's nice to meet you. How can I help you today?
User: I'm working on a machine learning project and need some advice.
Assistant: I'd be happy to help with your ML project. What specific aspect are you working on?
User: I'm trying to optimize the training pipeline. It's taking too long.
Assistant: Training optimization is important. Have you tried techniques like mixed precision training?
User: I've tried mixed precision but haven't looked into gradient accumulation yet.
Assistant: Gradient accumulation can help when you're memory-constrained.
"""

    print("\nOriginal Dialogue:")
    print("-" * 40)
    print(dialogue.strip())

    print("\n" + "=" * 60)
    print("Compressing...")

    compiler = MemoryCompiler(use_mock=True)
    ir = compiler.compile(dialogue, token_budget=500)
    compressed = compiler.generate(ir, style="structured")

    print("\nCompressed Context:")
    print("-" * 40)
    print(compressed)

    print("\n" + "=" * 60)
    original_tokens = len(dialogue.split()) * 1.3
    compressed_tokens = len(compressed.split()) * 1.3

    print("Statistics:")
    print(f"  Original tokens:   ~{int(original_tokens)}")
    print(f"  Compressed tokens: ~{int(compressed_tokens)}")
    print(f"  Reduction: {(1 - compressed_tokens/original_tokens):.1%}")
    print(f"  Entities: {len(list(ir.iter_entities()))}")
    print(f"  Facts: {len(list(ir.iter_facts()))}")

    return 0


def _demo_multi_session() -> int:
    """Multi-session merging demo."""
    from memory_compiler.pipeline import MemoryCompiler
    from memory_compiler.merging import MemoryMerger

    print("\nMulti-Session Merging Demo")
    print("-" * 40)

    session1 = """
User: I'm planning a trip to Japan next month.
Assistant: How exciting! Which cities are you visiting?
User: Tokyo and Kyoto. I love Japanese food.
Assistant: Both are wonderful. Try ramen in Tokyo!
"""

    session2 = """
User: Remember my Japan trip? I'm back!
Assistant: Welcome back! How was it?
User: Amazing! The ramen in Tokyo was incredible.
Assistant: I'm glad you enjoyed it! Any favorite spots?
User: Ichiran was my favorite ramen place.
"""

    compiler = MemoryCompiler(use_mock=True)

    ir1 = compiler.extract(session1)
    ir2 = compiler.extract(session2)

    print(f"Session 1: {len(list(ir1.iter_facts()))} facts")
    print(f"Session 2: {len(list(ir2.iter_facts()))} facts")

    merger = MemoryMerger()
    result = merger.merge([ir1, ir2])

    print(f"\nMerged: {len(list(result.merged_ir.iter_facts()))} facts")
    print(f"Entity matches: {len(result.entity_matches)}")

    compressed = compiler.generate(result.merged_ir)
    print(f"\nMerged Memory:\n{compressed}")

    return 0


def _demo_streaming() -> int:
    """Streaming compression demo."""
    import asyncio
    from memory_compiler.streaming import StreamingProcessor, DialogueTurn

    print("\nStreaming Compression Demo")
    print("-" * 40)

    async def run_streaming():
        processor = StreamingProcessor()
        await processor.start()

        turns = [
            DialogueTurn(role="user", content="Hi, I'm Bob and I work at Google."),
            DialogueTurn(role="assistant", content="Nice to meet you, Bob!"),
            DialogueTurn(role="user", content="I'm a software engineer on the Search team."),
            DialogueTurn(role="assistant", content="Search is a fascinating area!"),
            DialogueTurn(role="user", content="We're working on improving voice queries."),
        ]

        for turn in turns:
            print(f"  Processing: {turn.role}: {turn.content[:40]}...")
            await processor.add_turn(turn)

        context = await processor.get_context()
        stats = processor.get_stats()

        await processor.stop()

        print(f"\nCurrent Context:\n{context}")
        print(f"\nStats: {stats}")

    asyncio.run(run_streaming())
    return 0


if __name__ == "__main__":
    sys.exit(main())
