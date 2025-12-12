#!/usr/bin/env python3
"""Example: Memory IR visualization.

This example demonstrates how to visualize Memory IR as interactive
graphs using different output formats.
"""

import tempfile
from pathlib import Path

from memory_compiler import MemoryCompiler
from memory_compiler.visualization import IRVisualizer


def main():
    # Initialize compiler
    compiler = MemoryCompiler(use_mock=True)

    # Sample complex dialogue
    dialogue = """
    User: I'm Sarah, and I just got a new job at Microsoft in Seattle.
    Assistant: Congratulations Sarah! Microsoft is a great company. What role?
    User: I'll be working as a Senior Product Manager on the Azure team.
    Assistant: Azure is growing rapidly. Will you be relocating?
    User: Yes, I'm moving from San Francisco. My husband Tom will work remotely.
    Assistant: That's a big change! When do you start?
    User: Next month, March 15th. We're excited but nervous about the move.
    Assistant: Moving across states is challenging. Do you have housing sorted?
    User: We found an apartment in Bellevue, near the Microsoft campus.
    Assistant: Bellevue is lovely and very convenient for work.
    User: Tom is a software engineer at Google, so he'll continue remotely.
    Assistant: Remote work makes relocations much more feasible nowadays.
    """

    # Extract Memory IR
    ir = compiler.compile(dialogue, token_budget=2048)

    print("Memory IR Statistics:")
    print(f"  Entities: {len(list(ir.iter_entities()))}")
    print(f"  Facts: {len(list(ir.iter_facts()))}")
    print(f"  Events: {len(list(ir.iter_events()))}")
    print(f"  Relations: {len(ir.relations)}")
    print()

    # Create visualizer
    visualizer = IRVisualizer(ir)

    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ========== GraphViz DOT Format ==========
        print("=" * 60)
        print("GraphViz DOT Output")
        print("=" * 60)

        dot_output = visualizer.to_graphviz()
        print(dot_output[:500] + "...")

        dot_path = tmpdir / "memory_graph.dot"
        with open(dot_path, "w") as f:
            f.write(dot_output)
        print(f"\nSaved DOT file to: {dot_path}")
        print("Convert to PNG with: dot -Tpng memory_graph.dot -o memory_graph.png")

        # ========== Mermaid Diagram ==========
        print("\n" + "=" * 60)
        print("Mermaid Diagram Output")
        print("=" * 60)

        mermaid_output = visualizer.to_mermaid()
        print(mermaid_output)

        mermaid_path = tmpdir / "memory_graph.mmd"
        with open(mermaid_path, "w") as f:
            f.write(mermaid_output)
        print(f"\nSaved Mermaid file to: {mermaid_path}")
        print("Use https://mermaid.live/ to render")

        # ========== Interactive HTML ==========
        print("\n" + "=" * 60)
        print("Interactive HTML Output")
        print("=" * 60)

        html_path = tmpdir / "memory_visualization.html"
        visualizer.save_html(str(html_path))
        print(f"Saved interactive HTML to: {html_path}")

        # Show snippet of HTML
        html_content = visualizer.to_html()
        print(f"\nHTML size: {len(html_content)} bytes")
        print("Open the HTML file in a browser for interactive exploration")

        # ========== Summary Statistics ==========
        print("\n" + "=" * 60)
        print("Visualization Summary")
        print("=" * 60)

        # Count node types
        entity_count = len(list(ir.iter_entities()))
        fact_count = len(list(ir.iter_facts()))
        event_count = len(list(ir.iter_events()))

        print(f"Graph contains:")
        print(f"  - {entity_count} entity nodes (blue)")
        print(f"  - {fact_count} fact nodes (green)")
        print(f"  - {event_count} event nodes (orange)")
        print(f"  - {len(ir.relations)} relation edges")

        # Show entities
        print("\nEntities:")
        for entity in ir.iter_entities():
            print(f"  - {entity.name} ({entity.type.value}) - importance: {entity.importance_score:.2f}")

        # Show key facts
        print("\nKey Facts:")
        for fact in list(ir.iter_facts())[:5]:
            print(f"  - {fact.subject} → {fact.predicate} → {fact.object}")

        print("\n" + "=" * 60)
        print("Visualization examples completed!")
        print("Check the generated files for graph outputs.")


if __name__ == "__main__":
    main()
