"""Visualization tools for Memory IR.

This module provides tools to visualize Memory IR structures as graphs,
helping with debugging and understanding the extracted information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from memory_compiler.ir.entities import Entity
from memory_compiler.ir.facts import Fact
from memory_compiler.ir.memory_ir import MemoryIR
from memory_compiler.ir.relations import RelationType


class IRVisualizer:
    """Visualize Memory IR as interactive graphs.

    Supports multiple output formats:
    - GraphViz DOT format
    - Interactive HTML (using vis.js)
    - Mermaid diagrams
    - JSON for custom visualization

    Example:
        >>> visualizer = IRVisualizer(ir)
        >>> visualizer.save_html("memory_graph.html")
        >>> print(visualizer.to_mermaid())
    """

    def __init__(self, ir: MemoryIR) -> None:
        self.ir = ir

        # Color schemes
        self.entity_colors = {
            "person": "#4CAF50",
            "organization": "#2196F3",
            "location": "#FF9800",
            "project": "#9C27B0",
            "concept": "#00BCD4",
            "product": "#E91E63",
            "event": "#FFC107",
            "time": "#607D8B",
            "other": "#9E9E9E",
        }

        self.relation_colors = {
            "causes": "#F44336",
            "before": "#3F51B5",
            "after": "#3F51B5",
            "entails": "#4CAF50",
            "contradicts": "#F44336",
            "elaborates": "#9C27B0",
            "equivalent": "#00BCD4",
            "related": "#9E9E9E",
        }

    def to_graphviz(self) -> str:
        """Generate GraphViz DOT representation.

        Returns:
            DOT format string for GraphViz.
        """
        lines = [
            "digraph MemoryIR {",
            "    rankdir=LR;",
            "    node [shape=box, style=filled];",
            "",
        ]

        # Add entity nodes
        lines.append("    // Entities")
        for entity in self.ir.iter_entities():
            color = self.entity_colors.get(entity.type.value, "#9E9E9E")
            label = f"{entity.name}\\n({entity.type.value})"
            lines.append(
                f'    "{entity.id}" [label="{label}", fillcolor="{color}", '
                f'fontcolor="white"];'
            )

        lines.append("")

        # Add fact nodes
        lines.append("    // Facts")
        for fact in self.ir.iter_facts():
            label = f"{fact.subject}\\n{fact.predicate}\\n{fact.object}"
            importance = fact.importance_score
            # Color based on importance
            if importance >= 0.7:
                color = "#4CAF50"
            elif importance >= 0.4:
                color = "#FFC107"
            else:
                color = "#BDBDBD"
            lines.append(
                f'    "{fact.id}" [label="{label}", fillcolor="{color}", '
                f'shape=ellipse];'
            )

        lines.append("")

        # Add edges for entity-fact relationships
        lines.append("    // Entity-Fact edges")
        for fact in self.ir.iter_facts():
            subj_entity = self.ir.get_entity_by_name(fact.subject)
            obj_entity = self.ir.get_entity_by_name(fact.object)

            if subj_entity:
                lines.append(f'    "{subj_entity.id}" -> "{fact.id}" [style=dashed];')
            if obj_entity:
                lines.append(f'    "{fact.id}" -> "{obj_entity.id}" [style=dashed];')

        lines.append("")

        # Add relation edges
        lines.append("    // Relations")
        for relation in self.ir.relations.all_relations():
            color = self.relation_colors.get(relation.relation_type.value, "#9E9E9E")
            label = relation.relation_type.value
            lines.append(
                f'    "{relation.source_fact_id}" -> "{relation.target_fact_id}" '
                f'[label="{label}", color="{color}"];'
            )

        lines.append("}")

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram representation.

        Returns:
            Mermaid format string.
        """
        lines = ["graph LR"]

        # Add entity nodes
        for entity in self.ir.iter_entities():
            style = f"style {entity.id} fill:{self.entity_colors.get(entity.type.value, '#9E9E9E')}"
            lines.append(f"    {entity.id}[{entity.name}]")

        # Add fact nodes
        for fact in self.ir.iter_facts():
            fact_label = f"{fact.predicate}"
            lines.append(f"    {fact.id}(({fact_label}))")

        # Add edges
        for fact in self.ir.iter_facts():
            subj_entity = self.ir.get_entity_by_name(fact.subject)
            obj_entity = self.ir.get_entity_by_name(fact.object)

            if subj_entity and obj_entity:
                lines.append(f"    {subj_entity.id} --> {fact.id} --> {obj_entity.id}")
            elif subj_entity:
                lines.append(f"    {subj_entity.id} --> {fact.id}")

        # Add relations
        for relation in self.ir.relations.all_relations():
            lines.append(
                f"    {relation.source_fact_id} -.{relation.relation_type.value}.- "
                f"{relation.target_fact_id}"
            )

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Generate JSON representation for custom visualization.

        Returns:
            Dictionary with nodes and edges.
        """
        nodes = []
        edges = []

        # Add entity nodes
        for entity in self.ir.iter_entities():
            nodes.append({
                "id": entity.id,
                "label": entity.name,
                "type": "entity",
                "entity_type": entity.type.value,
                "color": self.entity_colors.get(entity.type.value, "#9E9E9E"),
                "importance": entity.importance_score,
                "attributes": entity.attributes,
            })

        # Add fact nodes
        for fact in self.ir.iter_facts():
            nodes.append({
                "id": fact.id,
                "label": fact.natural_language,
                "type": "fact",
                "fact_type": fact.fact_type.value,
                "importance": fact.importance_score,
                "confidence": fact.confidence,
            })

            # Add entity-fact edges
            subj_entity = self.ir.get_entity_by_name(fact.subject)
            obj_entity = self.ir.get_entity_by_name(fact.object)

            if subj_entity:
                edges.append({
                    "from": subj_entity.id,
                    "to": fact.id,
                    "type": "subject",
                    "label": "subject",
                })
            if obj_entity:
                edges.append({
                    "from": fact.id,
                    "to": obj_entity.id,
                    "type": "object",
                    "label": "object",
                })

        # Add relation edges
        for relation in self.ir.relations.all_relations():
            edges.append({
                "from": relation.source_fact_id,
                "to": relation.target_fact_id,
                "type": "relation",
                "relation_type": relation.relation_type.value,
                "color": self.relation_colors.get(relation.relation_type.value, "#9E9E9E"),
            })

        return {"nodes": nodes, "edges": edges}

    def to_html(self) -> str:
        """Generate interactive HTML visualization using vis.js.

        Returns:
            Complete HTML page with interactive graph.
        """
        graph_data = self.to_json()

        # Convert to vis.js format
        vis_nodes = []
        for node in graph_data["nodes"]:
            vis_node = {
                "id": node["id"],
                "label": node["label"][:30] + "..." if len(node["label"]) > 30 else node["label"],
                "title": node["label"],  # Tooltip
                "color": node.get("color", "#9E9E9E"),
            }
            if node["type"] == "entity":
                vis_node["shape"] = "box"
            else:
                vis_node["shape"] = "ellipse"
            vis_nodes.append(vis_node)

        vis_edges = []
        for edge in graph_data["edges"]:
            vis_edge = {
                "from": edge["from"],
                "to": edge["to"],
                "label": edge.get("label", ""),
                "arrows": "to",
            }
            if edge["type"] == "relation":
                vis_edge["color"] = edge.get("color", "#9E9E9E")
                vis_edge["dashes"] = True
            vis_edges.append(vis_edge)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Memory IR Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        #graph {{
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
        }}
        .stats {{
            margin-bottom: 20px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>Memory IR Visualization</h1>

    <div class="stats">
        <strong>Statistics:</strong>
        Entities: {len(list(self.ir.iter_entities()))} |
        Facts: {len(list(self.ir.iter_facts()))} |
        Relations: {len(self.ir.relations.all_relations())} |
        Events: {len(list(self.ir.iter_events()))}
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: #4CAF50;"></div>
            <span>Person</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #2196F3;"></div>
            <span>Organization</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9C27B0;"></div>
            <span>Project</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #00BCD4;"></div>
            <span>Concept</span>
        </div>
    </div>

    <div id="graph"></div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});

        var container = document.getElementById('graph');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                stabilization: {{
                    enabled: true,
                    iterations: 100
                }}
            }},
            nodes: {{
                font: {{ size: 12 }}
            }},
            edges: {{
                font: {{ size: 10 }},
                smooth: {{ type: 'continuous' }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};

        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

        return html

    def save_graphviz(self, path: str | Path) -> None:
        """Save GraphViz DOT file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_graphviz())
        logger.info(f"Saved GraphViz to: {path}")

    def save_html(self, path: str | Path) -> None:
        """Save interactive HTML visualization."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        logger.info(f"Saved HTML visualization to: {path}")

    def save_json(self, path: str | Path) -> None:
        """Save JSON representation."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
        logger.info(f"Saved JSON to: {path}")


def export_to_graphviz(ir: MemoryIR, path: str | Path) -> None:
    """Export Memory IR to GraphViz DOT file."""
    visualizer = IRVisualizer(ir)
    visualizer.save_graphviz(path)


def export_to_html(ir: MemoryIR, path: str | Path) -> None:
    """Export Memory IR to interactive HTML."""
    visualizer = IRVisualizer(ir)
    visualizer.save_html(path)


def print_ir_summary(ir: MemoryIR) -> str:
    """Print a text summary of Memory IR."""
    lines = []
    stats = ir.get_stats()

    lines.append("=" * 50)
    lines.append("MEMORY IR SUMMARY")
    lines.append("=" * 50)

    lines.append(f"\nEntities: {stats.num_entities}")
    for entity in ir.iter_entities():
        lines.append(f"  - {entity.name} ({entity.type.value})")

    lines.append(f"\nFacts: {stats.num_facts}")
    for fact in list(ir.iter_facts())[:10]:
        lines.append(f"  - {fact.natural_language}")
    if stats.num_facts > 10:
        lines.append(f"  ... and {stats.num_facts - 10} more")

    lines.append(f"\nRelations: {stats.num_relations}")
    for rel in ir.relations.all_relations()[:5]:
        lines.append(f"  - {rel.source_fact_id} --{rel.relation_type.value}--> {rel.target_fact_id}")

    lines.append(f"\nEvents: {stats.num_events}")
    for event in list(ir.iter_events())[:5]:
        lines.append(f"  - Turn {event.turn_index}: {event.description[:50]}...")

    lines.append("\n" + "=" * 50)

    return "\n".join(lines)
