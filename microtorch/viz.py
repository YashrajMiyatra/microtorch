"""
MicroTorch Computation Graph Visualizer.
Renders backward pass computational DAGs into ASCII, Mermaid, or Graphviz DOT formats.
"""

from typing import Set, Tuple
from .tensor import Tensor


def trace_graph(root: Tensor) -> Tuple[Set[Tensor], Set[Tuple[Tensor, Tensor]]]:
    """Traces all nodes and directed edges in the computation DAG."""
    nodes: Set[Tensor] = set()
    edges: Set[Tuple[Tensor, Tensor]] = set()

    def build(v: Tensor):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def to_mermaid(root: Tensor) -> str:
    """Generates a Mermaid graph string compatible with GitHub markdown."""
    nodes, edges = trace_graph(root)
    lines = ["```mermaid", "graph TD"]

    node_ids = {n: f"node_{i}" for i, n in enumerate(nodes)}

    for n, uid in node_ids.items():
        shape_str = f"{n.shape}" if n.shape else "scalar"
        op_label = f"[{n._op}] " if n._op else ""
        label = f"{op_label}{shape_str}"
        lines.append(f'    {uid}["{label}"]')

    for src, dst in edges:
        lines.append(f"    {node_ids[src]} --> {node_ids[dst]}")

    lines.append("```")
    return "\n".join(lines)


def to_ascii_tree(root: Tensor, indent: str = "", last: bool = True) -> str:
    """Renders a simple ASCII tree representation of the computation graph."""
    branch = "└── " if last else "├── "
    op_str = f" [op: {root._op}]" if root._op else ""
    shape_str = f"shape={root.shape}"
    line = f"{indent}{branch}Tensor({shape_str}{op_str})\n"

    children = list(root._prev)
    indent_next = indent + ("    " if last else "│   ")
    for i, child in enumerate(children):
        line += to_ascii_tree(child, indent_next, last=(i == len(children) - 1))
    return line
