"""Tree traversal utilities."""

from __future__ import annotations

import copy
from typing import Any


def create_node_mapping(tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a flat dict mapping node id → node for every node in the tree."""
    mapping: dict[str, dict[str, Any]] = {}

    def _walk(node: dict[str, Any]) -> None:
        nid = node.get("id")
        if nid:
            mapping[nid] = node
        for child in node.get("children", []):
            _walk(child)

    _walk(tree)
    return mapping


def remove_fields(
    obj: dict[str, Any] | list[Any],
    fields: list[str],
) -> dict[str, Any] | list[Any]:
    """Return a deep copy of *obj* with the specified fields removed at all levels."""
    obj = copy.deepcopy(obj)

    def _strip(node: Any) -> None:
        if isinstance(node, dict):
            for f in fields:
                node.pop(f, None)
            for v in node.values():
                _strip(v)
        elif isinstance(node, list):
            for item in node:
                _strip(item)

    _strip(obj)
    return obj


def get_all_leaf_nodes(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all leaf nodes (nodes with no children) in DFS order."""
    leaves: list[dict[str, Any]] = []

    def _walk(node: dict[str, Any]) -> None:
        children = node.get("children", [])
        if not children:
            leaves.append(node)
        else:
            for child in children:
                _walk(child)

    _walk(tree)
    return leaves


def pretty_print_tree(node: dict[str, Any], indent: int = 0) -> str:
    """Return a human-readable tree representation."""
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    lines = [f"{prefix}[{node.get('id', '?')}] {node.get('title', '')}"]
    summary = node.get("summary", "")
    if summary:
        lines.append("  " * (indent + 1) + f"↳ {summary[:80]}")
    for child in node.get("children", []):
        lines.append(pretty_print_tree(child, indent + 1))
    return "\n".join(lines)
