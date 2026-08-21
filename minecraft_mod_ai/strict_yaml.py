from __future__ import annotations

"""Strict YAML loading for configuration whose mapping keys define ownership.

PyYAML's ordinary ``safe_load`` accepts duplicate mapping keys with last-writer-wins
semantics. That is unsuitable for registries where a key selects a model, provider,
role, capability, or permission owner. YAML merge keys create another implicit
precedence rule, so reviewed ownership configs reject those as well.
"""

from typing import Any

import yaml


def assert_unambiguous_yaml(text: str, *, source: str) -> None:
    """Reject duplicate/non-scalar/merged mapping ownership before construction."""

    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} YAML is invalid: {exc}") from exc
    if root is None:
        return

    def visit(node: yaml.Node, path: str) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    line = int(getattr(key_node.start_mark, "line", -1)) + 1
                    raise ValueError(
                        f"{source} mapping key at {path} line {line} must be a scalar"
                    )
                key = str(key_node.value)
                line = int(getattr(key_node.start_mark, "line", -1)) + 1
                if key == "<<":
                    raise ValueError(
                        f"{source} YAML merge keys are forbidden at {path} line {line}"
                    )
                if key in seen:
                    raise ValueError(
                        f"Duplicate {source} YAML key {key!r} at {path} line {line}"
                    )
                seen.add(key)
                visit(value_node, f"{path}.{key}")
            return
        if isinstance(node, yaml.SequenceNode):
            for index, child in enumerate(node.value):
                visit(child, f"{path}[{index}]")

    visit(root, "$")


def safe_load_unique_keys(text: str, *, source: str) -> Any:
    """Safe-load YAML only after ownership ambiguity has been ruled out."""

    assert_unambiguous_yaml(text, source=source)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} YAML is invalid: {exc}") from exc


__all__ = ["assert_unambiguous_yaml", "safe_load_unique_keys"]
