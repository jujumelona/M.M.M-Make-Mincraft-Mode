"""Compile authored prose into a bounded implementation DAG, never heading slots.

The model supplies semantic responsibility boundaries. The host owns paths, coverage,
dependency order, output admission and finite refinement. This does not rewrite or
judge the saved gameplay design.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .custom_module_errors import CustomModuleGenerationError

MAX_NODES = 128
MAX_PAGES = 32
MAX_REFINEMENTS = 6


class ImplementationGraphError(CustomModuleGenerationError):
    pass


class OutputBudgetExhausted(CustomModuleGenerationError):
    """A decomposition transition; never a compiler repair request."""


def output_token_ceiling() -> int:
    try:
        value = int(os.environ.get("MMM_DIRECT_CODER_OUTPUT_TOKEN_CEILING", "4096"))
    except ValueError:
        value = 4096
    return max(1024, min(value, 8192))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


_STRINGS = {"type": "array", "items": {"type": "string"}}
NODE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["symbol", "kind", "resource_path", "responsibility", "requirements",
                 "obligations", "public_api", "depends_on", "activation", "estimated_tokens"],
    "properties": {
        "symbol": {"type": "string"},
        "kind": {"type": "string", "enum": ["java", "resource"]},
        "resource_path": {"type": "string"},
        "responsibility": {"type": "string"},
        "requirements": _STRINGS, "obligations": _STRINGS,
        "public_api": _STRINGS, "depends_on": _STRINGS,
        "activation": {"type": "boolean"},
        "estimated_tokens": {"type": "integer", "minimum": 1},
    },
}
PAGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["nodes", "done"],
    "properties": {
        "nodes": {"type": "array", "minItems": 1, "maxItems": 4, "items": NODE_SCHEMA},
        "done": {"type": "boolean"},
    },
}


def source_requirements(text: str) -> dict[str, str]:
    # Stable provenance, not task boundaries. A node can cite many lines across any
    # number of sections, and a line can be implemented by multiple collaborating nodes.
    return {f"R{i}": line for i, line in enumerate(text.splitlines(), 1) if line.strip()}


def node_cost(node: Mapping[str, Any]) -> int:
    # Conservative host floor plus the semantic compiler's whole-file estimate.
    # This is admission estimation, not a guarantee; runtime exhaustion refines the DAG.
    api_bytes = len(json.dumps(node["public_api"], ensure_ascii=False).encode())
    obligation_bytes = len(json.dumps(node["obligations"], ensure_ascii=False).encode())
    floor = 384 + api_bytes // 2 + max(obligation_bytes // 2, 256 * len(node["obligations"]))
    return max(node["estimated_tokens"], floor)


def admissible_tokens() -> int:
    return output_token_ceiling() * 3 // 4


def _decision(router: Any, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    callback = getattr(router, "generate_tool_decision", None)
    if not callable(callback):
        raise ImplementationGraphError("IMPLEMENTATION_IR_NATIVE_DECISION_REQUIRED")
    response = callback(
        "planner",
        [{"role": "system", "content": (
            "Compile the approved design into an implementation DAG, without changing gameplay. "
            "Markdown headings describe concerns, NOT Java classes. Group requirements across "
            "sections by state ownership and coherent responsibility. Separate state/models, "
            "persistence, services, integration, networking, UI/resources and verification where "
            "the design needs them. Cite every R identifier at least once. Return at most four "
            "nodes per page, then explicitly done=true. Each Java node is one public final class "
            "with a concrete name and frozen public_api declaration strings (no bodies). Include "
            "constructors/fields/methods used by consumers. List actual symbol dependencies, "
            "never an artificial previous-section chain. No dependency cycles. Only integration "
            "nodes need activation=true and public static void initialize(). Other classes may "
            "have constructors and state without lifecycle methods. JSON resources use kind "
            "resource and a path under the approved assets/data namespace. Java verification "
            "nodes must contain executable tests for the selected platform. Preserve all "
            "approved requirements, including failure handling and tests. Do not claim tests ran. "
            "Estimate COMPLETE serialized source output tokens, not just method bodies. "
            "Respect the host admission budget by factoring smaller collaborating types. "
            "For decomposition return 2-4 nodes, done=true: retain the original symbol, kind, "
            "resource_path, activation and exact public_api as a smaller facade; use helper "
            "symbols prefixed with OriginalSymbolPart. Move work into those helpers. Preserve "
            "all original requirement refs. Every replacement must be strictly smaller than "
            "the rejected task. Dependencies outside the replacement must already exist."
        )}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        tool_name=name, parameters=PAGE_SCHEMA,
        description="Host-owned implementation graph compilation; does not mutate source.",
    )
    if not isinstance(response, dict) or type(response.get("done")) is not bool:
        raise ImplementationGraphError("IMPLEMENTATION_IR_INVALID_PAGE")
    nodes = response.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 4:
        raise ImplementationGraphError("IMPLEMENTATION_IR_PAGE_LIMIT")
    return response


def validate_node(raw: Any, *, package: str, mod_id: str, refs: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != set(NODE_SCHEMA["required"]):
        raise ImplementationGraphError("IMPLEMENTATION_IR_INVALID_NODE")
    node = dict(raw)
    symbol = node["symbol"]
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Za-z0-9_]{0,95}", symbol):
        raise ImplementationGraphError("IMPLEMENTATION_IR_INVALID_SYMBOL")
    for key in ("requirements", "obligations", "public_api", "depends_on"):
        values = node[key]
        if not isinstance(values, list) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ImplementationGraphError(f"IMPLEMENTATION_IR_INVALID_{key.upper()}")
        if len(values) != len(set(values)):
            raise ImplementationGraphError(f"IMPLEMENTATION_IR_DUPLICATE_{key.upper()}")
    if not node["requirements"] or not set(node["requirements"]) <= refs or not node["obligations"]:
        raise ImplementationGraphError("IMPLEMENTATION_IR_REQUIREMENT_COVERAGE")
    if not isinstance(node["responsibility"], str) or not node["responsibility"].strip():
        raise ImplementationGraphError("IMPLEMENTATION_IR_RESPONSIBILITY_REQUIRED")
    if type(node["estimated_tokens"]) is not int or node["estimated_tokens"] < 1:
        raise ImplementationGraphError("IMPLEMENTATION_IR_INVALID_ESTIMATE")
    if type(node["activation"]) is not bool:
        raise ImplementationGraphError("IMPLEMENTATION_IR_INVALID_ACTIVATION")
    if node["kind"] == "java":
        if node["resource_path"] or not node["public_api"]:
            raise ImplementationGraphError("IMPLEMENTATION_IR_JAVA_API_REQUIRED")
        if any("{" in api or "}" in api for api in node["public_api"]):
            raise ImplementationGraphError("IMPLEMENTATION_IR_API_BODY_FORBIDDEN")
        node["public_api"] = [re.sub(r"\s+", " ", api.strip().rstrip(";").strip()) for api in node["public_api"]]
        if node["activation"] and "public static void initialize()" not in node["public_api"]:
            raise ImplementationGraphError("IMPLEMENTATION_IR_ACTIVATION_API_REQUIRED")
        node["path"] = f"src/main/java/{package.replace('.', '/')}/{symbol}.java"
    elif node["kind"] == "resource":
        path = node["resource_path"]
        if not isinstance(path, str) or "\\" in path or ".." in PurePosixPath(path).parts:
            raise ImplementationGraphError("IMPLEMENTATION_IR_RESOURCE_PATH_INVALID")
        prefixes = (f"src/main/resources/assets/{mod_id}/", f"src/main/resources/data/{mod_id}/")
        if not path.startswith(prefixes) or not path.endswith(".json") or node["activation"] or node["public_api"]:
            raise ImplementationGraphError("IMPLEMENTATION_IR_RESOURCE_SCOPE_INVALID")
        node["path"] = path
    else:
        raise ImplementationGraphError("IMPLEMENTATION_IR_KIND_INVALID")
    return node


def ordered_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_symbol = {n["symbol"]: n for n in nodes}
    if (len({n["symbol"].casefold() for n in nodes}) != len(nodes)
            or len({n["path"].casefold() for n in nodes}) != len(nodes)):
        raise ImplementationGraphError("IMPLEMENTATION_IR_DUPLICATE_OWNER")
    if len(nodes) > MAX_NODES:
        raise ImplementationGraphError("IMPLEMENTATION_IR_NODE_LIMIT")
    if any(not set(n["depends_on"]) <= by_symbol.keys() for n in nodes):
        raise ImplementationGraphError("IMPLEMENTATION_IR_UNKNOWN_DEPENDENCY")
    result: list[dict[str, Any]] = []
    ready: set[str] = set()
    while len(result) < len(nodes):
        batch = [n for n in nodes if n["symbol"] not in ready and set(n["depends_on"]) <= ready]
        if not batch:
            raise ImplementationGraphError("IMPLEMENTATION_IR_DEPENDENCY_CYCLE")
        result.extend(batch)
        ready.update(n["symbol"] for n in batch)
    return result


def refine_node(router: Any, node: dict[str, Any], *, nodes: list[dict[str, Any]],
                package: str, mod_id: str, requirements: dict[str, str], reason: str,
                budget: int | None = None) -> list[dict[str, Any]]:
    budget = budget or admissible_tokens()
    page = _decision(router, "decompose_implementation_node", {
        "rejected_node": node, "reason": reason, "admission_tokens": budget,
        "requirements": {r: requirements[r] for r in node["requirements"]},
        "existing_contracts": [{k: n[k] for k in ("symbol", "public_api", "depends_on")} for n in nodes],
    })
    if not page["done"] or len(page["nodes"]) < 2:
        raise ImplementationGraphError("IMPLEMENTATION_IR_DECOMPOSITION_REQUIRED")
    children = [validate_node(n, package=package, mod_id=mod_id, refs=set(node["requirements"])) for n in page["nodes"]]
    original = next((n for n in children if n["symbol"] == node["symbol"]), None)
    stable = ("public_api", "kind", "resource_path", "activation")
    if original is None or any(original[k] != node[k] for k in stable):
        raise ImplementationGraphError("IMPLEMENTATION_IR_PUBLIC_CONTRACT_DRIFT")
    if original["obligations"] == node["obligations"]:
        raise ImplementationGraphError("IMPLEMENTATION_IR_DECOMPOSITION_UNCHANGED_WORK")
    external = {n["symbol"] for n in nodes if n["symbol"] != node["symbol"]}
    for child in children:
        if child is not original and (not child["symbol"].startswith(node["symbol"] + "Part") or child["activation"]):
            raise ImplementationGraphError("IMPLEMENTATION_IR_HELPER_SCOPE_INVALID")
        if (child["symbol"] in external or node_cost(child) >= node_cost(node)
                or (reason == "OUTPUT_BUDGET_EXHAUSTED" and node_cost(child) > budget)):
            raise ImplementationGraphError("IMPLEMENTATION_IR_DECOMPOSITION_NO_PROGRESS")
    if set().union(*(set(n["requirements"]) for n in children)) != set(node["requirements"]):
        raise ImplementationGraphError("IMPLEMENTATION_IR_DECOMPOSITION_LOST_REQUIREMENTS")
    result = [n for n in nodes if n["symbol"] != node["symbol"]] + children
    ordered = ordered_nodes(result)
    by_symbol = {n["symbol"]: n for n in ordered}
    reachable = set(original["depends_on"])
    frontier = list(reachable)
    while frontier:
        for dependency in by_symbol[frontier.pop()]["depends_on"]:
            if dependency not in reachable:
                reachable.add(dependency)
                frontier.append(dependency)
    if any(child is not original and child["symbol"] not in reachable for child in children):
        raise ImplementationGraphError("IMPLEMENTATION_IR_UNUSED_SPLIT_HELPER")
    return ordered


def compile_graph(router: Any, *, text: str, package: str, mod_id: str,
                  target: dict[str, Any], context: str = "") -> dict[str, Any]:
    requirements = source_requirements(text)
    nodes: list[dict[str, Any]] = []
    for page_number in range(MAX_PAGES):
        page = _decision(router, "compile_implementation_graph", {
            "requirements": requirements, "platform": target, "project_context": context,
            "package": package, "mod_id": mod_id, "admission_tokens": admissible_tokens(),
            "page": page_number + 1, "accepted_nodes": nodes,
        })
        nodes.extend(validate_node(n, package=package, mod_id=mod_id, refs=set(requirements)) for n in page["nodes"])
        if len({n["symbol"].casefold() for n in nodes}) != len(nodes):
            raise ImplementationGraphError("IMPLEMENTATION_IR_DUPLICATE_OWNER")
        if page["done"]:
            break
    else:
        raise ImplementationGraphError("IMPLEMENTATION_IR_PAGE_LIMIT")
    nodes = ordered_nodes(nodes)
    if set().union(*(set(n["requirements"]) for n in nodes)) != set(requirements):
        raise ImplementationGraphError("IMPLEMENTATION_IR_UNCOVERED_REQUIREMENTS")
    for _ in range(MAX_REFINEMENTS):
        oversized = next((n for n in nodes if node_cost(n) > admissible_tokens()), None)
        if oversized is None:
            break
        nodes = refine_node(router, oversized, nodes=nodes, package=package, mod_id=mod_id,
                            requirements=requirements, reason="preflight_output_budget")
    if any(node_cost(n) > admissible_tokens() for n in nodes):
        raise ImplementationGraphError("IMPLEMENTATION_IR_REFINEMENT_LIMIT")
    return {"schema_version": "mmm/implementation-ir-v1", "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "requirements": requirements, "nodes": nodes}
