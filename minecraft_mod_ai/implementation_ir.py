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
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from .custom_module_errors import CustomModuleGenerationError
from .model_adapters.base import NativeToolDecisionRejected

MAX_NODES = 128
MAX_PAGES = 32
MAX_REFINEMENTS = 6
MAX_PAGE_CORRECTIONS = 2
MAX_UNIT_REQUIREMENTS = 6
MAX_BATCH_REQUIREMENTS = 8
MAX_BATCH_UNITS = 3


class ImplementationGraphError(CustomModuleGenerationError):
    pass


class _InvalidPage(ImplementationGraphError):
    def __init__(self, diagnostics: list[dict[str, Any]], page: Any,
                 preserve_nodes: list[dict[str, Any]] | None = None) -> None:
        self.feedback = {"diagnostics": diagnostics, "rejected_page": page,
                         "preserve_nodes": preserve_nodes or []}
        super().__init__("IMPLEMENTATION_IR_INVALID_PAGE: " + json.dumps(diagnostics, ensure_ascii=False))


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


_STRINGS = {"type": "array", "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "pattern": r"\S"}}
_JAVA_SYMBOL = r"^[A-Z][A-Za-z0-9_]{0,95}$"
NODE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["symbol", "kind", "resource_path", "responsibility", "requirements",
                 "obligations", "public_api", "depends_on", "activation", "estimated_tokens"],
    "properties": {
        "symbol": {"type": "string", "pattern": _JAVA_SYMBOL},
        "kind": {"type": "string", "enum": ["java", "resource"]},
        "resource_path": {"type": "string", "description": "Java: exactly empty string, host derives its path. Resource: exact namespaced JSON resource path."},
        "responsibility": {"type": "string", "minLength": 1, "pattern": r"\S"},
        "requirements": {**_STRINGS, "minItems": 1}, "obligations": {**_STRINGS, "minItems": 1},
        "public_api": _STRINGS, "depends_on": _STRINGS,
        "activation": {"type": "boolean"},
        "estimated_tokens": {"type": "integer", "minimum": 1},
    },
    "allOf": [
        {"if": {"properties": {"kind": {"const": "java"}}},
         "then": {"properties": {
             "resource_path": {"const": ""},
             "public_api": {"minItems": 1, "items": {
                 "type": "string",
                 "pattern": r"^(?!\s*public\s+(?:final\s+)?(?:class|interface|enum|record)\b)[^{}]+$",
             }},
         }},
         "else": {"properties": {
             "resource_path": {"minLength": 1, "pattern": r"^src/main/resources/(assets|data)/[^/]+/(?!.*(?:\.\.|\\)).+\.json$"},
             "public_api": {"maxItems": 0}, "activation": {"const": False},
         }}},
        {"if": {"properties": {"activation": {"const": True}}},
         "then": {"properties": {"public_api": {"contains": {"const": "public static void initialize()"}}}}},
    ],
}
PAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["nodes"],
    "properties": {
        "nodes": {"type": "array", "maxItems": 4, "items": NODE_SCHEMA},
        "done": {"type": "boolean"},
        "continuation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "remaining_unit_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    },
    "allOf": [
        {
            "if": {"properties": {"nodes": {"maxItems": 0}}},
            "then": {"properties": {"done": {"const": True}}},
        }
    ],
}


def _page_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = deepcopy(PAGE_SCHEMA)
    item = schema["properties"]["nodes"]["items"]
    reqs = list(payload.get("requirements", {}))
    if reqs:
        item["properties"]["requirements"]["items"] = {"type": "string", "enum": reqs}
    if payload.get("mod_id"):
        namespace = re.escape(str(payload["mod_id"]))
        item["allOf"][0]["else"]["properties"]["resource_path"]["pattern"] = (
            rf"^src/main/resources/(assets|data)/{namespace}/(?!.*(?:\.\.|\\)).+\.json$"
        )
    return schema


def _schema_diagnostics(page: Any, schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    from jsonschema.validators import validator_for

    diagnostics = []
    for error in validator_for(schema)(schema).iter_errors(page):
        parts = list(error.absolute_path)
        node = ""
        if isinstance(page, dict) and len(parts) >= 2 and parts[0] == "nodes" and isinstance(parts[1], int):
            candidate = page["nodes"][parts[1]]
            node = str(candidate.get("symbol", "")) if isinstance(candidate, dict) else ""
        diagnostics.append({"code": "IMPLEMENTATION_IR_SCHEMA_INVALID", "node": node,
                            "field": ".".join(map(str, parts)), "message": error.message[:1000]})
    return diagnostics


def _canonicalize_public_api_declaration(value: Any) -> Any:
    """Normalize mechanically repairable Java member syntax only."""
    if not isinstance(value, str):
        return value
    text = re.sub(r"\s+", " ", value.strip()).rstrip(";").strip()
    # public_api entries are members of the host-required top-level final class.
    # A top-level class/interface/enum/record is a semantic contract error, not a
    # syntax cleanup: leave it invalid so scoped repair can replace it.
    if re.match(r"^public\s+(?:final\s+)?(?:class|interface|enum|record)\b", text):
        return text
    # Method/field bodies are mechanically removable because the declaration itself
    # remains the same member contract.
    brace = text.find("{")
    if brace >= 0:
        text = text[:brace].rstrip()
    if "}" in text:
        text = text.split("}", 1)[0].rstrip()
    return text.rstrip(";").strip()


def _canonicalize_schema_page(page: Any) -> Any:
    if not isinstance(page, dict) or not isinstance(page.get("nodes"), list):
        return page
    normalized = deepcopy(page)
    for node in normalized["nodes"]:
        if not isinstance(node, dict) or node.get("kind") != "java":
            continue
        public_api = node.get("public_api")
        if isinstance(public_api, list):
            node["public_api"] = [_canonicalize_public_api_declaration(api) for api in public_api]
    return normalized


def _schema_repair_scope(
    page: Any,
    schema: Mapping[str, Any],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return host-frozen valid siblings and the number of invalid node slots.

    Scoped repair is only safe when every diagnostic belongs to a concrete node.
    Page-shape and graph-level failures intentionally fall back to whole-page repair.
    """
    if not isinstance(page, dict) or not isinstance(page.get("nodes"), list):
        return [], 0
    nodes = page["nodes"]
    bad_indexes: set[int] = set()
    for diagnostic in diagnostics:
        match = re.match(r"^nodes\\.(\\d+)(?:\\.|$)", str(diagnostic.get("field", "")))
        if not match:
            return [], 0
        index = int(match.group(1))
        if index < 0 or index >= len(nodes):
            return [], 0
        bad_indexes.add(index)
    if not bad_indexes:
        return [], 0
    item_schema = schema["properties"]["nodes"]["items"]
    preserve = [
        deepcopy(node)
        for index, node in enumerate(nodes)
        if index not in bad_indexes
        and isinstance(node, dict)
        and not _schema_diagnostics(node, item_schema)
    ]
    return preserve, len(bad_indexes)


def _merge_scoped_schema_repair(page: Any, feedback: Mapping[str, Any]) -> Any:
    """Ignore model rewrites of frozen siblings and merge only corrected nodes."""
    preserve = list(feedback.get("preserve_nodes") or [])
    repair_count = int(feedback.get("repair_count") or 0)
    if not preserve or repair_count <= 0:
        return page
    if not isinstance(page, dict) or not isinstance(page.get("nodes"), list):
        raise _InvalidPage(
            [{"code": "IMPLEMENTATION_IR_REPAIR_NODE_COUNT", "node": "", "field": "nodes",
              "message": "Scoped repair must return the invalid node(s); valid siblings are host-owned."}],
            page,
            preserve,
        )
    frozen_symbols = {
        str(node.get("symbol"))
        for node in preserve
        if isinstance(node, dict) and node.get("symbol")
    }
    repaired = [
        deepcopy(node)
        for node in page["nodes"]
        if not isinstance(node, dict) or str(node.get("symbol", "")) not in frozen_symbols
    ]
    if len(repaired) != repair_count:
        raise _InvalidPage(
            [{"code": "IMPLEMENTATION_IR_REPAIR_NODE_COUNT", "node": "", "field": "nodes",
              "message": (
                  f"Return exactly {repair_count} corrected invalid node(s); "
                  "the host reinserts frozen valid siblings."
              )}],
            page,
            preserve,
        )
    merged = deepcopy(page)
    merged["nodes"] = [deepcopy(node) for node in preserve] + repaired
    return merged


def source_requirements(text: str) -> dict[str, str]:
    # Stable provenance, not task boundaries. A node can cite many lines across any
    # number of sections, and a line can be implemented by multiple collaborating nodes.
    return {f"R{i}": line for i, line in enumerate(text.splitlines(), 1) if line.strip()}


def is_completion_boundary_error(exc: BaseException) -> bool:
    from .llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    if isinstance(exc, OutputBudgetExhausted):
        return True
    if isinstance(exc, LlamaCompletionBoundaryError):
        return (
            exc.kind == OUTPUT_EXHAUSTED
            or (exc.max_tokens > 0 and exc.completion_tokens >= exc.max_tokens)
        )
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and is_completion_boundary_error(cause):
        return True
    msg = str(exc).lower()
    return "exhausted the bounded output allowance" in msg or "output_exhausted" in msg


def decompose_authored_units(
    text: str, max_requirements_per_unit: int = MAX_UNIT_REQUIREMENTS
) -> list[dict[str, Any]]:
    """Deterministically slice authored prose into bounded implementation units."""
    req_map = source_requirements(text)
    if not req_map:
        return []

    lines = text.splitlines(keepends=False)
    heading_pattern = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$")

    sections: list[tuple[str, list[str]]] = []
    current_title = "overview"
    current_req_ids: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        req_id = f"R{idx}"
        if req_id not in req_map:
            continue
        match = heading_pattern.match(line)
        if match:
            if current_req_ids:
                sections.append((current_title, current_req_ids))
                current_req_ids = []
            current_title = re.sub(r"[ \t]+#+[ \t]*$", "", match[1]).strip("*_` ")
        current_req_ids.append(req_id)

    if current_req_ids:
        sections.append((current_title, current_req_ids))

    units: list[dict[str, Any]] = []
    unit_counter = 0

    for title, req_ids in sections:
        if len(req_ids) <= max_requirements_per_unit:
            units.append({
                "unit_id": f"unit_{unit_counter}",
                "title": title,
                "requirements": {r: req_map[r] for r in req_ids},
            })
            unit_counter += 1
        else:
            for chunk_idx in range(0, len(req_ids), max_requirements_per_unit):
                chunk = req_ids[chunk_idx : chunk_idx + max_requirements_per_unit]
                part_num = (chunk_idx // max_requirements_per_unit) + 1
                units.append({
                    "unit_id": f"unit_{unit_counter}",
                    "title": f"{title} (part {part_num})",
                    "requirements": {r: req_map[r] for r in chunk},
                })
                unit_counter += 1

    return units


def _next_active_batch(
    pending_units: list[dict[str, Any]],
    *,
    max_batch_units: int = MAX_BATCH_UNITS,
    max_batch_reqs: int = MAX_BATCH_REQUIREMENTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pending_units:
        return [], []
    batch: list[dict[str, Any]] = []
    total_reqs = 0
    for unit in pending_units:
        unit_req_count = len(unit.get("requirements", {}))
        if batch and (
            len(batch) >= max_batch_units
            or (total_reqs + unit_req_count > max_batch_reqs)
        ):
            break
        batch.append(unit)
        total_reqs += unit_req_count
    remaining = pending_units[len(batch) :]
    return batch, remaining


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
    try:
        response = callback(
            "planner",
            [{"role": "system", "content": (
                "Compile the approved design into an implementation DAG, without changing gameplay. "
                "Markdown headings describe concerns, NOT Java classes. Group requirements across "
                "sections by state ownership and coherent responsibility. Separate state/models, "
                "persistence, services, integration, networking, UI/resources and verification where "
                "the design needs them. Cite every R identifier in the active requirements at least once. "
                "Return at most four nodes per page. If remaining units exist, you may report them in "
                "continuation.remaining_unit_ids, or set done=true when all units are complete. "
                "Each Java node is one public final class with a concrete name and NONEMPTY public_api "
                "MEMBER declaration strings (no bodies and no public class/interface/enum/record type "
                "declarations). Represent finite states as public static final fields and supporting "
                "methods on that class. Java resource_path must be exactly the empty string: "
                "the host derives it. Include constructors/fields/methods used by consumers. List actual "
                "symbol dependencies (referencing already accepted_nodes or nodes in this page), "
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
                "the rejected task. Dependencies outside the replacement must already exist. "
                "When validation_feedback is supplied and preserve_nodes is nonempty, return ONLY "
                "the corrected invalid node(s); the host owns and reinserts preserve_nodes, so never "
                "rewrite or repeat them. Fix the named fields. Never repeat the rejected page. Do not "
                "repeat already accepted_nodes on later pages."
            )}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            tool_name=name, parameters=_page_schema(payload),
            description="Host-owned implementation graph compilation; does not mutate source.",
        )
    except NativeToolDecisionRejected as exc:
        matching = [r for r in exc.rejections if r.get("original_tool") == name]
        rejected_page = None
        if len(matching) == 1:
            try:
                rejected_page = json.loads(matching[0].get("raw_arguments") or "")
            except (ValueError, TypeError):
                pass
        if rejected_page is not None:
            # Native tool validation can reject mechanically repairable member syntax.
            # Accept host normalization only when it actually changed a schema-invalid
            # page into a schema-valid page; never swallow unrelated native rejections.
            page_schema = _page_schema(payload)
            original_diagnostics = _schema_diagnostics(rejected_page, page_schema)
            normalized_page = _canonicalize_schema_page(rejected_page)
            normalized_diagnostics = _schema_diagnostics(normalized_page, page_schema)
            if (
                original_diagnostics
                and normalized_page != rejected_page
                and not normalized_diagnostics
            ):
                return normalized_page
            rejected_page = normalized_page
            diagnostics = normalized_diagnostics or original_diagnostics
        else:
            diagnostics = []
        if not diagnostics:
                return rejected_page
        else:
            diagnostics = []
        if not diagnostics:
            diagnostics = [{"code": r.get("failure_code", "TOOL_DECISION_REJECTED"),
                            "node": "", "field": "tool_call", "message": str(r.get("error", ""))}
                           for r in exc.rejections]
        failure = _InvalidPage(diagnostics, rejected_page)
        failure.feedback["native_rejections"] = list(exc.rejections)
        raise failure from exc
    return response


def _validated_page(router: Any, name: str, payload: dict[str, Any], *,
                    validator: Callable[[dict[str, Any]], Any],
                    pending: dict[str, Any] | None = None,
                    checkpoint: Callable[[dict[str, Any]], None] | None = None) -> Any:
    """Correct a rejected decision with explicit feedback, never blind replay.

    Only admission failures are correctable here. Transport, output-limit and context
    exceptions retain their own type and cannot enter this loop.
    """
    from .root_cause_trace import emit_root_cause

    request_hash = digest({"name": name, "payload": payload})
    state = deepcopy(pending) if pending else {
        "request_hash": request_hash, "attempt": 0, "seen": [], "feedback": None,
    }
    if state.get("request_hash") != request_hash:
        raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")
    if state.get("terminal"):
        raise ImplementationGraphError(state["terminal"] + ": " + json.dumps(state["feedback"], ensure_ascii=False))
    schema = _page_schema(payload)
    while state["attempt"] <= MAX_PAGE_CORRECTIONS:
        request = dict(payload)
        if state["feedback"]:
            request["validation_feedback"] = state["feedback"]
        page = None
        try:
            page = _canonicalize_schema_page(_decision(router, name, request))
            # If a prior schema rejection identified valid siblings, those siblings
            # are host-owned. The model may return only the invalid nodes (preferred)
            # or may redundantly rewrite siblings; either way, sibling text is ignored.
            page = _merge_scoped_schema_repair(page, state.get("feedback") or {})
            diagnostics = _schema_diagnostics(page, schema)
            if diagnostics:
                raise _InvalidPage(diagnostics, page)
            result = validator(page)
        except _InvalidPage as exc:
            feedback = exc.feedback
            bad_page = feedback["rejected_page"]
            schema_failure = any(d["code"] == "IMPLEMENTATION_IR_SCHEMA_INVALID" for d in feedback["diagnostics"])
            scoped_repair_failure = any(
                d["code"] == "IMPLEMENTATION_IR_REPAIR_NODE_COUNT" for d in feedback["diagnostics"]
            )
            prior_feedback = state.get("feedback") or {}

            if schema_failure:
                preserve, repair_count = _schema_repair_scope(
                    bad_page, schema, feedback["diagnostics"]
                )
                if preserve and repair_count:
                    feedback["preserve_nodes"] = preserve
                    feedback["repair_count"] = repair_count

            # Once scoped schema repair starts, valid siblings remain frozen in host
            # state through every correction. Graph-level failures intentionally do
            # not inherit provisional siblings because those siblings may participate
            # in the graph error.
            keep_protected = schema_failure or scoped_repair_failure
            if keep_protected and prior_feedback.get("preserve_nodes"):
                protected = list(prior_feedback["preserve_nodes"])
                protected_symbols = {
                    n.get("symbol") for n in protected if isinstance(n, dict)
                }
                protected.extend(
                    n for n in feedback.get("preserve_nodes", [])
                    if isinstance(n, dict) and n.get("symbol") not in protected_symbols
                )
                feedback["preserve_nodes"] = protected
                feedback["repair_count"] = int(
                    prior_feedback.get("repair_count") or feedback.get("repair_count") or 0
                )
            elif not keep_protected:
                feedback["preserve_nodes"] = []
                feedback.pop("repair_count", None)
            fingerprint = digest({"page": bad_page, "diagnostics": feedback["diagnostics"] if bad_page is None else None})
            repeated = fingerprint in state["seen"]
            state["attempt"] += 1
            state["feedback"] = feedback
            state["seen"].append(fingerprint)
            if repeated or state["attempt"] > MAX_PAGE_CORRECTIONS:
                state["terminal"] = "IMPLEMENTATION_IR_NO_PROGRESS" if repeated else "IMPLEMENTATION_IR_CORRECTION_LIMIT"
            if checkpoint:
                checkpoint(state)
            emit_root_cause("implementation_graph_page_rejected", stage="production", operation=name,
                            result="REJECTED", details={"page": payload.get("page"),
                            "attempt": state["attempt"], "response_sha256": digest(bad_page),
                            "validation_feedback": feedback, "same_response": repeated})
            if state.get("terminal"):
                raise ImplementationGraphError(state["terminal"] + ": " + json.dumps(feedback["diagnostics"], ensure_ascii=False)) from exc
            continue
        emit_root_cause("implementation_graph_page_accepted", stage="production", operation=name,
                        result="PASS", details={"page": payload.get("page"), "response": page,
                        "response_sha256": digest(page), "corrections": state["attempt"]})
        return result
    raise ImplementationGraphError("IMPLEMENTATION_IR_CORRECTION_LIMIT: " + json.dumps(state["feedback"], ensure_ascii=False))


def validate_node(raw: Any, *, package: str, mod_id: str, refs: set[str]) -> dict[str, Any]:
    # The exact model-visible schema is also the host admission contract. Avoid
    # maintaining a second set of stricter, undisclosed per-kind validation rules.
    schema = _page_schema({"requirements": sorted(refs), "mod_id": mod_id})["properties"]["nodes"]["items"]
    diagnostics = _schema_diagnostics(raw, schema)
    if diagnostics:
        raise ImplementationGraphError("IMPLEMENTATION_IR_INVALID_NODE: " + json.dumps(diagnostics, ensure_ascii=False))
    node = deepcopy(raw)
    node["estimated_tokens"] = int(node["estimated_tokens"])
    if node["kind"] == "java":
        node["public_api"] = [re.sub(r"\s+", " ", api.strip().rstrip(";").strip()) for api in node["public_api"]]
        node["path"] = f"src/main/java/{package.replace('.', '/')}/{node['symbol']}.java"
    else:
        node["path"] = node["resource_path"]
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
                budget: int | None = None, pending: dict[str, Any] | None = None,
                checkpoint: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    budget = budget or admissible_tokens()
    payload = {
        "rejected_node": node, "reason": reason, "admission_tokens": budget,
        "requirements": {r: requirements[r] for r in node["requirements"]},
        "existing_contracts": [{k: n[k] for k in ("symbol", "public_api", "depends_on")} for n in nodes],
        "package": package, "mod_id": mod_id,
    }
    def admit(page: dict[str, Any]) -> list[dict[str, Any]]:
        try:
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
        except ImplementationGraphError as exc:
            raise _InvalidPage([{"code": str(exc).split(":", 1)[0], "node": node["symbol"],
                                 "field": "nodes", "message": str(exc)}], page) from exc

    return _validated_page(router, "decompose_implementation_node", payload, validator=admit,
                           pending=pending, checkpoint=checkpoint)


def _admit_graph_page(page: dict[str, Any], *, accepted: list[dict[str, Any]],
                      package: str, mod_id: str, requirements: dict[str, str]) -> list[dict[str, Any]]:
    proposed: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    valid_raw: list[dict[str, Any]] = []
    for index, raw in enumerate(page["nodes"]):
        try:
            proposed.append(validate_node(raw, package=package, mod_id=mod_id, refs=set(requirements)))
            valid_raw.append(raw)
        except ImplementationGraphError as exc:
            diagnostics.append({"code": str(exc).split(":", 1)[0], "node": str(raw.get("symbol", "")),
                                "field": f"nodes.{index}", "message": str(exc)})
    if diagnostics:
        raise _InvalidPage(diagnostics, page, valid_raw)
    combined = accepted + proposed
    try:
        # Forward edges are legal across pages. Validate the known subgraph for cycles
        # and duplicate ownership before accepting the page, then resolve missing nodes.
        known = {n["symbol"] for n in combined}
        ordered_nodes([{**n, "depends_on": [dep for dep in n["depends_on"] if dep in known]} for n in combined])
    except ImplementationGraphError as exc:
        raise _InvalidPage([{"code": str(exc), "node": "", "field": "nodes",
                             "message": str(exc) + "; correct only this page, accepted nodes are frozen"}], page) from exc
    return combined


def compile_graph(router: Any, *, text: str, package: str, mod_id: str,
                  target: dict[str, Any], context: str = "",
                  resume: dict[str, Any] | None = None,
                  checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    from .root_cause_trace import emit_root_cause

    all_requirements = source_requirements(text)
    all_req_keys = set(all_requirements.keys())
    units = decompose_authored_units(text, max_requirements_per_unit=MAX_UNIT_REQUIREMENTS)

    request_hash = digest({"text": text, "package": package, "mod_id": mod_id, "target": target})
    state = deepcopy(resume) if resume else {
        "schema_version": "mmm/implementation-ir-draft-v1",
        "request_hash": request_hash,
        "nodes": [],
        "page": 1,
        "done": False,
        "pending": None,
        "refinements": 0,
        "unit_queue": deepcopy(units),
    }
    if state.get("request_hash") != request_hash or state.get("schema_version") != "mmm/implementation-ir-draft-v1":
        raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")

    if not state.get("unit_queue"):
        state["unit_queue"] = deepcopy(units)

    def save() -> None:
        if checkpoint:
            checkpoint(deepcopy(state))

    def save_pending(pending: dict[str, Any]) -> None:
        state["pending"] = pending
        save()

    def admit(page: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        combined = _admit_graph_page(
            page,
            accepted=state["nodes"],
            package=package,
            mod_id=mod_id,
            requirements=all_requirements,
        )
        page_covered = set().union(*(set(n["requirements"]) for n in combined))
        known = {n["symbol"] for n in combined}
        missing = any(not set(n["depends_on"]) <= known for n in combined)
        if not page["nodes"] and (not combined or set(all_req_keys) != page_covered or missing):
            raise _InvalidPage(
                [{"code": "IMPLEMENTATION_IR_UNCOVERED_REQUIREMENTS", "node": "", "field": "nodes",
                  "message": "Add nodes for remaining requirements and unresolved dependencies before finishing."}],
                page,
            )
        continuation = page.get("continuation") or {}
        rem_units = continuation.get("remaining_unit_ids")
        if rem_units:
            page_done = False
        else:
            page_done = page.get("done", True)
        return combined, page_done

    while not state["done"] and state["page"] <= MAX_PAGES:
        nodes = state["nodes"]
        covered = set().union(*(set(n["requirements"]) for n in nodes)) if nodes else set()
        known = {n["symbol"] for n in nodes}
        missing = sorted({dep for n in nodes for dep in n["depends_on"] if dep not in known})

        pending_units = [u for u in state["unit_queue"] if not set(u["requirements"].keys()) <= covered]
        if covered == all_req_keys and not missing and state.get("done"):
            save()
            break

        batch_units, rem_after_batch = _next_active_batch(
            pending_units,
            max_batch_units=MAX_BATCH_UNITS,
            max_batch_reqs=MAX_BATCH_REQUIREMENTS,
        )
        if batch_units:
            batch_reqs = {r: all_requirements[r] for u in batch_units for r in u["requirements"] if r not in covered}
        else:
            batch_reqs = {}

        remaining_req_list = [r for r in all_requirements if r not in covered]
        active_reqs = (
            batch_reqs
            if batch_reqs
            else (
                all_requirements
                if covered == all_req_keys
                else {r: all_requirements[r] for r in remaining_req_list[:MAX_BATCH_REQUIREMENTS]}
            )
        )

        payload = {
            "current_units": [u["title"] for u in batch_units],
            "unit_ids": [u["unit_id"] for u in batch_units],
            "requirements": active_reqs,
            "remaining_requirements": remaining_req_list,
            "remaining_unit_ids": [u["unit_id"] for u in rem_after_batch],
            "unresolved_dependencies": missing,
            "platform": target,
            "project_context": context,
            "package": package,
            "mod_id": mod_id,
            "admission_tokens": admissible_tokens(),
            "page": state["page"],
            "accepted_nodes": nodes,
        }

        try:
            combined, page_done = _validated_page(
                router,
                "compile_implementation_graph",
                payload,
                validator=admit,
                pending=state["pending"],
                checkpoint=save_pending,
            )
            state["nodes"] = combined
            state["page"] += 1
            state["pending"] = None
            covered = set().union(*(set(n["requirements"]) for n in combined))
            known = {n["symbol"] for n in combined}
            all_dependencies_known = all(set(n["depends_on"]) <= known for n in combined)

            state["unit_queue"] = [
                u for u in state["unit_queue"] if not set(u["requirements"].keys()) <= covered
            ]
            state["done"] = page_done and covered == all_req_keys and all_dependencies_known
            save()
        except Exception as exc:
            if is_completion_boundary_error(exc):
                emit_root_cause(
                    "implementation_graph_output_exhausted",
                    stage="production",
                    result="REDUCING_SCOPE",
                    details={
                        "page": state["page"],
                        "batch_units": [u["unit_id"] for u in batch_units],
                        "requirement_count": len(payload["requirements"]),
                    },
                )
                save()
                active_req_items = list(payload["requirements"].items())
                if len(batch_units) > 1:
                    first_u = batch_units[0]
                    other_u = [u for u in state["unit_queue"] if u["unit_id"] != first_u["unit_id"]]
                    state["unit_queue"] = [first_u] + other_u
                    save()
                    continue
                elif len(active_req_items) > 1:
                    mid = max(1, len(active_req_items) // 2)
                    target_u = batch_units[0] if batch_units else {"unit_id": "unit_split", "title": "split"}
                    sub_1 = {
                        "unit_id": f"{target_u['unit_id']}_a",
                        "title": f"{target_u['title']} (slice 1)",
                        "requirements": dict(active_req_items[:mid]),
                    }
                    sub_2 = {
                        "unit_id": f"{target_u['unit_id']}_b",
                        "title": f"{target_u['title']} (slice 2)",
                        "requirements": dict(active_req_items[mid:]),
                    }
                    other_u = [u for u in state["unit_queue"] if u["unit_id"] != target_u["unit_id"]]
                    state["unit_queue"] = [sub_1, sub_2] + other_u
                    save()
                    continue
                else:
                    raise
            raise

    if not state["done"]:
        raise ImplementationGraphError("IMPLEMENTATION_IR_PAGE_LIMIT")
    nodes = ordered_nodes(state["nodes"])

    def save_refinement(pending: dict[str, Any]) -> None:
        state["refinement_pending"] = pending
        save()

    while True:
        oversized = next((n for n in nodes if node_cost(n) > admissible_tokens()), None)
        if oversized is None:
            break
        if not state.get("refinement_pending"):
            if state["refinements"] >= MAX_REFINEMENTS:
                raise ImplementationGraphError("IMPLEMENTATION_IR_REFINEMENT_LIMIT")
            state["refinements"] += 1
            save()
        nodes = refine_node(router, oversized, nodes=nodes, package=package, mod_id=mod_id,
                            requirements=all_requirements, reason="preflight_output_budget",
                            pending=state.get("refinement_pending"), checkpoint=save_refinement)
        state["nodes"] = nodes
        state.pop("refinement_pending", None)
        save()
    return {"schema_version": "mmm/implementation-ir-v1", "source_text": text,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "requirements": all_requirements, "nodes": nodes}
