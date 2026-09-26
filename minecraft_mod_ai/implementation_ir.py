"""Compile authored prose into bounded host-owned implementation units.

The host owns responsibility boundaries, paths, lifecycle, requirement coverage,
dependency topology, output admission and finite refinement. The text model is used
later for bounded source generation, not for implementation-graph planning.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from .custom_module_errors import CustomModuleGenerationError
from .implementation_lifecycle import activation_public_api

IMPLEMENTATION_IR_DRAFT_SCHEMA_VERSION = "mmm/implementation-ir-draft-v14"


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


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


_STRINGS = {"type": "array", "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "pattern": r"\S"}}
_JAVA_SYMBOL = r"^[A-Z][A-Za-z0-9_]*$"
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
        "activation": {"type": "boolean", "description": "True only for runtime integration. The host adds public static void initialize() to the coder contract; do not duplicate the hook."},
        "estimated_tokens": {"type": "integer", "minimum": 1},
    },
    "allOf": [
        {"if": {"properties": {"kind": {"const": "java"}}},
         "then": {"properties": {
             "resource_path": {"const": ""},
             "public_api": {"items": {
                 "type": "string",
                 "pattern": r"^(?!\s*(?:(?:public|protected|private|static|final|abstract)\s+)*(?:class|interface|enum|record)\b)(?!.*=\s*;?\s*$)[^{}]+$",
             }},
         }},
         "else": {"properties": {
             "resource_path": {"minLength": 1, "pattern": r"^src/main/resources/(assets|data)/[^/]+/(?!.*(?:\.\.|\\)).+\.json$"},
             "public_api": {"maxItems": 0}, "activation": {"const": False},
         }}},
        {"if": {"properties": {"kind": {"const": "java"}, "activation": {"const": False}}},
         "then": {"properties": {"public_api": {"minItems": 1}}}},
    ],
}
PAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["nodes"],
    "properties": {
        "nodes": {"type": "array", "items": NODE_SCHEMA},
    },
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
    owners = [n["symbol"] for n in payload.get("accepted_nodes", [])]
    if owners:
        # Existing owners only contribute deltas. Do not ask the model to reproduce
        # frozen identity/API/state simply to attach the next authored requirement.
        contribution = {
            "type": "object", "additionalProperties": False,
            "required": ["symbol", "requirements"],
            "properties": deepcopy(item["properties"]),
        }
        contribution["properties"]["obligations"].pop("minItems", None)
        java_owners = [n["symbol"] for n in payload["accepted_nodes"] if n["kind"] == "java"]
        contribution["allOf"] = [{
            "if": {"properties": {"symbol": {"enum": java_owners}}},
            "then": {"properties": {
                "kind": {"const": "java"},
                **deepcopy(item["allOf"][0]["then"]["properties"]),
            }},
            "else": {"properties": {
                "kind": {"const": "resource"},
                **deepcopy(item["allOf"][0]["else"]["properties"]),
            }},
        }]
        schema["properties"]["nodes"]["items"] = {
            "if": {"required": ["symbol"], "properties": {"symbol": {"enum": owners}}},
            "then": contribution, "else": item,
        }
    return schema


def _expand_owner_contributions(page: dict[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    owners = {n["symbol"]: n for n in payload.get("accepted_nodes", [])}
    expanded = deepcopy(page)
    for index, raw in enumerate(expanded["nodes"]):
        old = owners.get(raw["symbol"])
        if old is None:
            continue
        full = {**deepcopy(old), **raw}
        # Empty contributions do not erase obligations or APIs already held by the
        # host. Requirements are this page's refs; admission merges prior refs later.
        for field in ("obligations", "public_api", "depends_on"):
            if not full[field]:
                full[field] = deepcopy(old[field])
        full["activation"] = old["activation"] or full["activation"]
        expanded["nodes"][index] = full
    return expanded


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
                            "field": ".".join(map(str, parts)), "message": error.message})
    return diagnostics


def _canonicalize_public_api_declaration(value: Any) -> Any:
    """Normalize mechanically repairable Java member syntax only."""
    if not isinstance(value, str):
        return value
    text = re.sub(r"\s+", " ", value.strip()).rstrip(";").strip()
    # public_api entries are members of the host-required top-level final class.
    # A top-level class/interface/enum/record is a semantic contract error, not a
    # syntax cleanup: leave it invalid so scoped repair can replace it.
    if re.match(r"^(?:(?:public|protected|private|static|final|abstract)\s+)*(?:class|interface|enum|record)\b", text):
        return text
    # Method/field bodies are mechanically removable because the declaration itself
    # remains the same member contract.
    brace = text.find("{")
    if brace >= 0:
        text = text[:brace].rstrip()
    if "}" in text:
        text = text.split("}", 1)[0].rstrip()
    return text.rstrip(";").strip()


def _canonicalize_schema_page(page: Any, payload: Mapping[str, Any] | None = None) -> Any:
    if not isinstance(page, dict) or not isinstance(page.get("nodes"), list):
        return page
    normalized = deepcopy(page)
    # Legacy control fields are host-owned now. Ignore them before schema validation
    # so old checkpoints/tests cannot force a model-visible pagination protocol.
    normalized.pop("done", None)
    normalized.pop("continuation", None)
    owners = {n["symbol"]: n for n in (payload or {}).get("accepted_nodes", [])}
    for node in normalized["nodes"]:
        if not isinstance(node, dict):
            continue
        symbol = node.get("symbol")
        owner = owners.get(symbol, {}) if isinstance(symbol, str) else {}
        kind = node.get("kind", owner.get("kind"))
        if kind != "java":
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
        match = re.match(r"^nodes\.(\d+)(?:\.|$)", str(diagnostic.get("field", "")))
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


def _decompose_generic_units(text: str) -> list[dict[str, Any]]:
    """Explicit generic compiler mode. Authored production never selects it."""
    from .authored_ir_parser import parse_markdown_heading

    req_map = source_requirements(text)
    if not req_map:
        return []
    sections: list[tuple[str, list[str]]] = []
    current_title = "generic"
    current: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        req_id = f"R{idx}"
        heading = parse_markdown_heading(line)
        if heading is not None:
            if current:
                sections.append((current_title, current))
                current = []
            current_title = heading[1]
        current.append(req_id)
    if current:
        sections.append((current_title, current))
    return [
        {
            "unit_id": f"generic_{index}",
            "title": title,
            "requirements": {req: req_map[req] for req in refs},
            "context_requirements": {},
        }
        for index, (title, refs) in enumerate(sections)
    ]


def decompose_authored_units(text: str) -> list[dict[str, Any]]:
    """Strict canonical authored decomposition; malformed input never falls back."""
    from .authored_ir_parser import (
        AuthoredDesignSchemaError,
        decompose_canonical_authored_units,
    )

    try:
        return decompose_canonical_authored_units(text, source_requirements(text))
    except AuthoredDesignSchemaError as exc:
        raise ImplementationGraphError(str(exc)) from exc

def _next_active_unit(
    pending_units: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not pending_units:
        return None, []
    return pending_units[0], pending_units[1:]


def node_cost(node: Mapping[str, Any]) -> int:
    """Use the model-authored complete-source estimate without invented coefficients."""
    return int(node["estimated_tokens"])


def admissible_tokens(router: Any) -> int | None:
    """Return the coder runtime output budget when the router exposes it.

    Production ModelRouter uses the coder's bounded generation budget.
    Alternate routers may expose implementation_output_budget explicitly. If no
    runtime budget is knowable, skip speculative preflight sizing and let typed
    output exhaustion drive recursive decomposition.
    """
    explicit = getattr(router, "implementation_output_budget", None)
    if explicit is not None:
        value = int(explicit)
        return value if value > 0 else None
    registry = getattr(router, "registry", None)
    profile = getattr(router, "profile", None)
    if registry is None or profile is None:
        return None
    try:
        config = registry.role(profile, "coder")
        from .model_context_budget import tool_action_token_budget
        value = int(tool_action_token_budget(config))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _decision(router: Any, name: str, payload: dict[str, Any], *,
              state: dict[str, Any] | None = None,
              checkpoint: Callable[[], None] | None = None) -> dict[str, Any]:
    """Use the single host-owned lowering interface; no planner/tool fallback."""
    frontend = getattr(router, "generate_implementation_decision", None)
    if not callable(frontend):
        raise ImplementationGraphError("IMPLEMENTATION_IR_HOST_LOWERING_REQUIRED")
    return frontend(
        name, payload,
        state=state if state is not None else {},
        checkpoint=checkpoint or (lambda: None),
    )

def _repair_measure(feedback: Mapping[str, Any]) -> tuple[int, int, int]:
    """Well-founded repair measure; smaller means objectively closer to admission."""
    diagnostics = list(feedback.get("diagnostics") or [])
    schema_like = {
        "IMPLEMENTATION_IR_SCHEMA_INVALID",
        "IMPLEMENTATION_IR_REPAIR_NODE_COUNT",
    }
    schema_phase = int(any(str(item.get("code", "")) in schema_like for item in diagnostics))
    if any(item.get("code") == "IMPLEMENTATION_IR_SCHEMA_INVALID"
           and str(item.get("field", "")) in ("", "nodes") for item in diagnostics):
        schema_phase = 2
    invalid_nodes: set[str] = set()
    for item in diagnostics:
        field = str(item.get("field", ""))
        match = re.match(r"^nodes\.(\d+)(?:\.|$)", field)
        invalid_nodes.add(match.group(1) if match else field or str(item.get("node", "")))
    return schema_phase, len(invalid_nodes), len(diagnostics)


def _validated_page(router: Any, name: str, payload: dict[str, Any], *,
                    validator: Callable[[dict[str, Any]], Any],
                    pending: dict[str, Any] | None = None,
                    checkpoint: Callable[[dict[str, Any]], None] | None = None) -> Any:
    """Correct only when each rejected response makes measurable progress."""
    from .root_cause_trace import emit_root_cause

    request_hash = digest({"name": name, "payload": payload})
    state = deepcopy(pending) if pending else {
        "request_hash": request_hash,
        "attempt": 0,
        "seen": [],
        "feedback": None,
        "repair_measure": None,
    }
    if state.get("request_hash") != request_hash:
        raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")
    if state.get("terminal"):
        raise ImplementationGraphError(
            state["terminal"] + ": " + json.dumps(state["feedback"], ensure_ascii=False)
        )
    schema = _page_schema(payload)

    def save_decisions() -> None:
        if checkpoint:
            checkpoint(deepcopy(state))

    while True:
        request = dict(payload)
        if state["feedback"]:
            request["validation_feedback"] = state["feedback"]
        page = None
        try:
            page = _canonicalize_schema_page(_decision(
                router, name, request, state=state.setdefault("native_decisions", {}),
                checkpoint=save_decisions,
            ), payload)
            page = _merge_scoped_schema_repair(page, state.get("feedback") or {})
            diagnostics = _schema_diagnostics(page, schema)
            if diagnostics:
                raise _InvalidPage(diagnostics, page)
            result = validator(_expand_owner_contributions(page, payload))
        except _InvalidPage as exc:
            feedback = exc.feedback
            bad_page = feedback["rejected_page"]
            schema_failure = any(
                d["code"] == "IMPLEMENTATION_IR_SCHEMA_INVALID"
                for d in feedback["diagnostics"]
            )
            scoped_repair_failure = any(
                d["code"] == "IMPLEMENTATION_IR_REPAIR_NODE_COUNT"
                for d in feedback["diagnostics"]
            )
            prior_feedback = state.get("feedback") or {}

            if schema_failure:
                preserve, repair_count = _schema_repair_scope(
                    bad_page, schema, feedback["diagnostics"]
                )
                if preserve and repair_count:
                    feedback["preserve_nodes"] = preserve
                    feedback["repair_count"] = repair_count

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
                    prior_feedback.get("repair_count")
                    or feedback.get("repair_count")
                    or 0
                )
            elif not keep_protected:
                feedback["preserve_nodes"] = []
                feedback.pop("repair_count", None)

            fingerprint = digest({
                "page": bad_page,
                "diagnostics": feedback["diagnostics"] if bad_page is None else None,
            })
            measure = _repair_measure(feedback)
            prior_measure_raw = state.get("repair_measure")
            prior_measure = tuple(prior_measure_raw) if prior_measure_raw is not None else None
            repeated = fingerprint in state["seen"]
            improved = prior_measure is None or measure < prior_measure

            state["attempt"] += 1
            state["feedback"] = feedback
            state["seen"].append(fingerprint)
            state["repair_measure"] = list(measure)
            if repeated or not improved:
                state["terminal"] = "IMPLEMENTATION_IR_NO_PROGRESS"

            if checkpoint:
                checkpoint(state)
            emit_root_cause(
                "implementation_graph_page_rejected",
                stage="production",
                operation=name,
                result="REJECTED",
                details={
                    "page": payload.get("page"),
                    "attempt": state["attempt"],
                    "response_sha256": digest(bad_page),
                    "validation_feedback": feedback,
                    "same_response": repeated,
                    "repair_measure": list(measure),
                    "strictly_improved": improved,
                },
            )
            if state.get("terminal"):
                raise ImplementationGraphError(
                    state["terminal"] + ": "
                    + json.dumps(feedback["diagnostics"], ensure_ascii=False)
                ) from exc
            continue

        emit_root_cause(
            "implementation_graph_page_accepted",
            stage="production",
            operation=name,
            result="PASS",
            details={
                "page": payload.get("page"),
                "response": page,
                "response_sha256": digest(page),
                "corrections": state["attempt"],
            },
        )
        return result


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
        try:
            node["public_api"] = activation_public_api(node["public_api"], active=node["activation"])
        except ValueError as exc:
            raise ImplementationGraphError(str(exc)) from exc
        node["path"] = f"src/main/java/{package.replace('.', '/')}/{node['symbol']}.java"
    else:
        node["path"] = node["resource_path"]
    return node


def ordered_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_symbol = {n["symbol"]: n for n in nodes}
    if (len({n["symbol"].casefold() for n in nodes}) != len(nodes)
            or len({n["path"].casefold() for n in nodes}) != len(nodes)):
        raise ImplementationGraphError("IMPLEMENTATION_IR_DUPLICATE_OWNER")
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
    if budget is None:
        budget = admissible_tokens(router)
    payload = {
        "rejected_node": node, "reason": reason, "admission_tokens": budget,
        "requirements": {r: requirements[r] for r in node["requirements"]},
        "existing_contracts": [{k: n[k] for k in ("symbol", "public_api", "depends_on")} for n in nodes],
        "package": package, "mod_id": mod_id,
    }
    def admit(page: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            if len(page["nodes"]) < 2:
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
                if (
                    child["symbol"] in external
                    or node_cost(child) >= node_cost(node)
                    or (
                        reason == "OUTPUT_BUDGET_EXHAUSTED"
                        and budget is not None
                        and node_cost(child) > budget
                    )
                ):
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



def _stable_union(left: list[str], right: list[str]) -> list[str]:
    result = list(left)
    seen = set(left)
    for value in right:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _java_parameter_types(raw: str) -> tuple[str, ...]:
    """Best-effort Java signature normalization for host-side conflict detection."""
    parts: list[str] = []
    current: list[str] = []
    generic_depth = 0
    paren_depth = 0
    for char in raw:
        if char == "<":
            generic_depth += 1
        elif char == ">" and generic_depth:
            generic_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        if char == "," and generic_depth == 0 and paren_depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or raw.strip():
        parts.append("".join(current).strip())

    normalized: list[str] = []
    for part in parts:
        value = re.sub(r"\s+", " ", part.strip())
        value = re.sub(r"^(?:final\s+)+", "", value)
        # Drop a conventional parameter name while retaining the complete type.
        value = re.sub(r"\s+[A-Za-z_$][A-Za-z0-9_$]*$", "", value)
        value = re.sub(r"\s*([<>,\[\]])\s*", r"\1", value)
        normalized.append(value.strip())
    return tuple(normalized)


def _public_api_contract_key(declaration: str) -> tuple[Any, ...]:
    """Return the Java ownership key whose duplicate declaration cannot disagree."""
    text = re.sub(r"\s+", " ", declaration.strip().rstrip(";"))
    nested = re.search(
        r"\b(class|interface|enum|record)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b",
        text,
    )
    if nested:
        return ("type", nested.group(2))

    call = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\((.*)\)\s*(?:throws\b.*)?$", text)
    if call:
        return ("call", call.group(1), _java_parameter_types(call.group(2)))

    field = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*$", text)
    if field:
        return ("field", field.group(1))
    return ("raw", text)


def _model_node_view(node: Mapping[str, Any]) -> dict[str, Any]:
    """Compact host-owned graph state for deterministic lowering and checkpoints."""
    return {
        key: deepcopy(node[key])
        for key in (
            "symbol",
            "kind",
            "resource_path",
            "responsibility",
            "obligations",
            "estimated_tokens",
            "public_api",
            "depends_on",
            "activation",
        )
        if key in node
    }


def _merge_accepted_owner(existing: dict[str, Any], proposed: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Monotonically extend one already admitted owner without rewriting its contract."""
    for field in ("symbol", "kind", "resource_path", "path"):
        if existing.get(field) != proposed.get(field):
            raise ImplementationGraphError(
                f"IMPLEMENTATION_IR_DUPLICATE_CONTRACT_CONFLICT: {existing['symbol']} changed {field}"
            )

    existing_by_key = {
        _public_api_contract_key(api): api for api in existing["public_api"]
    }
    # Accepted API ownership is immutable. Later pages may restate the same Java
    # signature with a drifted return type/modifier/throws clause; that is not a
    # repairable decision and must never consume another model call. Preserve the
    # admitted declaration and admit only genuinely new contract keys (including
    # real overloads with different parameter types).
    novel_api: list[str] = []
    seen_keys = set(existing_by_key)
    for api in proposed["public_api"]:
        key = _public_api_contract_key(api)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        novel_api.append(api)

    merged = deepcopy(existing)
    merged["activation"] = existing["activation"] or proposed["activation"]
    merged["requirements"] = _stable_union(existing["requirements"], proposed["requirements"])
    merged["obligations"] = _stable_union(existing["obligations"], proposed["obligations"])
    merged["public_api"] = existing["public_api"] + novel_api
    try:
        merged["public_api"] = activation_public_api(merged["public_api"], active=merged["activation"])
    except ValueError as exc:
        raise ImplementationGraphError(str(exc)) from exc
    merged["depends_on"] = _stable_union(existing["depends_on"], proposed["depends_on"])
    # The first admitted responsibility remains the owner identity. Later pages add
    # precise requirement text through requirements[] and obligations[] instead of
    # rewriting that identity.
    merged["estimated_tokens"] = max(
        int(existing["estimated_tokens"]), int(proposed["estimated_tokens"])
    )

    progressed = any(
        merged[field] != existing[field]
        for field in ("requirements", "obligations", "public_api", "depends_on", "activation")
    )
    return merged, progressed


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

    # Two new nodes in the same page may never compete for one owner. Only an exact
    # symbol already admitted by an earlier page is eligible for monotonic extension.
    proposed_symbols = [n["symbol"].casefold() for n in proposed]
    proposed_paths = [n["path"].casefold() for n in proposed]
    if len(set(proposed_symbols)) != len(proposed_symbols) or len(set(proposed_paths)) != len(proposed_paths):
        raise _InvalidPage(
            [{"code": "IMPLEMENTATION_IR_DUPLICATE_OWNER", "node": "", "field": "nodes",
              "message": "Two nodes in this page claim the same symbol/path owner."}],
            page,
        )

    combined = [deepcopy(node) for node in accepted]
    owner_index = {node["symbol"]: index for index, node in enumerate(combined)}
    progressed = False
    try:
        for node in proposed:
            accepted_index = owner_index.get(node["symbol"])
            if accepted_index is None:
                # Case-insensitive symbol/path aliases are still distinct claims and
                # must remain hard failures rather than being guessed as extensions.
                if any(
                    node["symbol"].casefold() == old["symbol"].casefold()
                    or node["path"].casefold() == old["path"].casefold()
                    for old in combined
                ):
                    raise ImplementationGraphError(
                        f"IMPLEMENTATION_IR_DUPLICATE_OWNER: ambiguous owner alias {node['symbol']}"
                    )
                owner_index[node["symbol"]] = len(combined)
                combined.append(node)
                progressed = True
                continue

            merged, extended = _merge_accepted_owner(combined[accepted_index], node)
            combined[accepted_index] = merged
            progressed = progressed or extended

        if proposed and not progressed:
            raise ImplementationGraphError(
                "IMPLEMENTATION_IR_PAGE_NO_PROGRESS: page only repeated already accepted owners"
            )

        # Forward edges are legal across pages. Validate the known subgraph for cycles
        # and duplicate ownership before accepting the page, then resolve missing nodes.
        known = {n["symbol"] for n in combined}
        ordered_nodes([{**n, "depends_on": [dep for dep in n["depends_on"] if dep in known]} for n in combined])
    except ImplementationGraphError as exc:
        code = str(exc).split(":", 1)[0]
        # Pure no-progress is deterministic host evidence, not a model-correctable
        # schema/semantic mistake. Do not spend another LLM call replaying it.
        if code == "IMPLEMENTATION_IR_PAGE_NO_PROGRESS":
            raise
        raise _InvalidPage(
            [{"code": code, "node": "", "field": "nodes",
              "message": str(exc) + "; accepted nodes are host-owned and may only be extended monotonically"}],
            page,
        ) from exc
    return combined


def _compile_units(
    text: str, supplied: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    return _decompose_generic_units(text) if supplied is None else deepcopy(supplied)


def compile_authored_graph(router: Any, *, text: str, package: str, mod_id: str,
                           target: dict[str, Any], context: str = "",
                           resume: dict[str, Any] | None = None,
                           checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Compile canonical authored production. No generic fallback is reachable."""
    units = decompose_authored_units(text)
    graph = compile_graph(
        router, text=text, package=package, mod_id=mod_id, target=target,
        context=context, units=units, resume=resume, checkpoint=checkpoint,
    )
    execution_refs = set().union(*(set(unit["requirements"]) for unit in units))
    graph["requirements"] = {
        ref: value for ref, value in graph["requirements"].items()
        if ref in execution_refs
    }
    return graph

def compile_graph(router: Any, *, text: str, package: str, mod_id: str,
                  target: dict[str, Any], context: str = "",
                  units: list[dict[str, Any]] | None = None,
                  resume: dict[str, Any] | None = None,
                  checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Compile until host-observed semantic work is complete.

    There is no page/node/refinement counter. Termination is well-founded:
    successful requirement pages strictly reduce uncovered requirements; after
    coverage, dependency pages strictly reduce the unresolved dependency set; and
    output exhaustion strictly bisects the current model-visible requirement set.
    """
    from .root_cause_trace import emit_root_cause

    all_req_keys = set(all_requirements := source_requirements(text))
    units = _compile_units(text, units)
    execution_req_keys = set().union(
        *(set(unit["requirements"]) for unit in units)
    ) if units else set(all_req_keys)

    request_hash = digest({"text": text, "package": package, "mod_id": mod_id, "target": target})
    resumed = deepcopy(resume) if resume else None
    if resumed is not None and resumed.get("request_hash") != request_hash:
        raise ImplementationGraphError("IMPLEMENTATION_IR_CHECKPOINT_DRIFT")

    if resumed is not None and resumed.get("schema_version") == IMPLEMENTATION_IR_DRAFT_SCHEMA_VERSION:
        state = resumed
    else:
        state = {
            "schema_version": IMPLEMENTATION_IR_DRAFT_SCHEMA_VERSION,
            "request_hash": request_hash,
            "nodes": [],
            "page": 1,
            "done": False,
            "pending": None,
            "unit_queue": deepcopy(units),
            "dependency_scope": None,
        }
        if resumed is not None:
            emit_root_cause(
                "implementation_graph_checkpoint_invalidated",
                stage="production",
                operation="compile_implementation_graph",
                result="RECOMPILE",
                details={
                    "old_schema_version": resumed.get("schema_version"),
                    "new_schema_version": IMPLEMENTATION_IR_DRAFT_SCHEMA_VERSION,
                },
            )

    if "unit_queue" not in state:
        state["unit_queue"] = deepcopy(units)
    if "dependency_scope" not in state:
        state["dependency_scope"] = None

    def save() -> None:
        if checkpoint:
            checkpoint(deepcopy(state))

    def save_pending(pending: dict[str, Any]) -> None:
        state["pending"] = pending
        save()

    while True:
        nodes = state["nodes"]
        covered = set().union(*(set(node["requirements"]) for node in nodes)) if nodes else set()
        known = {node["symbol"] for node in nodes}
        missing = {
            dependency
            for node in nodes
            for dependency in node["depends_on"]
            if dependency not in known
        }

        if covered == execution_req_keys and not missing:
            state["done"] = True
            state["pending"] = None
            save()
            break
        state["done"] = False

        pending_units = [
            unit
            for unit in state["unit_queue"]
            if not set(unit["requirements"]) <= covered
        ]
        if covered != execution_req_keys and not pending_units:
            pending_units = [
                deepcopy(unit)
                for unit in units
                if not set(unit["requirements"]) <= covered
            ]
            state["unit_queue"] = deepcopy(pending_units)

        active_unit, _remaining_units = _next_active_unit(pending_units)
        if active_unit is not None:
            active_req_ids = [
                req_id for req_id in active_unit["requirements"] if req_id not in covered
            ]
            active_reqs = {req_id: all_requirements[req_id] for req_id in active_req_ids}
            state["dependency_scope"] = None
        else:
            if covered != execution_req_keys:
                raise ImplementationGraphError("IMPLEMENTATION_IR_WORK_QUEUE_DRIFT")
            dependency_req_ids: list[str] = []
            seen_dependency_reqs: set[str] = set()
            for accepted_node in nodes:
                if not set(accepted_node["depends_on"]).intersection(missing):
                    continue
                for req_id in accepted_node["requirements"]:
                    if req_id not in seen_dependency_reqs:
                        seen_dependency_reqs.add(req_id)
                        dependency_req_ids.append(req_id)
            if not dependency_req_ids:
                raise ImplementationGraphError("IMPLEMENTATION_IR_DEPENDENCY_CONTEXT_MISSING")
            scope = state.get("dependency_scope")
            if isinstance(scope, list) and scope:
                selected = [req_id for req_id in scope if req_id in seen_dependency_reqs]
                if not selected:
                    selected = dependency_req_ids
                    state["dependency_scope"] = None
            else:
                selected = dependency_req_ids
            active_reqs = {req_id: all_requirements[req_id] for req_id in selected}

        if not active_reqs:
            raise ImplementationGraphError("IMPLEMENTATION_IR_EMPTY_ACTIVE_WORK")

        covered_before = set(covered)
        missing_before = set(missing)
        active_req_keys = set(active_reqs)

        def admit(page: dict[str, Any], *, covered_before=covered_before,
                  missing_before=missing_before, active_req_keys=active_req_keys) -> list[dict[str, Any]]:
            combined = _admit_graph_page(
                page,
                accepted=state["nodes"],
                package=package,
                mod_id=mod_id,
                requirements=all_requirements,
            )
            new_covered = set().union(
                *(set(node["requirements"]) for node in combined)
            ) if combined else set()
            new_known = {node["symbol"] for node in combined}
            new_missing = {
                dependency
                for node in combined
                for dependency in node["depends_on"]
                if dependency not in new_known
            }

            if covered_before != execution_req_keys:
                gained = (new_covered - covered_before).intersection(active_req_keys)
                if not gained:
                    raise ImplementationGraphError(
                        "IMPLEMENTATION_IR_PAGE_NO_PROGRESS: "
                        "active requirement coverage did not increase"
                    )
            elif not new_missing < missing_before:
                raise ImplementationGraphError(
                    "IMPLEMENTATION_IR_PAGE_NO_PROGRESS: "
                    "unresolved dependencies did not strictly decrease"
                )
            return combined

        runtime_budget = admissible_tokens(router)
        payload = {
            "current_units": [active_unit["title"]] if active_unit is not None else [],
            "unit_ids": [active_unit["unit_id"]] if active_unit is not None else [],
            "requirements": active_reqs,
            "unit_context": (
                {
                    **active_unit.get("context_requirements", {}),
                    **active_unit["requirements"],
                }
                if active_unit is not None
                else active_reqs
            ),
            "planned_units": [unit["title"] for unit in units],
            "unresolved_dependencies": sorted(missing),
            "platform": target,
            "project_context": context,
            "package": package,
            "mod_id": mod_id,
            "page": state["page"],
            "accepted_nodes": [_model_node_view(node) for node in nodes],
        }
        if runtime_budget is not None:
            payload["admission_tokens"] = runtime_budget

        try:
            combined = _validated_page(
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
            state["dependency_scope"] = None

            new_covered = set().union(
                *(set(node["requirements"]) for node in combined)
            ) if combined else set()
            state["unit_queue"] = [
                unit
                for unit in state["unit_queue"]
                if not set(unit["requirements"]) <= new_covered
            ]
            new_known = {node["symbol"] for node in combined}
            new_missing = {
                dependency
                for node in combined
                for dependency in node["depends_on"]
                if dependency not in new_known
            }
            state["done"] = new_covered == execution_req_keys and not new_missing
            save()
        except Exception as exc:
            if not is_completion_boundary_error(exc):
                raise

            emit_root_cause(
                "implementation_graph_output_exhausted",
                stage="production",
                result="REDUCING_SCOPE",
                details={
                    "page": state["page"],
                    "unit_id": active_unit["unit_id"] if active_unit is not None else None,
                    "requirement_count": len(active_reqs),
                },
            )
            state["pending"] = None
            active_items = list(active_reqs.items())
            if len(active_items) <= 1:
                save()
                raise

            midpoint = len(active_items) // 2
            left_items = active_items[:midpoint]
            right_items = active_items[midpoint:]
            if active_unit is not None:
                left = {
                    "unit_id": f"{active_unit['unit_id']}.left",
                    "title": f"{active_unit['title']} (left)",
                    "requirements": dict(left_items),
                }
                right = {
                    "unit_id": f"{active_unit['unit_id']}.right",
                    "title": f"{active_unit['title']} (right)",
                    "requirements": dict(right_items),
                }
                rebuilt: list[dict[str, Any]] = []
                replaced = False
                for unit in state["unit_queue"]:
                    if not replaced and unit["unit_id"] == active_unit["unit_id"]:
                        rebuilt.extend((left, right))
                        replaced = True
                    else:
                        rebuilt.append(unit)
                if not replaced:
                    rebuilt = [left, right, *state["unit_queue"]]
                state["unit_queue"] = rebuilt
            else:
                state["dependency_scope"] = [req_id for req_id, _ in left_items]
            save()
            continue

    nodes = ordered_nodes(state["nodes"])
    runtime_budget = admissible_tokens(router)

    def save_refinement(pending: dict[str, Any]) -> None:
        state["refinement_pending"] = pending
        save()

    if runtime_budget is not None:
        while True:
            oversized = next(
                (node for node in nodes if node_cost(node) > runtime_budget),
                None,
            )
            if oversized is None:
                break
            nodes = refine_node(
                router,
                oversized,
                nodes=nodes,
                package=package,
                mod_id=mod_id,
                requirements=all_requirements,
                reason="preflight_output_budget",
                budget=runtime_budget,
                pending=state.get("refinement_pending"),
                checkpoint=save_refinement,
            )
            state["nodes"] = nodes
            state.pop("refinement_pending", None)
            save()

    return {
        "schema_version": "mmm/implementation-ir-v1",
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "requirements": all_requirements,
        "nodes": nodes,
    }
