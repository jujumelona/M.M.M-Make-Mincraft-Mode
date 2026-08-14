"""Runtime safety invariants for bounded research and native llama.cpp grammar.

This contract owns two host-side safety boundaries that must hold independently of
model quality: hierarchical synthesis must strictly converge, and tool schemas sent
to llama.cpp must stay inside its grammar compiler's conservative JSON-schema subset.
The authoritative evidence ledger and original tool definitions remain untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any


_SYNTHESIS_PROTOCOL_V3 = "mmm/research-hierarchical-synthesis-v3"
_SYNTHESIS_NODE_BYTES = 1_400
_GRAMMAR_ERROR = "failed to initialize samplers: failed to parse grammar"
_INSTALLED = False


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _utf8_prefix(value: Any, max_bytes: int) -> str:
    text = str(value).strip()
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def _compact_synthesis_note(
    note: Mapping[str, Any],
    *,
    domain_id: str | None = None,
    max_bytes: int = _SYNTHESIS_NODE_BYTES,
) -> dict[str, Any]:
    """Project one synthesis node to a byte-bounded, schema-shaped summary.

    The durable page/claim ledgers remain lossless.  Intermediate tree nodes are only
    routing summaries, so bounding them is preferable to atomizing one node into many
    children (which can make a reduction tree grow instead of converge).
    """

    resolved_domain = _utf8_prefix(
        domain_id if domain_id is not None else note.get("domain_id", "unknown"),
        128,
    ) or "unknown"
    result: dict[str, Any] = {
        "domain_id": resolved_domain,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "sufficient": bool(note.get("sufficient", False)),
    }

    def append_if_fits(field: str, item: Any) -> bool:
        candidate = copy.deepcopy(result)
        candidate[field].append(item)
        if _json_bytes(candidate) > max_bytes:
            return False
        result[field].append(item)
        return True

    for raw_claim in list(note.get("claims", ()) or ())[:3]:
        if not isinstance(raw_claim, Mapping):
            continue
        claim = _utf8_prefix(raw_claim.get("claim", ""), 280)
        if not claim:
            continue
        refs = [
            _utf8_prefix(ref, 112)
            for ref in list(raw_claim.get("evidence_refs", ()) or ())[:2]
            if _utf8_prefix(ref, 112)
        ]
        if not append_if_fits("claims", {"claim": claim, "evidence_refs": refs}):
            break

    for raw_gap in list(note.get("gaps", ()) or ())[:2]:
        gap = _utf8_prefix(raw_gap, 240)
        if gap and not append_if_fits("gaps", gap):
            break

    for raw_query in list(note.get("next_queries", ()) or ())[:2]:
        query = _utf8_prefix(raw_query, 240)
        if query and not append_if_fits("next_queries", query):
            break

    # Host fallback evidence is already retained verbatim in evidence-ledger.jsonl.
    # Make that provenance explicit without copying the raw fragment back into the tree.
    fragment = note.get("evidence_fragment")
    if (
        not result["claims"]
        and isinstance(fragment, Mapping)
        and not result["gaps"]
    ):
        page_ref = _utf8_prefix(fragment.get("page_ref", "evidence page"), 128)
        digest = _utf8_prefix(fragment.get("content_sha256", ""), 80)
        receipt = f"Unparsed evidence retained in durable ledger: {page_ref}"
        if digest:
            receipt += f" sha256={digest}"
        append_if_fits("gaps", _utf8_prefix(receipt, 300))
        result["sufficient"] = False

    return result


def _merge_synthesis_notes(
    notes: Sequence[Mapping[str, Any]],
    *,
    domain_id: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "domain_id": domain_id,
        "claims": [],
        "gaps": [],
        "next_queries": [],
        "sufficient": bool(notes) and all(bool(note.get("sufficient")) for note in notes),
    }
    for note in notes:
        merged["claims"].extend(list(note.get("claims", ()) or ()))
        merged["gaps"].extend(list(note.get("gaps", ()) or ()))
        merged["next_queries"].extend(list(note.get("next_queries", ()) or ()))
    return _compact_synthesis_note(merged, domain_id=domain_id)


def _frontier_sha(notes: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        _compact_synthesis_note(note)
        for note in notes
    ]
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _install_synthesis_convergence(module: Any) -> None:
    if getattr(module, "_mmm_synthesis_convergence_v3", False):
        return

    module._SYNTHESIS_PROTOCOL_SCHEMA = _SYNTHESIS_PROTOCOL_V3
    synthesize_group = module._synthesize_group_with_recovery
    emit = module._emit_research_progress

    def group_synthesis_notes(notes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Always create a contracting pairwise frontier within the byte budget."""

        compact = [_compact_synthesis_note(note) for note in notes]
        groups = [compact[index : index + 2] for index in range(0, len(compact), 2)]
        # Two bounded nodes plus JSON separators are comfortably below 3600 bytes.
        for group in groups:
            if _json_bytes(group) > int(module._SYNTHESIS_INPUT_BYTES):
                raise RuntimeError("bounded synthesis pair exceeded its transport budget")
        return groups

    def terminal_gap(domain_id: str, reason: str) -> dict[str, Any]:
        return {
            "domain_id": domain_id,
            "claims": [],
            "gaps": [reason],
            "next_queries": [],
            "sufficient": False,
        }

    def hierarchical_synthesis(
        agentic_module: Any,
        router: Any,
        *,
        prompt: str,
        domain: Mapping[str, Any],
        page_notes: list[dict[str, Any]],
        domain_key: str,
        failures: list[dict[str, str]],
    ) -> dict[str, Any]:
        domain_id = str(domain.get("domain_id", "")).strip() or "unknown"
        current = page_notes or [
            terminal_gap(domain_id, "No readable evidence page note was produced.")
        ]
        current = [_compact_synthesis_note(note, domain_id=domain_id) for note in current]
        initial_count = len(current)
        max_levels = 2 * math.ceil(math.log2(max(2, initial_count))) + 4
        seen: set[str] = set()

        for level in range(max_levels):
            fingerprint = _frontier_sha(current)
            if fingerprint in seen:
                return terminal_gap(
                    domain_id,
                    "Bounded research synthesis detected a repeated semantic frontier; "
                    "full evidence remains in the durable ledger.",
                )
            seen.add(fingerprint)

            groups = group_synthesis_notes(current)
            next_level: list[dict[str, Any]] = []
            for group_index, group in enumerate(groups):
                next_level.extend(
                    synthesize_group(
                        agentic_module,
                        router,
                        prompt=prompt,
                        domain=domain,
                        group=group,
                        domain_key=domain_key,
                        failures=failures,
                        level=level,
                        group_label=str(group_index),
                    )
                )
            next_level = [
                _compact_synthesis_note(note, domain_id=domain_id)
                for note in next_level
            ]
            if len(next_level) == 1:
                return next_level[0]

            if not next_level:
                return terminal_gap(
                    domain_id,
                    "Bounded research synthesis produced an empty frontier; full evidence "
                    "remains in the durable ledger.",
                )

            if len(next_level) >= len(current):
                # Recovery may split a malformed group.  Never feed that expanded frontier
                # back to the model: deterministically collapse pairs on the host instead.
                next_level = [
                    _merge_synthesis_notes(
                        next_level[index : index + 2],
                        domain_id=domain_id,
                    )
                    for index in range(0, len(next_level), 2)
                ]
                emit(
                    "synthesis_host_contraction",
                    domain_id=domain_id,
                    level=level,
                    frontier_in=len(current),
                    frontier_out=len(next_level),
                )
                if len(next_level) == 1:
                    return next_level[0]

            if len(next_level) >= len(current):
                return terminal_gap(
                    domain_id,
                    "Bounded research synthesis could not contract the frontier; full "
                    "evidence remains in the durable ledger.",
                )

            emit(
                "synthesis_frontier",
                domain_id=domain_id,
                level=level,
                frontier_in=len(current),
                group_count=len(groups),
                frontier_out=len(next_level),
            )
            current = next_level

        return terminal_gap(
            domain_id,
            "Bounded research synthesis reached its logarithmic safety fuse; full evidence "
            "remains in the durable ledger.",
        )

    group_synthesis_notes._mmm_strict_contraction_v3 = True
    hierarchical_synthesis._mmm_strict_contraction_v3 = True
    module._group_synthesis_notes = group_synthesis_notes
    module._hierarchical_synthesis = hierarchical_synthesis
    module._mmm_synthesis_convergence_v3 = True


def _resolve_local_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    value: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            return schema
        value = value[token]
    return value if isinstance(value, Mapping) else schema


def _inferred_scalar_type(values: Sequence[Any]) -> str | None:
    if not values:
        return None
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "number"
    if all(isinstance(value, str) for value in values):
        return "string"
    return None


def _grammar_safe_schema(
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project JSON Schema to the conservative subset accepted by llama.cpp grammar."""

    root = schema if root is None else root
    resolved = _resolve_local_ref(schema, root)
    if resolved is not schema:
        return _grammar_safe_schema(resolved, root=root)

    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = resolved.get(union_key)
        if isinstance(branches, list):
            candidates = [item for item in branches if isinstance(item, Mapping)]
            non_null = [
                item for item in candidates
                if item.get("type") != "null"
            ]
            if non_null:
                return _grammar_safe_schema(non_null[0], root=root)

    raw_type = resolved.get("type")
    if isinstance(raw_type, list):
        types = [str(item) for item in raw_type if str(item) != "null"]
        raw_type = types[0] if types else "string"
    if not isinstance(raw_type, str):
        if isinstance(resolved.get("properties"), Mapping):
            raw_type = "object"
        elif "items" in resolved:
            raw_type = "array"
        else:
            enum = resolved.get("enum")
            raw_type = _inferred_scalar_type(enum) if isinstance(enum, list) else None
            raw_type = raw_type or "string"

    if raw_type == "object":
        properties_raw = resolved.get("properties")
        properties: dict[str, Any] = {}
        if isinstance(properties_raw, Mapping):
            for key, value in properties_raw.items():
                if isinstance(value, Mapping):
                    properties[str(key)] = _grammar_safe_schema(value, root=root)
                else:
                    properties[str(key)] = {"type": "string"}
        result: dict[str, Any] = {"type": "object", "properties": properties}
        required_raw = resolved.get("required")
        if isinstance(required_raw, list):
            required = [str(key) for key in required_raw if str(key) in properties]
            if required:
                result["required"] = required
        return result

    if raw_type == "array":
        items = resolved.get("items")
        return {
            "type": "array",
            "items": (
                _grammar_safe_schema(items, root=root)
                if isinstance(items, Mapping)
                else {"type": "string"}
            ),
        }

    if raw_type not in {"string", "integer", "number", "boolean", "null"}:
        raw_type = "string"
    result = {"type": raw_type}
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum and _inferred_scalar_type(enum) == raw_type:
        result["enum"] = list(enum)
    return result


def _grammar_safe_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return copy.deepcopy(dict(tool))
    safe_function: dict[str, Any] = {
        "name": str(function.get("name", "")),
    }
    description = function.get("description")
    if isinstance(description, str) and description:
        safe_function["description"] = description
    parameters = function.get("parameters")
    if isinstance(parameters, Mapping):
        projected = _grammar_safe_schema(parameters)
        if projected.get("type") != "object":
            projected = {"type": "object", "properties": {}}
    else:
        projected = {"type": "object", "properties": {}}
    safe_function["parameters"] = projected
    return {"type": "function", "function": safe_function}


def _grammar_fallback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    retry = copy.deepcopy(dict(payload))
    tools = retry.get("tools")
    if isinstance(tools, list):
        minimal: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            safe = _grammar_safe_tool(tool)
            function = safe.get("function")
            if isinstance(function, dict):
                function["parameters"] = {"type": "object", "properties": {}}
            minimal.append(safe)
        retry["tools"] = minimal
    response_format = retry.get("response_format")
    if isinstance(response_format, dict) and "schema" in response_format:
        retry["response_format"] = {"type": "json_object"}
    return retry


def _is_grammar_error(response: Any) -> bool:
    if int(getattr(response, "status_code", 0) or 0) != 400:
        return False
    try:
        body = str(response.text).lower()
    except Exception:
        return False
    return _GRAMMAR_ERROR in body


def _install_llama_grammar_safety(policy_module: Any, adapter_module: Any) -> None:
    current_payload = policy_module._server_payload
    if not getattr(current_payload, "_mmm_grammar_safe_tools_v1", False):
        @wraps(current_payload)
        def server_payload(adapter: Any, request: Any) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            tools = payload.get("tools")
            if isinstance(tools, list):
                payload = dict(payload)
                payload["tools"] = [
                    _grammar_safe_tool(tool)
                    for tool in tools
                    if isinstance(tool, Mapping)
                ]
            return payload

        server_payload._mmm_grammar_safe_tools_v1 = True
        policy_module._server_payload = server_payload

    current_post = adapter_module._post_completion
    if not getattr(current_post, "_mmm_grammar_retry_v1", False):
        @wraps(current_post)
        def post_completion(server_url: str, payload: Mapping[str, Any]) -> Any:
            response = current_post(server_url, payload)
            if not _is_grammar_error(response):
                return response
            retry_payload = _grammar_fallback_payload(payload)
            if retry_payload == payload:
                return response
            print(
                "llama server: grammar compile rejected schema; retrying once with "
                "host-validated minimal grammar",
                flush=True,
            )
            return current_post(server_url, retry_payload)

        post_completion._mmm_grammar_retry_v1 = True
        adapter_module._post_completion = post_completion


def install() -> None:
    """Install both invariants after the normal runtime bootstrap has composed owners."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import agentic_pre_design_rag, llama_server_hardware_policy
    from .model_adapters import llama_cpp_adapter

    _install_synthesis_convergence(agentic_pre_design_rag)
    _install_llama_grammar_safety(llama_server_hardware_policy, llama_cpp_adapter)
    _INSTALLED = True
