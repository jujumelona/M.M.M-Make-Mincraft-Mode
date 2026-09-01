"""Runtime safety invariants for bounded research and native llama.cpp tool grammar.

This contract owns three host-side safety boundaries that must hold independently of
model quality: bounded research has exactly one structured-output repair owner and no
model-authored evidence cursor, hierarchical synthesis strictly converges, and tool
schemas sent to llama.cpp stay inside its grammar compiler's conservative JSON-schema
subset. Detailed response schemas are validated and repaired by ``structured_output``
on the host; transport failures are never retried here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from typing import Any

_SYNTHESIS_PROTOCOL_V4 = "mmm/research-hierarchical-synthesis-v4"
_SYNTHESIS_NODE_BYTES = 1_400
_MIN_SYNTHESIS_INPUT_BYTES = 10_240
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

    The durable page/evidence ledgers remain lossless. Intermediate tree nodes are only
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
    canonical = [_compact_synthesis_note(note) for note in notes]
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _synthesis_worker_count(router: Any, width: int) -> int:
    """Reuse the one native llama capacity authority for independent synthesis groups."""

    if router is None or width <= 1:
        return 1
    try:
        from .central_intelligence_amplifier import _research_domain_worker_count

        return max(1, min(int(width), int(_research_domain_worker_count(router, width))))
    except Exception:
        return 1


def _latest_json_payload(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if str(message.get("role", "")) != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _bound_research_schema(
    module: Any,
    response_schema: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Bind host-owned semantic invariants into the host-only detailed schema."""

    schema = copy.deepcopy(dict(response_schema))
    payload = _latest_json_payload(messages)
    domain = payload.get("domain")
    domain_id = ""
    if isinstance(domain, Mapping):
        domain_id = str(domain.get("domain_id", "")).strip()

    properties = schema.get("properties")
    research_note = properties.get("research_note") if isinstance(properties, Mapping) else None
    if isinstance(research_note, dict):
        note_properties = research_note.get("properties")
        if isinstance(note_properties, dict) and domain_id:
            domain_schema = note_properties.get("domain_id")
            if isinstance(domain_schema, dict):
                domain_schema["const"] = domain_id
        # A zero-claim note may be a valid unresolved result, but it must never claim
        # sufficiency and then get persisted as a successful research conclusion.
        all_of = research_note.setdefault("allOf", [])
        all_of.append(
            {
                "if": {
                    "properties": {"sufficient": {"const": True}},
                    "required": ["sufficient"],
                },
                "then": {"properties": {"claims": {"minItems": 1}}},
            }
        )

    evidence_page = payload.get("evidence_page")
    if isinstance(evidence_page, Mapping) and isinstance(properties, dict):
        # Defensive projection for callers that still pass the retired v2 continuation
        # shape. The model extracts semantics; the host already knows the exact supplied
        # span, page length, and digest and therefore owns its completion receipt.
        properties.pop("continuation", None)
        schema.pop("required", None)

    return schema, domain_id


def _aligned_research_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    response_schema: Mapping[str, Any],
    domain_id: str,
) -> list[dict[str, Any]]:
    """Make the first generation prompt describe the same envelope the host validates."""

    result = [dict(message) for message in messages]
    required = response_schema.get("required")
    required_keys = [str(item) for item in required] if isinstance(required, list) else []
    properties = response_schema.get("properties")
    research_only = (
        isinstance(properties, Mapping)
        and "research_note" in properties
        and set(properties) <= {"research_note"}
    )
    if required_keys == ["research_note"] or research_only:
        shape = 'exactly one top-level JSON object with exactly the key "research_note"'
    elif "research_note" in required_keys and "continuation" in required_keys:
        shape = (
            'exactly one top-level JSON object with exactly the keys "research_note" '
            'and "continuation"'
        )
    else:
        shape = "exactly one top-level JSON object matching the requested response contract"
    directive = (
        f" Return {shape}; do not return a bare research_note body."
        + (f' research_note.domain_id must be exactly "{domain_id}".' if domain_id else "")
        + " If no evidence-backed design claim can be extracted, set sufficient=false and "
        "record the reason in gaps; never set sufficient=true with an empty claims array. "
        "Do not emit continuation, next_offset, tail_sha256, or page-completion fields."
    )
    for message in result:
        if str(message.get("role", "")) == "system" and isinstance(message.get("content"), str):
            message["content"] = str(message["content"]).rstrip() + directive
            break
    else:
        result.insert(0, {"role": "system", "content": directive.strip()})
    return result


def _install_bounded_research_efficiency(module: Any) -> None:
    """Give host schema repair sole ownership of bounded research correction."""

    if getattr(module, "_mmm_single_structured_repair_owner_v1", False):
        return

    module._SYNTHESIS_INPUT_BYTES = max(
        int(getattr(module, "_SYNTHESIS_INPUT_BYTES", 0)),
        _MIN_SYNTHESIS_INPUT_BYTES,
    )

    def generate_bounded(
        agentic_module: Any,
        router: Any,
        *,
        messages: list[dict[str, str]],
        response_schema: Mapping[str, Any],
        parser: Any,
        progress_label: str,
    ) -> Any:
        bound_schema, domain_id = _bound_research_schema(
            module,
            response_schema,
            messages,
        )
        aligned_messages = _aligned_research_messages(
            messages,
            response_schema=bound_schema,
            domain_id=domain_id,
        )
        module._emit_research_progress("model_attempt", label=progress_label, attempt=1)
        emit_failure = getattr(module, "_emit_bounded_failure", None)
        normalize_json = getattr(module, "_normalize_bounded_json_text", None)
        hash_text = getattr(module, "_sha256_text", None)

        def report_failure(event: str, *, error: Exception, raw_output: str) -> None:
            if callable(emit_failure):
                emit_failure(
                    event,
                    progress_label=progress_label,
                    raw_output=raw_output,
                    error=error,
                )

        raw = ""
        try:
            raw = router.generate_text(
                "planner",
                aligned_messages,
                response_format="json",
                response_schema=bound_schema,
                tool_stage="research",
                enable_tools=False,
            )
        except Exception as exc:
            report_failure("bounded_model_failure", error=exc, raw_output=raw)
            raise

        try:
            return parser(raw)
        except Exception as first_error:
            report_failure("bounded_parse_failure", error=first_error, raw_output=raw)
            try:
                normalized = normalize_json(raw) if callable(normalize_json) else raw
                parsed = parser(normalized)
            except Exception as normalized_error:
                report_failure(
                    "bounded_host_normalization_failure",
                    error=normalized_error,
                    raw_output=raw,
                )
                raise module._BoundedResearchOutputError(
                    "bounded structured output failed after host repair: "
                    f"{type(normalized_error).__name__}: {normalized_error}"
                ) from normalized_error
            module._emit_research_progress(
                "bounded_host_normalized",
                label=progress_label,
                attempt=1,
                raw_output_sha256=hash_text(raw) if callable(hash_text) else "",
            )
            return parsed

    generate_bounded._mmm_single_structured_repair_owner_v1 = True
    module._generate_bounded = generate_bounded
    module._mmm_single_structured_repair_owner_v1 = True


def _install_synthesis_convergence(module: Any) -> None:
    if getattr(module, "_mmm_synthesis_convergence_v4", False):
        return

    module._SYNTHESIS_PROTOCOL_SCHEMA = _SYNTHESIS_PROTOCOL_V4
    module._SYNTHESIS_INPUT_BYTES = max(
        int(getattr(module, "_SYNTHESIS_INPUT_BYTES", 0)),
        _MIN_SYNTHESIS_INPUT_BYTES,
    )
    synthesize_group = module._synthesize_group_with_recovery
    emit = module._emit_research_progress
    original_group = getattr(module, "_group_synthesis_notes", None)

    def group_synthesis_notes(notes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Preserve raw evidence leaves, then pairwise-contract synthesized notes."""

        has_raw_evidence = any(
            isinstance(note.get("evidence_fragment"), Mapping) for note in notes
        )
        if has_raw_evidence and callable(original_group):
            # Raw page fragments are the lossless first model input. Raising the bounded
            # synthesis transport budget lets the existing four-item packer actually pack
            # several 1.8-KB pages instead of degenerating into one model call per page.
            groups = original_group(notes)
        else:
            compact = [_compact_synthesis_note(note) for note in notes]
            groups = [compact[index : index + 2] for index in range(0, len(compact), 2)]
        for group in groups:
            if _json_bytes(group) > int(module._SYNTHESIS_INPUT_BYTES):
                raise RuntimeError("bounded synthesis group exceeded its transport budget")
        return groups

    def terminal_gap(domain_id: str, reason: str) -> dict[str, Any]:
        return {
            "domain_id": domain_id,
            "claims": [],
            "gaps": [reason],
            "next_queries": [],
            "sufficient": False,
        }

    def final_result(
        note: Mapping[str, Any],
        *,
        domain_id: str,
        failures: list[dict[str, str]],
    ) -> dict[str, Any]:
        result = _compact_synthesis_note(note, domain_id=domain_id)
        if result["claims"]:
            return result
        reason = (
            "No evidence-backed design-relevant claim survived bounded synthesis; full "
            "evidence remains in the durable ledger."
        )
        result["sufficient"] = False
        if not result["gaps"]:
            result["gaps"] = [reason]
        marker = {"unit": "synthesis:final", "error": reason}
        if marker not in failures:
            failures.append(marker)
        return result

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
        current = (
            [dict(note) for note in page_notes]
            if page_notes
            else [terminal_gap(domain_id, "No readable evidence page note was produced.")]
        )
        initial_count = len(current)
        max_levels = 2 * math.ceil(math.log2(max(2, initial_count))) + 4
        seen: set[str] = set()

        for level in range(max_levels):
            fingerprint = _frontier_sha(current)
            if fingerprint in seen:
                return final_result(
                    terminal_gap(
                        domain_id,
                        "Bounded research synthesis detected a repeated semantic frontier; "
                        "full evidence remains in the durable ledger.",
                    ),
                    domain_id=domain_id,
                    failures=failures,
                )
            seen.add(fingerprint)

            groups = group_synthesis_notes(current)

            def run_group(
                group_index: int,
                group: list[dict[str, Any]],
            ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
                local_failures: list[dict[str, str]] = []
                notes = list(
                    synthesize_group(
                        agentic_module,
                        router,
                        prompt=prompt,
                        domain=domain,
                        group=group,
                        domain_key=domain_key,
                        failures=local_failures,
                        level=level,
                        group_label=str(group_index),
                    )
                )
                return notes, local_failures

            workers = _synthesis_worker_count(router, len(groups))
            group_results: dict[
                int,
                tuple[list[dict[str, Any]], list[dict[str, str]]],
            ] = {}
            if workers <= 1:
                for group_index, group in enumerate(groups):
                    group_results[group_index] = run_group(group_index, group)
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="mmm_research_synthesis",
                ) as pool:
                    futures = []
                    for group_index, group in enumerate(groups):
                        context = copy_context()
                        future = pool.submit(
                            context.run,
                            run_group,
                            group_index,
                            group,
                        )
                        futures.append((group_index, future))
                    for group_index, future in futures:
                        group_results[group_index] = future.result()

            next_level: list[dict[str, Any]] = []
            for group_index in range(len(groups)):
                group_notes, group_failures = group_results[group_index]
                next_level.extend(group_notes)
                failures.extend(group_failures)

            next_level = [
                _compact_synthesis_note(note, domain_id=domain_id)
                for note in next_level
            ]
            if len(next_level) == 1:
                return final_result(
                    next_level[0],
                    domain_id=domain_id,
                    failures=failures,
                )

            if not next_level:
                return final_result(
                    terminal_gap(
                        domain_id,
                        "Bounded research synthesis produced an empty frontier; full evidence "
                        "remains in the durable ledger.",
                    ),
                    domain_id=domain_id,
                    failures=failures,
                )

            if len(next_level) >= len(current):
                # Once every raw leaf in this frontier has reached the model, a valid but
                # non-contracting set of summaries can be collapsed deterministically on
                # the host. This prevents an infinite model loop without dropping evidence
                # before its first synthesis pass.
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
                    return final_result(
                        next_level[0],
                        domain_id=domain_id,
                        failures=failures,
                    )

            if len(next_level) >= len(current):
                return final_result(
                    terminal_gap(
                        domain_id,
                        "Bounded research synthesis could not contract the frontier; full "
                        "evidence remains in the durable ledger.",
                    ),
                    domain_id=domain_id,
                    failures=failures,
                )

            emit(
                "synthesis_frontier",
                domain_id=domain_id,
                level=level,
                frontier_in=len(current),
                group_count=len(groups),
                frontier_out=len(next_level),
                parallel_workers=workers,
            )
            current = next_level

        return final_result(
            terminal_gap(
                domain_id,
                "Bounded research synthesis reached its logarithmic safety fuse; full "
                "evidence remains in the durable ledger.",
            ),
            domain_id=domain_id,
            failures=failures,
        )

    group_synthesis_notes._mmm_strict_contraction_v4 = True
    hierarchical_synthesis._mmm_strict_contraction_v4 = True
    module._group_synthesis_notes = group_synthesis_notes
    module._hierarchical_synthesis = hierarchical_synthesis
    module._mmm_synthesis_convergence_v4 = True


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
    """Project tool JSON Schema to the conservative subset accepted by llama.cpp."""

    root = schema if root is None else root
    resolved = _resolve_local_ref(schema, root)
    if resolved is not schema:
        return _grammar_safe_schema(resolved, root=root)

    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = resolved.get(union_key)
        if isinstance(branches, list):
            candidates = [item for item in branches if isinstance(item, Mapping)]
            non_null = [item for item in candidates if item.get("type") != "null"]
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
                properties[str(key)] = (
                    _grammar_safe_schema(value, root=root)
                    if isinstance(value, Mapping)
                    else {"type": "string"}
                )
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
    safe_function: dict[str, Any] = {"name": str(function.get("name", ""))}
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


def _install_llama_tool_schema_projection(policy_module: Any) -> None:
    """Make the first tool request grammar-safe without any response retry."""

    current_payload = policy_module._server_payload
    if getattr(current_payload, "_mmm_grammar_safe_tools_v1", False):
        return

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


def install() -> None:
    """Install one repair owner, convergence, and first-request tool projection."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import agentic_pre_design_rag, llama_server_hardware_policy

    _install_bounded_research_efficiency(agentic_pre_design_rag)
    _install_synthesis_convergence(agentic_pre_design_rag)
    _install_llama_tool_schema_projection(llama_server_hardware_policy)
    _INSTALLED = True


__all__ = ["install"]
