"""Runtime-token-aware, host-owned Java member prompt projection.

The authoring contract is the current method, the assigned behavior and its
verified callable APIs, NOT the entire project inventory repeated per method.
This projection is used only after the exact llama.cpp tokenizer reports that
the full prompt cannot fit. No Java API is invented or inferred from prompt text.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


_CURRENT_MEMBER_FIELDS = (
    "target_path", "class", "requirement", "side", "implementation_requirements",
    "member_role", "verified_event_callback", "handler_signature",
    "current_unit_purpose", "observed_source_name", "observed_source_task_id",
    "assigned_behavior", "assigned_behaviors",
    "class_declaration", "owner_type", "owner", "declaration_metadata",
    "host_signature", "declared_member_api", "callback_input_api",
    "host_state_work_order_scope",
    "runtime_state_observations", "verified_existing_runtime_reads", "source_owned_work",
    "prerequisite_source", "repair_current_declaration", "repair_only_this_method",
    "diagnostic", "error", "previous_implementation", "previous_candidate",
    "previous_source", "correction_obligation",
)


def _java_payloads(messages):
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            parsed = json.loads(message.get("content") or "")
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            yield parsed


def _has_member_contract(messages):
    return any(any(key in payload for key in (
        "response_source_frame", "immutable_source_frame", "immutable_signature"))
        for payload in _java_payloads(messages))


def _unknown_member_overflow(messages, remaining, required):
    return remaining < required and not _has_member_contract(messages)



def _callback_type_api(current: Mapping[str, Any]) -> tuple[list[dict], set[str]]:
    """Project only exact callable ABI for this member's Fabric callback.

    The symbol index/Gradle compiler remain authoritative. This is a read-only
    projection, NOT a source-code rewrite or guessing a class from natural
    language. Discard unrelated nested owner catalogues and method metadata
    that often prompt small coders to emit Java imports instead of statements.
    """
    event = current.get("verified_event_callback")
    records = current.get("callback_input_api")
    if not isinstance(event, Mapping) or not isinstance(records, list):
        return records if isinstance(records, list) else [], set()
    owners = {str(v) for v in (
        *event.get("parameters", ()), event.get("return_type", ""),
    ) if isinstance(v, str) and v}
    projected = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        name = str(record.get("verified_class") or "")
        bridge = record.get("verified_receiver_path")
        if name not in owners and not (
            isinstance(bridge, Mapping)
            and bridge.get("source_type") in owners
            and bridge.get("destination_type") == name
        ):
            continue
        members = []
        for original in record.get("members", ()):
            if not isinstance(original, Mapping):
                continue
            # Signature fidelity: retain the actual indexed names and types,
            # including exact overload parameter types, never abbreviated
            # pseudo-APIs. Metadata unrelated to Java calls is omitted.
            members.append({key: original[key] for key in (
                "kind", "owner", "name", "static", "return_type",
                "parameters", "generic_return_type", "generic_parameters",
            ) if key in original})
        projected.append({
            "verified_class": name,
            "members": members,
            "member_count": record.get("member_count"),
            "members_truncated": record.get("members_truncated", False),
            **({"verified_receiver_path": dict(bridge)}
               if isinstance(bridge, Mapping) else {}),
        })
    return projected, owners


def _relevant_current_api(evidence: Mapping[str, Any], owners: set[str],
                          current: Mapping[str, Any]) -> dict:
    """Retain real callable records, excluding search hits unrelated to task.

    Unrelated lookup hits such as javax.tools.StandardLocation.locationFor()
    appeared in the space-combat task merely because the prompt said
    'location'. Retaining those full classes displaced relevant event evidence
    and elicited an 8K-token stream of synthetic import directives.
    """
    if not owners:
        from .construction_file_api_evidence import scoped_body_api_records
        return {
            **{k: evidence[k] for k in (
                "target_api", "typed_input_producers", "selected_target_api",
            ) if k in evidence},
            "class_members": scoped_body_api_records(evidence),
        }
    behavior = current.get("assigned_behavior") or {}
    words = set()
    if isinstance(behavior, Mapping):
        words = {
            w.casefold() for key in ("intent", "trigger", "effects")
            for w in re.findall(r"[A-Za-z]{4,}", str(behavior.get(key) or ""))
        }
    def relevant(row):
        if not isinstance(row, Mapping):
            return False
        owner = str(row.get("owner") or row.get("verified_class") or "")
        return (owner in owners or any(
            word.casefold() in words for word in
            re.findall(r"[A-Z]?[a-z]+", owner.rsplit(".", 1)[-1])
            if len(word) >= 4
        ))
    return {
        "target_api": [dict(row) for row in evidence.get("target_api", ())
                       if relevant(row)],
        "typed_input_producers": [dict(row) for row in evidence.get(
            "typed_input_producers", ()) if isinstance(row, Mapping)
            and str(row.get("required_type") or "") in owners],
        "class_members": [
            {**dict(row), "members": [
                {key: member[key] for key in (
                    "kind", "owner", "name", "static", "return_type",
                    "parameters", "generic_return_type", "generic_parameters",
                ) if key in member}
                for member in row.get("members", ()) if isinstance(member, Mapping)
            ]}
            for row in evidence.get("class_members", ())
            if relevant(row)
        ],
    }

def _member_state_names(current: Mapping[str, Any]) -> set[str]:
    """State names in the immutable behavior contract, not the full world model."""
    behavior = current.get("assigned_behavior") or current.get("host_behavior_contract") or {}
    if not isinstance(behavior, Mapping):
        return set()
    return {str(name) for key in ("state_reads", "state_writes")
            for name in (behavior.get(key) or ()) if isinstance(name, str) and name}


def _member_state_access(records: Any, names: set[str]) -> list[dict]:
    """Retain exact overload signatures for only this behavior's state API.

    This is a prompt projection. The original full frame remains in the host
    and is used by parser, state provenance and target compilation.
    """
    if not isinstance(records, list):
        return []
    allowed = {"readState_" + name for name in names}
    allowed |= {"writeState_" + name for name in names}
    result = []
    for row in records:
        if not isinstance(row, Mapping):
            continue
        methods = [dict(method) for method in row.get("methods", ())
                   if isinstance(method, Mapping) and method.get("name") in allowed]
        if methods:
            result.append({"owner": row["owner"], "methods": methods})
    return result


def _member_prerequisites(records: Any, names: set[str],
                          state_owners: set[str]) -> Any:
    """Project large generated state owners without dropping other source ABIs.

    Keep all non-state source, plus the state owner's general transaction
    methods and the exact read/write accessors needed by the assigned behavior.
    The original source hashes/paths are still visible to the host verifier.
    """
    if not isinstance(records, list):
        return records
    prefixes = tuple(("readState_" + n, "writeState_" + n) for n in sorted(names))
    allowed = {name for pair in prefixes for name in pair}
    import re
    projected = []
    for api in records:
        if not isinstance(api, Mapping):
            projected.append(api)
            continue
        entry = dict(api)
        types = []
        for declared in api.get("types", ()):
            if not isinstance(declared, Mapping):
                types.append(declared)
                continue
            kind = dict(declared)
            if str(kind.get("owner") or "") in state_owners:
                def relevant(header):
                    found = re.search(r"\b(?:readState_|writeState_)[A-Za-z0-9_$]+\s*\(", str(header))
                    if not found:
                        return True
                    member = re.search(r"(readState_|writeState_)[A-Za-z0-9_$]+", found.group())
                    return bool(member and member.group() in allowed)
                kind["members"] = [h for h in declared.get("members", ()) if relevant(h)]
            types.append(kind)
        entry["types"] = types
        projected.append(entry)
    return projected


def _member_work_order_scope(value: Any, names: set[str]) -> Any:
    """Do not send names of hundreds of unrelated persistent variables."""
    if not isinstance(value, Mapping):
        return value
    projected = dict(value)
    for key in ("declared_readable", "declared_writable",
                "undeclared_plan_reads", "undeclared_plan_writes"):
        if isinstance(projected.get(key), list):
            projected[key] = [name for name in projected[key] if name in names]
    return projected


def scoped_java_member_messages(
    messages: Sequence[Mapping[str, Any]], frame: Mapping[str, Any],
) -> tuple[Mapping[str, str], ...]:
    """Project a bounded one-member task without changing any semantic fields.

    Project source, previous completed *bodies*, entire classpath inventories
    and duplicated source frames are not needed to author this one body. The
    host already owns them for compilation and admission. Keep verified
    signatures, callback input APIs, declared state accessors and assigned
    behavior. If no recognizable current-member task exists, do not invent one.
    """
    current: dict[str, Any] | None = None
    inspected: Mapping[str, Any] | None = None
    prerequisite: Any = None
    for parsed in _java_payloads(messages):
        if isinstance(parsed.get("inspected_target_api"), Mapping):
            inspected = parsed["inspected_target_api"]
        if "materialized_prerequisite_api" in parsed:
            prerequisite = parsed["materialized_prerequisite_api"]
        if any(key in parsed for key in ("response_source_frame", "immutable_source_frame", "immutable_signature")):
            current = parsed
    if current is None:
        return tuple(dict(m) for m in messages)
    if _uses_direct_view(current):
        return _direct_view(messages, frame, current)

    task = {key: current[key] for key in _CURRENT_MEMBER_FIELDS if key in current}
    # Only a behavior-owned member has an explicit state allowlist. A
    # supporting member without one must not silently lose its permitted API.
    bounded_state = isinstance(current.get("assigned_behavior"), Mapping)
    names = _member_state_names(current)
    # A behavior may own a source observation even when its canonical state
    # read/write lists are empty. An empty allowlist is NOT proof that the
    # upstream work order's missing-source diagnostics are irrelevant.
    if bounded_state and names:
        task["host_state_work_order_scope"] = _member_work_order_scope(
            task.get("host_state_work_order_scope"), names)
    callback_api, verified_callback_types = _callback_type_api(current)
    if verified_callback_types:
        task["callback_input_api"] = callback_api
    # The frame, full project source and completed Java bodies remain host data;
    # repeating them in LLM input duplicates many KB for every method.
    task["immutable_signature"] = {
        "kind": frame.get("kind"),
        "prefix": frame.get("prefix"),
        "suffix": frame.get("suffix"),
    }
    state_contracts = frame.get("host_state_access") or ()
    state_owners = {str(row["owner"]) for row in state_contracts
                    if isinstance(row, Mapping) and row.get("owner")}
    task["host_state_access"] = (_member_state_access(state_contracts, names)
                                 if bounded_state and names else list(state_contracts))
    if inspected:
        # Keep callable evidence rather than project-wide source inventories.
        task["inspected_target_api"] = _relevant_current_api(
            inspected, verified_callback_types, current,
        )
    if prerequisite is not None:
        # Generated content (registries, item constants, helper classes) is
        # absent from the target classpath inventory. Its actual declarations
        # must coexist with inspected Minecraft/Fabric API evidence.
        task["materialized_prerequisite_api"] = (
            _member_prerequisites(prerequisite, names, state_owners)
            if bounded_state and names else prerequisite)
    repair_api = current.get("api_evidence")
    if isinstance(repair_api, Mapping):
        # Compiler diagnostics sometimes return very large search result sets
        # dominated by unrelated JDK owners. Restrict them to the actual
        # callback type evidence. Preserve the full authoritative prerequisite
        # declarations and retain the original host index for javac.
        if verified_callback_types:
            projected = _relevant_current_api({
                "target_api": repair_api.get("selected_target_api", ()),
                "typed_input_producers": repair_api.get("typed_input_producers", ()),
                "class_members": repair_api.get("class_members", ()),
            }, verified_callback_types, current)
            task["api_evidence"] = {
                "selected_target_api": projected["target_api"],
                "typed_input_producers": projected["typed_input_producers"],
                "class_members": projected["class_members"],
            }
            if "materialized_prerequisite_api" in repair_api:
                task["api_evidence"]["materialized_prerequisite_api"] = (_member_prerequisites(
                    repair_api["materialized_prerequisite_api"], names, state_owners)
                    if bounded_state and names else repair_api["materialized_prerequisite_api"])
        else:
            task["api_evidence"] = {
                key: repair_api[key] for key in (
                    "selected_target_api", "typed_input_producers", "class_members",
                ) if key in repair_api
            }
            if "materialized_prerequisite_api" in repair_api:
                task["api_evidence"]["materialized_prerequisite_api"] = (_member_prerequisites(
                    repair_api["materialized_prerequisite_api"], names, state_owners)
                    if bounded_state and names else repair_api["materialized_prerequisite_api"])
    return (
        {"role": "system", "content": (
            "Complete exactly the current Java member. Preserve the assigned "
            "behavior and every implementation requirement. Use only the supplied "
            "verified APIs and typed state accessors. The host owns package, "
            "imports, class, signature and enclosing braces. Return exactly one "
            'JSON string field named "implementation" containing only its body. '
            "Do not invent state keys, classes or methods. NEVER emit "
            "import statements or package/class declarations in an "
            "implementation string; those are host-owned. For an event "
            "handler, use its exact verified callback parameters and return "
            "type. No invented type catalogue or speculative imports."
        )},
        {"role": "user", "content": json.dumps(
            task, ensure_ascii=False, separators=(",", ":"), default=str,
        )},
    )


def ensure_live_java_source_capacity(adapter: Any, config: Any, request: Any,
                                     original_messages: Sequence[Mapping[str, Any]],
                                     source_frame: Mapping[str, Any]):
    """Project one complete member FIRST, then count exact live llama tokens.

    State/API source stays in the host compiler. The 9B coder sees only ABI
    signatures belonging to its behavior. A 46K-token prompt cannot be
    repaired by increasing its token limit after dispatch.
    """
    if str(getattr(config, "adapter", "")).casefold() != "llama_cpp":
        return request
    counter = getattr(adapter, "input_context_accounting", None)
    if not callable(counter):
        return request

    from dataclasses import replace
    from .model_context_budget import fit_messages_to_context
    from .model_adapters.base import ModelConfigurationError

    before = counter(request)
    scoped = scoped_java_member_messages(original_messages, source_frame)
    if scoped == tuple(original_messages):
        required = max(512, min(2048, int(before.context_tokens) // 16))
        if _unknown_member_overflow(original_messages, before.remaining_tokens, required):
            raise ModelConfigurationError(
                "CONSTRUCTION_JAVA_CONTEXT_NOT_SCOPABLE: no current member "
                f"contract; input_tokens={before.input_tokens} "
                f"n_ctx={before.context_tokens}")
    else:
        request = replace(request, messages=fit_messages_to_context(
            scoped, config=config, tools=()))
    accounting = counter(request)
    required = max(512, min(2048, int(accounting.context_tokens) // 16))
    if accounting.remaining_tokens < required:
        raise ModelConfigurationError(
            "CONSTRUCTION_JAVA_VERIFIED_API_CONTEXT_TOO_LARGE: the member has "
            "more exact verified API evidence than this runtime slot can hold. "
            "Split the actual behavior/source owner, not its original meaning; "
            f"input_tokens={accounting.input_tokens} "
            f"n_ctx={accounting.context_tokens} required_output_tokens={required}")
    if scoped != tuple(original_messages):
        print("CONSTRUCTION_JAVA_EXACT_CONTEXT_SCOPED: " + json.dumps({
            "before_tokens": before.input_tokens,
            "after_tokens": accounting.input_tokens,
            "context_tokens": accounting.context_tokens,
            "retained_current_member": True,
        }), flush=True)
    return request


__all__ = ["scoped_java_member_messages", "ensure_live_java_source_capacity"]


def _uses_direct_view(current):
    return any((current.get("host_java_authoring"), current.get("host_incremental_java")))


def _direct_view(messages, frame, current):
    if current.get("host_incremental_java"):
        # This is already one operand with exact lexical scope and a paginated
        # API/work view. A legacy member projection would erase its operands.
        return tuple(dict(message) for message in messages)
    from .java_authoring_prompt import java_body_messages
    return tuple(java_body_messages(messages, frame, focus=True))
