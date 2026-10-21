"""Runtime-token-aware, host-owned Java member prompt projection.

The authoring contract is the current method, the assigned behavior and its
verified callable APIs, NOT the entire project inventory repeated per method.
This projection is used only after the exact llama.cpp tokenizer reports that
the full prompt cannot fit. No Java API is invented or inferred from prompt text.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


_CURRENT_MEMBER_FIELDS = (
    "target_path", "class", "requirement", "side", "implementation_requirements",
    "member_role", "verified_event_callback", "handler_signature",
    "current_unit_purpose", "assigned_behavior", "assigned_behaviors",
    "class_declaration", "owner_type", "owner", "declaration_metadata",
    "host_signature", "declared_member_api", "callback_input_api",
    "prerequisite_source", "repair_current_declaration", "repair_only_this_method",
    "diagnostic", "error", "previous_implementation", "previous_candidate",
    "previous_source", "correction_obligation",
)



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
        if name not in owners:
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
        import re
        words = {
            w.casefold() for key in ("intent", "trigger", "effects")
            for w in re.findall(r"[A-Za-z]{4,}", str(behavior.get(key) or ""))
        }
    def relevant(row):
        if not isinstance(row, Mapping):
            return False
        owner = str(row.get("owner") or row.get("verified_class") or "")
        return (owner in owners or any(
            word in words for word in
            __import__("re").findall(r"[A-Z]?[a-z]+", owner.rsplit(".", 1)[-1])
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
    for message in messages:
        if str(message.get("role", "")) != "user":
            continue
        try:
            parsed = json.loads(str(message.get("content") or ""))
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        if isinstance(parsed.get("inspected_target_api"), Mapping):
            inspected = parsed["inspected_target_api"]
        if "materialized_prerequisite_api" in parsed:
            prerequisite = parsed["materialized_prerequisite_api"]
        if "response_source_frame" in parsed or "immutable_source_frame" in parsed:
            current = parsed
    if current is None:
        return tuple(dict(m) for m in messages)

    task = {key: current[key] for key in _CURRENT_MEMBER_FIELDS if key in current}
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
    if "host_state_access" in frame:
        task["host_state_access"] = frame["host_state_access"]
    if inspected:
        # Keep callable evidence rather than project-wide source inventories.
        task["inspected_target_api"] = _relevant_current_api(
            inspected, verified_callback_types, current,
        )
    elif prerequisite is not None:
        # No selected inspected API exists, so the prerequisite declarations
        # are the only source of authoritative types; do not discard them.
        task["materialized_prerequisite_api"] = prerequisite
    repair_api = current.get("api_evidence")
    if isinstance(repair_api, Mapping):
        # Compiler-guided repairs use a separate, already narrowed evidence
        # ABI. It must survive context compaction: without it the small model
        # might replace an invalid call with an invented one.
        task["api_evidence"] = {
            key: repair_api[key] for key in (
                "selected_target_api", "typed_input_producers", "class_members",
                "materialized_prerequisite_api",
            ) if key in repair_api
        }
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
    """Use llama.cpp's EXACT chat-template token count before Java generation.

    A bytes/characters estimate cannot prove the request will fit. The server
    endpoint is authoritative, including its actual runtime slot. Never send an
    impossible request or silently truncate the current feature's behavior.
    """
    if str(getattr(config, "adapter", "")).casefold() != "llama_cpp":
        return request
    counter = getattr(adapter, "input_context_accounting", None)
    if not callable(counter):
        return request
    accounting = counter(request)
    # Give the body writer real output capacity, not a 1-token decode.
    required = max(512, min(2048, int(accounting.context_tokens) // 16))
    if accounting.remaining_tokens >= required:
        return request

    from dataclasses import replace
    from .model_context_budget import fit_messages_to_context
    from .model_adapters.base import ModelConfigurationError

    scoped = scoped_java_member_messages(original_messages, source_frame)
    if scoped == tuple(original_messages):
        raise ModelConfigurationError(
            "CONSTRUCTION_JAVA_CONTEXT_NOT_SCOPABLE: no current member contract; "
            f"input_tokens={accounting.input_tokens} n_ctx={accounting.context_tokens}"
        )
    reduced = replace(request, messages=fit_messages_to_context(
        scoped, config=config, tools=(),
    ))
    fitted = counter(reduced)
    if fitted.remaining_tokens < required:
        raise ModelConfigurationError(
            "CONSTRUCTION_JAVA_VERIFIED_API_CONTEXT_TOO_LARGE: split/retrieve verified "
            "API evidence for this member before generation; "
            f"input_tokens={fitted.input_tokens} n_ctx={fitted.context_tokens} "
            f"required_output_tokens={required}"
        )
    print("CONSTRUCTION_JAVA_EXACT_CONTEXT_SCOPED: " + json.dumps({
        "before_tokens": accounting.input_tokens,
        "after_tokens": fitted.input_tokens,
        "context_tokens": fitted.context_tokens,
        "retained_current_member": True,
    }), flush=True)
    return reduced


__all__ = ["scoped_java_member_messages", "ensure_live_java_source_capacity"]
