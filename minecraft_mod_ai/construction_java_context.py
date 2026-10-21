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
        from .construction_file_api_evidence import scoped_body_api_records
        task["inspected_target_api"] = {
            key: inspected[key] for key in (
                "target_api", "typed_input_producers", "selected_target_api",
            ) if key in inspected
        }
        task["inspected_target_api"]["class_members"] = scoped_body_api_records(inspected)
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
            "Do not invent state keys, classes or methods."
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
