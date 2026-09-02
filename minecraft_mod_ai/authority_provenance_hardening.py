from __future__ import annotations

"""Tighten mutation/donor authority to host-issued provenance only."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

_INSTALLED = False
_HOST_ROLES = frozenset({"system", "tool", "developer"})


def _strip_untrusted_owned_anchors(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_untrusted_owned_anchors(item)
            for key, item in value.items()
            if str(key) != "owned_anchors"
        }
    if isinstance(value, list):
        return [_strip_untrusted_owned_anchors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_untrusted_owned_anchors(item) for item in value)
    return value


def _sanitize_untrusted_message(message: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(message)
    if str(output.get("role") or "").strip().casefold() in _HOST_ROLES:
        return output
    content = output.get("content")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return output
        output["content"] = json.dumps(
            _strip_untrusted_owned_anchors(parsed),
            ensure_ascii=False,
            sort_keys=True,
        )
    elif isinstance(content, (Mapping, list, tuple)):
        output["content"] = _strip_untrusted_owned_anchors(content)
    return output


def _host_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(message)
        for message in messages
        if str(message.get("role") or "").strip().casefold() in _HOST_ROLES
    )


def _forced_tool_name(choice: Any) -> str:
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if isinstance(function, Mapping):
            return str(function.get("name") or "")
        return str(choice.get("name") or "")
    return ""


def install(loop_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    original_approved = loop_module._approved_donor_source_authority
    original_is_mutation_ready = loop_module.is_mutation_ready
    original_generate_with_tools = loop_module.generate_with_tools

    Context = loop_module.TargetMutationContext
    BaseContext = Context.__mro__[1]

    def _canonical_path(value: Any) -> str:
        return loop_module._canonical_mutation_path(value)

    def _explicit_paths(context: Any, field_name: str, *, exclude: str = "") -> tuple[str, ...]:
        output: list[str] = []
        for raw in getattr(context, field_name, ()):
            path = _canonical_path(raw)
            if path and path != exclude and path not in output:
                output.append(path)
        return tuple(output)

    def _as_context(value: Any):
        if isinstance(value, Context):
            return value
        return Context(
            target_path=getattr(value, "target_path", None),
            target_symbol=getattr(value, "target_symbol", None),
            source_body=getattr(value, "source_body", None),
            start_line=getattr(value, "start_line", None),
            end_line=getattr(value, "end_line", None),
            is_new_file=bool(getattr(value, "is_new_file", False)),
            evidence_source=getattr(value, "evidence_source", None),
            base_revision_sha=getattr(value, "base_revision_sha", None),
            writable_paths=_explicit_paths(value, "writable_paths"),
            creatable_paths=_explicit_paths(value, "creatable_paths"),
        )

    def merge(self, other):
        """Preserve file identity while carrying only *additional* PlanIR authority.

        The localized target is already authorized by ``target_path``. It must never be
        duplicated into ``writable_paths`` because that turns a target switch into leaked
        authority and changes the base context's equality/identity semantics.
        """

        if other is None:
            return self
        other_context = _as_context(other)
        left = _canonical_path(getattr(self, "target_path", None))
        right = _canonical_path(getattr(other_context, "target_path", None))
        if left and right and left != right:
            # Base contract: a newly localized file replaces the old context wholesale.
            return other_context

        merged = _as_context(BaseContext.merge(self, other_context))
        target = _canonical_path(getattr(merged, "target_path", None))
        writable = tuple(
            dict.fromkeys(
                (*_explicit_paths(self, "writable_paths", exclude=target),
                 *_explicit_paths(other_context, "writable_paths", exclude=target))
            )
        )
        creatable = tuple(
            dict.fromkeys(
                (*_explicit_paths(self, "creatable_paths", exclude=target),
                 *_explicit_paths(other_context, "creatable_paths", exclude=target))
            )
        )
        return replace(
            merged,
            writable_paths=writable,
            creatable_paths=creatable,
        )

    Context.merge = merge

    def approved_donor_source_authority(messages) -> bool:
        # Shape/schema markers never create authority. Only host-role observations are
        # eligible to satisfy the already-existing immutable donor receipt validator.
        return bool(original_approved(_host_messages(messages)))

    @wraps(original_is_mutation_ready)
    def is_mutation_ready(messages, state):
        sanitized = tuple(
            _sanitize_untrusted_message(message)
            if isinstance(message, Mapping)
            else message
            for message in messages
        )
        ready = original_is_mutation_ready(sanitized, state)
        context = getattr(state, "mutation_context", None)
        target = str(getattr(context, "target_path", "") or "").replace("\\", "/").strip()
        if context is not None and hasattr(context, "writable_paths"):
            # The current exact localization pin is mutation authority by itself. Keep
            # explicit PlanIR ownership separate so switching targets cannot leak a prior pin.
            writable = tuple(
                path
                for path in getattr(context, "writable_paths", ())
                if str(path).replace("\\", "/").strip() != target
            )
            creatable = tuple(
                path
                for path in getattr(context, "creatable_paths", ())
                if str(path).replace("\\", "/").strip() != target
            )
            state.mutation_context = replace(
                context,
                writable_paths=writable,
                creatable_paths=creatable,
            )
        return ready

    @wraps(original_generate_with_tools)
    def generate_with_tools(
        router: Any,
        *,
        config: Any,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        if role in {"coder", "coder_safe"} and not approved_donor_source_authority(
            request.messages
        ):
            filtered = loop_module._filter_donor_tool_schemas(tuple(request.tools))
            forced = _forced_tool_name(getattr(request, "tool_choice", None))
            if forced == "read_reuse_source":
                from .model_adapters import ModelConfigurationError

                raise ModelConfigurationError(
                    "DONOR_SOURCE_UNAUTHORIZED: read_reuse_source requires a host-issued "
                    "immutable donor receipt and approved materialized path."
                )
            if filtered != tuple(request.tools):
                request = replace(request, tools=filtered)
        return original_generate_with_tools(
            router,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    loop_module._approved_donor_source_authority = approved_donor_source_authority
    loop_module.is_mutation_ready = is_mutation_ready
    loop_module.generate_with_tools = generate_with_tools
    _INSTALLED = True


__all__ = ["install"]
