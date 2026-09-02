from __future__ import annotations

"""Bind host-issued PlanIR mutation ownership to exact writable files and isolate donor reads."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

_MARKER = "_mmm_planir_mutation_authority_v1"
_DONOR_TOOL = "read_reuse_source"
_HOST_ROLES = frozenset({"system", "tool", "developer"})
_HEX_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")
_HEX_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_DONOR_ROOT_FRAGMENT = ".minecraft_ai/reuse/donors/"


def _structured_payload(content: Any) -> Any | None:
    if isinstance(content, (Mapping, list, tuple)):
        return content
    if isinstance(content, str):
        raw = content.strip()
        if not raw.startswith(("{", "[")):
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return None


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
    role = str(output.get("role") or "").strip().casefold()
    if role in _HOST_ROLES:
        return output
    content = output.get("content")
    payload = _structured_payload(content)
    if payload is None:
        return output
    stripped = _strip_untrusted_owned_anchors(payload)
    output["content"] = (
        json.dumps(stripped, ensure_ascii=False, sort_keys=True)
        if isinstance(content, str)
        else stripped
    )
    return output


def _canonical_owned_locator(locator: Any, loop_module: Any) -> str:
    raw = str(locator or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = raw.split("#", 1)[0].strip()
    while path.startswith("./"):
        path = path[2:]
    if not path or ":" in path or path.startswith("/"):
        return ""
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    if path.startswith(".minecraft_ai/reuse/donors/"):
        return ""
    canonical = loop_module._canonical_mutation_path(path)
    if not canonical or not loop_module._is_workspace_file_path(canonical):
        return ""
    return canonical


def _active_task_ids(payload: Mapping[str, Any]) -> frozenset[str]:
    active: set[str] = set()
    direct = str(payload.get("task_id") or payload.get("task_ref") or "").strip()
    if direct:
        active.add(direct)
    module = payload.get("module")
    if isinstance(module, Mapping):
        module_id = str(module.get("module_id") or "").strip()
        if module_id:
            active.add(module_id)
        config = module.get("config")
        if isinstance(config, Mapping):
            evidence_task = config.get("evidence_task")
            if isinstance(evidence_task, Mapping):
                task_id = str(evidence_task.get("task_id") or "").strip()
                if task_id:
                    active.add(task_id)
    return frozenset(active)


def _owned_anchor_sets(payload: Any, loop_module: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        return (), ()
    active_ids = _active_task_ids(payload)
    writable: list[str] = []
    creatable: list[str] = []

    def add_unique(target: list[str], value: str) -> None:
        if value and value not in target:
            target.append(value)

    def walk(node: Any, *, parent_key: str = "", is_root: bool = False) -> None:
        if isinstance(node, Mapping):
            task_id = str(node.get("task_id") or "").strip()
            anchors = node.get("owned_anchors")
            is_task = bool(task_id) and isinstance(anchors, (list, tuple))
            is_active = is_root or parent_key == "evidence_task" or task_id in active_ids
            if is_task and is_active:
                for anchor in anchors:
                    if not isinstance(anchor, Mapping):
                        continue
                    path = _canonical_owned_locator(anchor.get("locator"), loop_module)
                    if not path:
                        continue
                    add_unique(writable, path)
                    if str(anchor.get("status") or "").strip().casefold() == "host_reserved":
                        add_unique(creatable, path)
            for key, value in node.items():
                walk(value, parent_key=str(key), is_root=False)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, parent_key=parent_key, is_root=False)

    walk(payload, is_root=True)
    return tuple(writable), tuple(creatable)


def _collect_message_owned_anchors(
    messages: Sequence[Mapping[str, Any]], loop_module: Any
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    writable: list[str] = []
    creatable: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip().casefold()
        if role not in _HOST_ROLES:
            continue
        payload = _structured_payload(message.get("content"))
        if payload is None:
            continue
        paths, creates = _owned_anchor_sets(payload, loop_module)
        for path in paths:
            if path not in writable:
                writable.append(path)
        for path in creates:
            if path not in creatable:
                creatable.append(path)
    return tuple(writable), tuple(creatable)


def _walk_scalar_fields(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_scalar_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_scalar_fields(item)


def _approved_donor_authority(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Require a host-role immutable donor receipt before exposing donor reads."""
    for message in messages:
        role = str(message.get("role") or "").strip().casefold()
        if role not in _HOST_ROLES:
            continue
        payload = _structured_payload(message.get("content"))
        if payload is None:
            continue
        donor_path = False
        commit = False
        digest = False
        license_id = False
        for key, item in _walk_scalar_fields(payload):
            key_cf = key.casefold()
            if isinstance(item, str):
                text = item.replace("\\", "/").strip()
                if _DONOR_ROOT_FRAGMENT in text and ".." not in PurePosixPath(text).parts:
                    donor_path = True
                if "commit" in key_cf and _HEX_COMMIT.fullmatch(text):
                    commit = True
                if ("sha256" in key_cf or "hash" in key_cf) and _HEX_SHA256.fullmatch(text):
                    digest = True
                if key_cf in {"license", "license_id"} and text:
                    license_id = True
        if donor_path and commit and digest and license_id:
            return True
    return False


def _tool_name(schema: Any) -> str:
    if isinstance(schema, Mapping):
        fn = schema.get("function")
        return str(fn.get("name") or "").strip() if isinstance(fn, Mapping) else ""
    return str(getattr(schema, "name", "") or "").strip()


def _forced_tool_name(choice: Any) -> str:
    if not isinstance(choice, Mapping):
        return ""
    fn = choice.get("function")
    if isinstance(fn, Mapping):
        return str(fn.get("name") or "").strip()
    return str(choice.get("name") or "").strip()


def _filter_donor_tool_schemas(schemas: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(schema for schema in schemas if _tool_name(schema) != _DONOR_TOOL)


def install(loop_module: Any) -> None:
    if getattr(loop_module, _MARKER, False):
        return

    BaseContext = loop_module.TargetMutationContext
    original_extract = loop_module._extract_mutation_context_from_payload
    original_is_ready = loop_module.is_mutation_ready
    original_context_dict = loop_module._mutation_context_dict
    original_generate = loop_module.generate_with_tools

    def _union_paths(*groups: Sequence[str]) -> tuple[str, ...]:
        out: list[str] = []
        for group in groups:
            for raw in group:
                path = loop_module._canonical_mutation_path(raw)
                if path and path not in out:
                    out.append(path)
        return tuple(out)

    @dataclass(frozen=True)
    class AuthorizedTargetMutationContext(BaseContext):
        # Only *additional* host-issued PlanIR authority lives here. The currently
        # localized target_path is already an exact mutation pin and is intentionally
        # not duplicated into writable_paths.
        writable_paths: tuple[str, ...] = ()
        creatable_paths: tuple[str, ...] = ()

        def merge(self, other):
            if other is None:
                return self
            other_ctx = _authorize_context(other)
            left = loop_module._canonical_mutation_path(self.target_path)
            right = loop_module._canonical_mutation_path(other_ctx.target_path)
            if left and right and left != right:
                # A newly localized file replaces the old localization context. Carrying
                # the previous pin into writable_paths would leak mutation authority.
                return other_ctx
            merged = BaseContext.merge(self, other_ctx)
            merged_ctx = _authorize_context(merged)
            return replace(
                merged_ctx,
                writable_paths=_union_paths(self.writable_paths, other_ctx.writable_paths),
                creatable_paths=_union_paths(self.creatable_paths, other_ctx.creatable_paths),
            )

    def _authorize_context(
        context: Any,
        *,
        writable: Sequence[str] = (),
        creatable: Sequence[str] = (),
    ):
        if isinstance(context, AuthorizedTargetMutationContext):
            current = context
        else:
            current = AuthorizedTargetMutationContext(
                target_path=context.target_path,
                target_symbol=context.target_symbol,
                source_body=context.source_body,
                start_line=context.start_line,
                end_line=context.end_line,
                is_new_file=context.is_new_file,
                evidence_source=context.evidence_source,
                base_revision_sha=context.base_revision_sha,
            )
        target = loop_module._canonical_mutation_path(current.target_path)
        writable_paths = _union_paths(
            tuple(path for path in current.writable_paths if path != target),
            tuple(path for path in writable if loop_module._canonical_mutation_path(path) != target),
        )
        creatable_paths = _union_paths(current.creatable_paths, creatable)
        return replace(current, writable_paths=writable_paths, creatable_paths=creatable_paths)

    @wraps(original_extract)
    def extract(payload):
        context = original_extract(payload)
        if context is None:
            return None
        writable, creatable = _owned_anchor_sets(payload, loop_module)
        return _authorize_context(context, writable=writable, creatable=creatable)

    @wraps(original_is_ready)
    def is_mutation_ready(messages, state):
        # Structured user/assistant content may describe a task but cannot grant host
        # mutation authority. Strip only forgeable owned_anchors before localization.
        sanitized = tuple(
            _sanitize_untrusted_message(message)
            if isinstance(message, Mapping)
            else message
            for message in messages
        )
        ready = original_is_ready(sanitized, state)
        writable, creatable = _collect_message_owned_anchors(messages, loop_module)
        if writable or creatable:
            with state._lock:
                if state.mutation_context is not None:
                    state.mutation_context = _authorize_context(
                        state.mutation_context,
                        writable=writable,
                        creatable=creatable,
                    )
                    ready = state.mutation_context.is_mutation_ready
        return ready

    def mutation_target_error(tool_name, arguments, context):
        if tool_name != "apply_source_edit":
            return None
        if context is None or not context.is_mutation_ready:
            return (
                "MUTATION_TARGET_UNBOUND: apply_source_edit requires a host-localized "
                "target file and exact source/new-file evidence before ACT."
            )
        context = _authorize_context(context)
        supplied = ""
        for key in loop_module._SOURCE_EDIT_PATH_KEYS:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                supplied = loop_module._canonical_mutation_path(value)
                break
        pinned = loop_module._canonical_mutation_path(context.target_path)
        allowed = set(_union_paths(context.writable_paths, (pinned,) if pinned else ()))
        if not pinned or not supplied:
            return (
                "MUTATION_TARGET_UNBOUND: apply_source_edit requires an explicit path "
                "matching the host-localized target."
            )
        if supplied not in allowed:
            return (
                "MUTATION_TARGET_DRIFT: writable exact-set "
                f"{sorted(allowed)!r} does not authorize model-supplied path {supplied!r}."
            )
        operation = str(arguments.get("operation") or "").strip().casefold()
        if operation in loop_module._SOURCE_CREATE_OPERATIONS:
            can_create = supplied in set(context.creatable_paths)
            if supplied == pinned:
                can_create = can_create or bool(context.is_new_file)
            if not can_create:
                return (
                    "MUTATION_TARGET_CREATION_CONFLICT: create-file operation is not "
                    f"authorized for existing target {supplied!r}."
                )
        return None

    @wraps(original_context_dict)
    def mutation_context_dict(ctx):
        data = original_context_dict(ctx)
        if ctx is None:
            return data
        authorized = _authorize_context(ctx)
        data["writable_paths"] = list(authorized.writable_paths)
        data["creatable_paths"] = list(authorized.creatable_paths)
        return data

    @wraps(original_generate)
    def generate_with_tools(
        router,
        *,
        config,
        adapter,
        request,
        runtime,
        stage,
        role,
    ):
        if role in {"coder", "coder_safe"} and not _approved_donor_authority(request.messages):
            if _forced_tool_name(request.tool_choice) == _DONOR_TOOL:
                raise loop_module.ModelConfigurationError(
                    "DONOR_SOURCE_UNAUTHORIZED: read_reuse_source cannot be forced "
                    "without an approved host materialized donor receipt/path."
                )
            request = replace(request, tools=_filter_donor_tool_schemas(request.tools))
        return original_generate(
            router,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    loop_module.TargetMutationContext = AuthorizedTargetMutationContext
    loop_module._extract_mutation_context_from_payload = extract
    loop_module.is_mutation_ready = is_mutation_ready
    loop_module._mutation_target_error = mutation_target_error
    loop_module._mutation_context_dict = mutation_context_dict
    loop_module.generate_with_tools = generate_with_tools
    loop_module._planir_owned_anchor_sets = lambda payload: _owned_anchor_sets(payload, loop_module)
    loop_module._approved_donor_source_authority = _approved_donor_authority
    loop_module._filter_donor_tool_schemas = _filter_donor_tool_schemas
    setattr(loop_module, _MARKER, True)


__all__ = ["install"]
