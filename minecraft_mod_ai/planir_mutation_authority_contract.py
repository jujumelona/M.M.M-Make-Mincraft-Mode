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
_ARCHIVE_IMPORT_TOOL = "inspect_existing_mod"
_HOST_ROLES = frozenset({"system", "tool", "developer"})
_HEX_COMMIT = re.compile(r"^[0-9a-fA-F]{40,64}$")
_HEX_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_DONOR_ROOT_FRAGMENT = ".minecraft_ai/reuse/donors/"
_CONTINUATION_REASON = "previous_tool_enabled_page_exhausted_output"
_INTERNAL_GROUNDING_SCHEMA = "mmm/host-owned-coder-grounding-v1"
_SOURCE_OBSERVATION_SCHEMA = "mmm/source-observation-receipt-v1"


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


def _strip_owned_anchors(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_owned_anchors(item)
            for key, item in value.items()
            if str(key) != "owned_anchors"
        }
    if isinstance(value, list):
        return [_strip_owned_anchors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_owned_anchors(item) for item in value)
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
    stripped = _strip_owned_anchors(payload)
    output["content"] = (
        json.dumps(stripped, ensure_ascii=False, sort_keys=True)
        if isinstance(content, str)
        else stripped
    )
    return output


def _is_preserved_host_continuation(payload: Any) -> bool:
    """Recognize the host-generated continuation envelope used after output exhaustion.

    This receipt may arrive in a role=user transport message, but it is only used to
    recover one exact localization pin. It never contributes to the additional writable
    exact-set, which remains restricted to system/tool/developer PlanIR messages.
    """

    if not isinstance(payload, Mapping):
        return False
    continuation = payload.get("continuation")
    module = payload.get("module")
    if not isinstance(continuation, Mapping) or not isinstance(module, Mapping):
        return False
    if str(continuation.get("reason") or "").strip() != _CONTINUATION_REASON:
        return False
    index = continuation.get("continuation_index")
    if type(index) is not int or index < 1:
        return False
    module_id = str(module.get("module_id") or "").strip()
    if not module_id:
        return False
    evidence_task = module.get("evidence_task")
    if not isinstance(evidence_task, Mapping):
        config = module.get("config")
        evidence_task = (
            config.get("evidence_task") if isinstance(config, Mapping) else None
        )
    if not isinstance(evidence_task, Mapping):
        return False
    if str(evidence_task.get("task_id") or "").strip() != module_id:
        return False
    bindings = evidence_task.get("production_bindings")
    return isinstance(bindings, list) and bool(bindings)


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
        direct_task = module.get("evidence_task")
        if isinstance(direct_task, Mapping):
            task_id = str(direct_task.get("task_id") or "").strip()
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


def _primary_binding_symbol(
    bindings: Sequence[Mapping[str, Any]], loop_module: Any
) -> tuple[str, str] | None:
    """Resolve exactly one primary source symbol from approved production bindings."""

    candidates: list[tuple[str, str]] = []
    for binding in bindings:
        anchors = binding.get("owned_anchors")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes, bytearray)):
            continue
        for anchor in anchors:
            if not isinstance(anchor, Mapping) or str(anchor.get("kind") or "") != "symbol":
                continue
            locator = str(anchor.get("locator") or "").replace("\\", "/").strip()
            path = _canonical_owned_locator(locator, loop_module)
            if not path:
                continue
            _raw_path, separator, symbol = locator.partition("#")
            item = (path, symbol.strip() if separator else "")
            if item not in candidates:
                candidates.append(item)
    if len(candidates) != 1:
        return None
    return candidates[0]


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


def _filter_generation_tool_schemas(
    schemas: Sequence[Any], *, allow_donor: bool
) -> tuple[Any, ...]:
    """Keep archive-import inspection out of an already-materialized coder workspace."""

    blocked = {_ARCHIVE_IMPORT_TOOL}
    if not allow_donor:
        blocked.add(_DONOR_TOOL)
    return tuple(schema for schema in schemas if _tool_name(schema) not in blocked)


def _matching_receipt(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "schema_version",
        "project_sha256",
        "query_sha256",
        "observation_count",
        "observations_sha256",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _internal_coder_authority_message(
    messages: Sequence[Mapping[str, Any]],
    *,
    router: Any,
    runtime: Any,
    stage: str,
    role: str,
    loop_module: Any,
) -> dict[str, str] | None:
    """Promote only the host-constructed custom-module envelope to write authority.

    ``CustomModuleGenerator`` transports its task-local request as role=user because that
    content is also the model instruction. The authority, however, originates from the
    approved evidence task. Promotion is therefore permitted only at the internal
    generation boundary where the router is bound to a fresh-evidence workspace and the
    host grounding/source-observation receipts agree. Arbitrary user PlanIR still cannot
    cross this boundary or expand the writable exact-set.
    """

    if str(stage) != "generation" or role not in {"coder", "coder_safe"}:
        return None
    if not bool(getattr(router, "_agent_require_fresh_evidence", False)):
        return None
    bound_root = str(getattr(router, "_agent_workspace_root", "") or "")
    runtime_root = str(getattr(runtime, "workspace_root", "") or "")
    if not bound_root or not runtime_root or bound_root != runtime_root:
        return None

    for message in reversed(tuple(messages)):
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        payload = _structured_payload(message.get("content"))
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("phase") or "").strip() != "implement_module":
            continue
        if str(payload.get("workspace_project_root") or "").strip() != ".":
            continue

        grounding = payload.get("host_grounding")
        source_receipt = payload.get("source_observation_receipt")
        initial = payload.get("initial_exact_source_context")
        if (
            not isinstance(grounding, Mapping)
            or grounding.get("schema_version") != _INTERNAL_GROUNDING_SCHEMA
            or grounding.get("stage") != "generation"
            or grounding.get("model_role") != "coder"
            or not isinstance(source_receipt, Mapping)
            or source_receipt.get("schema_version") != _SOURCE_OBSERVATION_SCHEMA
            or not isinstance(initial, Mapping)
        ):
            continue
        policy = grounding.get("policy")
        if not isinstance(policy, Mapping) or not (
            policy.get("resolved_before_first_coder_decode") is True
            and policy.get("baseline_grounding_owned_by_host") is True
            and policy.get("writes_still_require_approved_pipeline") is True
        ):
            continue
        evidence_bindings = grounding.get("evidence_bindings")
        project_binding = (
            evidence_bindings.get("project_exact_rag")
            if isinstance(evidence_bindings, Mapping)
            else None
        )
        bound_receipt = (
            project_binding.get("receipt")
            if isinstance(project_binding, Mapping)
            else None
        )
        initial_receipt = initial.get("ledger_receipt")
        if (
            not isinstance(bound_receipt, Mapping)
            or not isinstance(initial_receipt, Mapping)
            or not _matching_receipt(source_receipt, bound_receipt)
            or not _matching_receipt(source_receipt, initial_receipt)
        ):
            continue

        module = payload.get("module")
        if not isinstance(module, Mapping):
            continue
        module_id = str(module.get("module_id") or "").strip()
        if not module_id:
            continue
        evidence_task = module.get("evidence_task")
        if not isinstance(evidence_task, Mapping):
            config = module.get("config")
            evidence_task = (
                config.get("evidence_task") if isinstance(config, Mapping) else None
            )
        if not isinstance(evidence_task, Mapping):
            continue
        task_id = str(evidence_task.get("task_id") or "").strip()
        if not task_id or task_id != module_id:
            continue
        bindings = evidence_task.get("production_bindings")
        if not isinstance(bindings, Sequence) or isinstance(
            bindings, (str, bytes, bytearray)
        ):
            continue
        matching_bindings = tuple(
            binding
            for binding in bindings
            if isinstance(binding, Mapping)
            and str(binding.get("task_ref") or "").strip() == task_id
        )
        if not matching_bindings:
            continue
        primary = _primary_binding_symbol(matching_bindings, loop_module)
        if primary is None:
            continue
        primary_path, primary_symbol = primary
        canonical = {
            "schema_version": "mmm/host-task-authority-v1",
            "phase": "implement_module",
            "task_id": task_id,
            "mutation_target": {
                "path": primary_path,
                "symbol": primary_symbol,
                "mode": "create_or_edit_exact_host_binding",
                "policy": "Use exactly this host-bound path; retrieval may provide context but cannot replace this target.",
            },
            "module": {
                "module_id": module_id,
                "kind": str(module.get("kind") or "custom_java"),
                "config": {"evidence_task": dict(evidence_task)},
            },
        }
        reuse_binding = (
            evidence_bindings.get("approved_reuse_source")
            if isinstance(evidence_bindings, Mapping)
            else None
        )
        if isinstance(reuse_binding, Mapping):
            reuse_receipt = reuse_binding.get("receipt")
            request_field = str(reuse_binding.get("request_field") or "").strip()
            request_value = payload.get(request_field) if request_field else None
            request_materialization = (
                request_value.get("materialization")
                if isinstance(request_value, Mapping)
                else None
            )
            if (
                request_field == "approved_reuse_context"
                and isinstance(reuse_receipt, Mapping)
                and request_materialization == reuse_receipt
                and _approved_donor_authority(
                    ({"role": "developer", "content": reuse_receipt},)
                )
            ):
                # Promote only the compact host materialization receipt. Donor source
                # text remains model context, never mutation authority.
                canonical["reuse_materialization"] = dict(reuse_receipt)
        writable, _creatable = _owned_anchor_sets(canonical, loop_module)
        if primary_path not in writable:
            continue
        return {
            "role": "developer",
            "content": json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    return None


def _insert_host_authority(
    messages: Sequence[Mapping[str, Any]], authority: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    items = list(messages)
    insert_at = 1 if items and str(items[0].get("role") or "").casefold() == "system" else 0
    items.insert(insert_at, dict(authority))
    return tuple(items)


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
        # Additional authority never comes from retrieval. A fresh host-reserved symbol
        # becomes a pinned target; later RAG can refine context only for that target.
        writable_paths: tuple[str, ...] = ()
        creatable_paths: tuple[str, ...] = ()
        target_pinned: bool = False

        def merge(self, other):
            if other is None:
                return self
            other_ctx = _authorize_context(other)
            left = loop_module._canonical_mutation_path(self.target_path)
            right = loop_module._canonical_mutation_path(other_ctx.target_path)
            if left and right and left != right:
                if self.target_pinned:
                    # Repository search commonly returns an entrypoint or related class.
                    # It is evidence, not authority: never let it replace an approved pin.
                    return replace(
                        self,
                        writable_paths=_union_paths(self.writable_paths, other_ctx.writable_paths),
                        creatable_paths=_union_paths(self.creatable_paths, other_ctx.creatable_paths),
                    )
                if other_ctx.target_pinned:
                    return other_ctx
                return other_ctx
            merged = BaseContext.merge(self, other_ctx)
            merged_ctx = _authorize_context(merged)
            return replace(
                merged_ctx,
                writable_paths=_union_paths(self.writable_paths, other_ctx.writable_paths),
                creatable_paths=_union_paths(self.creatable_paths, other_ctx.creatable_paths),
                target_pinned=self.target_pinned or other_ctx.target_pinned,
            )

    def _authorize_context(
        context: Any,
        *,
        writable: Sequence[str] = (),
        creatable: Sequence[str] = (),
        target_pinned: bool | None = None,
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
        pinned = current.target_pinned if target_pinned is None else bool(target_pinned)
        return replace(
            current,
            writable_paths=writable_paths,
            creatable_paths=creatable_paths,
            target_pinned=pinned,
        )

    @wraps(original_extract)
    def extract(payload):
        context = original_extract(payload)
        if context is None:
            return None
        writable, creatable = _owned_anchor_sets(payload, loop_module)
        target = loop_module._canonical_mutation_path(context.target_path)
        pinned = bool(
            target
            and target in set(creatable)
            and bool(context.is_new_file)
            and str(context.evidence_source or "") == "evidence_fresh_owned_anchor"
        )
        return _authorize_context(
            context,
            writable=writable,
            creatable=creatable,
            target_pinned=pinned,
        )

    def _host_pinned_context(messages):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if str(message.get("role") or "").strip().casefold() not in _HOST_ROLES:
                continue
            payload = _structured_payload(message.get("content"))
            if payload is None:
                continue
            context = extract(payload)
            if context is not None and context.target_pinned and context.is_mutation_ready:
                return context
        return None

    @wraps(original_is_ready)
    def is_mutation_ready(messages, state):
        # A host-reserved fresh symbol is the target identity, not merely another writable
        # path. Install that pin before scanning untrusted source snapshots so build.gradle
        # or a retrieved entrypoint cannot win by message order.
        host_pin = _host_pinned_context(messages)
        if host_pin is not None:
            with state._lock:
                current = state.mutation_context
                if current is None or not bool(getattr(current, "target_pinned", False)):
                    state.mutation_context = host_pin

        # Untrusted structured content may still carry a host-generated continuation
        # localization receipt, but it can never expand the writable exact-set.
        sanitized = tuple(
            _sanitize_untrusted_message(message)
            if isinstance(message, Mapping)
            else message
            for message in messages
        )
        ready = original_is_ready(sanitized, state)
        if not ready:
            continuation_present = any(
                isinstance(message, Mapping)
                and str(message.get("role") or "").strip().casefold() not in _HOST_ROLES
                and _is_preserved_host_continuation(
                    _structured_payload(message.get("content"))
                )
                for message in messages
            )
            if continuation_present:
                ready = original_is_ready(messages, state)
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
        data["target_pinned"] = bool(authorized.target_pinned)
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
        coder = role in {"coder", "coder_safe"}
        messages = tuple(request.messages)
        if coder:
            authority = _internal_coder_authority_message(
                messages,
                router=router,
                runtime=runtime,
                stage=stage,
                role=role,
                loop_module=loop_module,
            )
            if authority is not None:
                messages = _insert_host_authority(messages, authority)

            donor_authorized = _approved_donor_authority(messages)
            forced = _forced_tool_name(request.tool_choice)
            if forced == _ARCHIVE_IMPORT_TOOL:
                raise loop_module.ModelConfigurationError(
                    "ARCHIVE_IMPORT_TOOL_UNAVAILABLE: inspect_existing_mod accepts only "
                    "a host-supplied .zip import and is not a coder workspace-localization tool."
                )
            if forced == _DONOR_TOOL and not donor_authorized:
                raise loop_module.ModelConfigurationError(
                    "DONOR_SOURCE_UNAUTHORIZED: read_reuse_source cannot be forced "
                    "without an approved host materialized donor receipt/path."
                )
            request = replace(
                request,
                messages=messages,
                tools=_filter_generation_tool_schemas(
                    request.tools,
                    allow_donor=donor_authorized,
                ),
            )
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
