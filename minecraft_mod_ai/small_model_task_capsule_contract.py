from __future__ import annotations

"""Bind a small coder to one host-approved PlanIR task.

Planning, localization, destination selection, and donor selection are host concerns.
Repository retrieval is evidence only.  The model receives a compact task capsule and
chooses semantic edits; any model-authored path is only a hint and is rebound to an exact
owned anchor before the existing mutation/security pipeline executes it.
"""

import contextvars
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

from .root_cause_trace import emit_root_cause, trace_scope

_MARKER = "_mmm_small_model_task_capsule_v1"
_SCHEMA = "mmm/small-model-task-capsule-v1"
_SOURCE_EDIT_TOOL = "apply_source_edit"
_JAVA_VERIFY_TOOLS = frozenset({"java_diagnostics", "jdt_diagnostics"})
_PATH_ALIASES = ("file", "target_path", "target_file")
_ALLOWED_TARGET_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
)
# Only task-local facts that can change the implementation belong in the coder request.
# In particular semantic_outcome is essential; request_context/planner provenance is not.
_COMPACT_TASK_FIELDS = (
    "task_id",
    "task_sha256",
    "semantic_outcome",
    "requirement_refs",
    "gap_refs",
    "owned_anchors",
    "reuse_refs",
    "consumes",
    "provides",
    "depends_on",
    "conditional_predicates",
    "required_gates",
    "acceptance",
    "public_acceptance",
    "done_predicate",
    "impact_probes",
    "production_bindings",
    "artifact_obligations",
    "dependency_reasons",
    "handoff_sha256",
    "asset_bindings",
)
_CURRENT_CAPSULE: contextvars.ContextVar["TaskCapsule | None"] = contextvars.ContextVar(
    "mmm_small_model_task_capsule", default=None
)


class TaskCapsuleContractError(RuntimeError):
    """Approved task ownership could not form an executable small-coder contract."""


@dataclass(frozen=True)
class TaskAnchor:
    kind: str
    path: str
    symbol: str
    status: str
    ownership: str = ""
    module_id: str = ""
    source_set: str = ""

    @property
    def locator(self) -> str:
        return self.path + (f"#{self.symbol}" if self.symbol else "")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "locator": self.locator,
            "status": self.status,
        }
        for key, value in (
            ("ownership", self.ownership),
            ("module_id", self.module_id),
            ("source_set", self.source_set),
        ):
            if value:
                result[key] = value
        return result


@dataclass(frozen=True)
class TaskCapsule:
    task_id: str
    module_kind: str
    primary_path: str
    primary_symbol: str
    anchors: tuple[TaskAnchor, ...]
    reuse_action: str
    task_sha256: str
    capsule_sha256: str

    @property
    def writable_paths(self) -> tuple[str, ...]:
        return tuple(anchor.path for anchor in self.anchors)

    @property
    def creatable_paths(self) -> tuple[str, ...]:
        return tuple(
            anchor.path
            for anchor in self.anchors
            if anchor.status.casefold() == "host_reserved"
        )

    @property
    def test_paths(self) -> tuple[str, ...]:
        return tuple(
            anchor.path
            for anchor in self.anchors
            if anchor.kind == "test"
            or anchor.path.startswith("src/test/java/")
            or PurePosixPath(anchor.path).stem.endswith("Test")
        )

    def anchor_for_path(self, path: str) -> TaskAnchor | None:
        return next((anchor for anchor in self.anchors if anchor.path == path), None)

    def to_host_authority_payload(self) -> dict[str, Any]:
        primary = self.anchor_for_path(self.primary_path)
        if primary is None:
            raise TaskCapsuleContractError(
                "TASK_CAPSULE_PRIMARY_MISSING: primary path disappeared from capsule anchors."
            )
        task: dict[str, Any] = {
            "task_id": self.task_id,
            "owned_anchors": [anchor.to_dict() for anchor in self.anchors],
            # Donor reuse changes implementation ingredients, never destination authority.
            "production_bindings": [
                {
                    "task_ref": self.task_id,
                    "reuse_action": "fresh",
                    "owned_anchors": [primary.to_dict()],
                }
            ],
        }
        if self.task_sha256:
            task["task_sha256"] = self.task_sha256
        return {
            "schema_version": _SCHEMA,
            "phase": "implement_module",
            "task_id": self.task_id,
            "capsule_sha256": self.capsule_sha256,
            "mutation_target": {
                "path": self.primary_path,
                "symbol": self.primary_symbol,
                "mode": "host_bound_exact_task_target",
                "policy": (
                    "Model path text is non-authoritative. Every edit is rebound to this "
                    "task's concrete owned-anchor set before execution."
                ),
            },
            "module": {
                "module_id": self.task_id,
                "kind": self.module_kind,
                "config": {"evidence_task": task},
            },
            "reuse_action": self.reuse_action,
        }


def _canonical_path(locator: Any) -> tuple[str, str]:
    raw = str(locator or "").replace("\\", "/").strip()
    if not raw:
        return "", ""
    raw_path, separator, raw_symbol = raw.partition("#")
    path = raw_path.strip()
    while path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or ":" in path:
        return "", ""
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return "", ""
    normalized = PurePosixPath(path).as_posix()
    if not any(normalized.startswith(prefix) for prefix in _ALLOWED_TARGET_PREFIXES):
        return "", ""
    return normalized, raw_symbol.strip() if separator else ""


def _task_anchor(value: Any) -> TaskAnchor | None:
    if not isinstance(value, Mapping):
        return None
    path, symbol = _canonical_path(value.get("locator"))
    if not path:
        return None
    return TaskAnchor(
        kind=str(value.get("kind") or "").strip(),
        path=path,
        symbol=symbol,
        status=str(value.get("status") or "").strip(),
        ownership=str(value.get("ownership") or "").strip(),
        module_id=str(value.get("module_id") or "").strip(),
        source_set=str(value.get("source_set") or "").strip(),
    )


def _evidence_task(module: Any) -> Mapping[str, Any] | None:
    config = getattr(module, "config", None)
    if not isinstance(config, Mapping):
        return None
    task = config.get("evidence_task")
    return task if isinstance(task, Mapping) else None


def _matching_bindings(
    task: Mapping[str, Any], task_id: str
) -> tuple[Mapping[str, Any], ...]:
    raw = task.get("production_bindings")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(
        binding
        for binding in raw
        if isinstance(binding, Mapping)
        and str(binding.get("task_ref") or "").strip() == task_id
    )


def _binding_symbol_candidates(
    bindings: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for binding in bindings:
        raw_anchors = binding.get("owned_anchors")
        if not isinstance(raw_anchors, Sequence) or isinstance(
            raw_anchors, (str, bytes, bytearray)
        ):
            continue
        for raw_anchor in raw_anchors:
            anchor = _task_anchor(raw_anchor)
            if anchor is None or anchor.kind != "symbol":
                continue
            item = (anchor.path, anchor.symbol)
            if item not in result:
                result.append(item)
    return tuple(result)


def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_task_capsule(module: Any) -> TaskCapsule | None:
    """Compile approved PlanIR authority; never fall back from malformed approved tasks."""

    if str(getattr(module, "kind", "") or "").strip() != "custom_java":
        return None
    task = _evidence_task(module)
    if task is None:
        return None

    module_id = str(getattr(module, "module_id", "") or "").strip()
    task_id = str(task.get("task_id") or "").strip()
    if not module_id or not task_id or module_id != task_id:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_ID_MISMATCH: module_id and evidence_task.task_id must match."
        )

    raw_anchors = task.get("owned_anchors")
    if not isinstance(raw_anchors, Sequence) or isinstance(
        raw_anchors, (str, bytes, bytearray)
    ):
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_ANCHORS_MISSING: approved custom task has no owned_anchors."
        )
    anchors: list[TaskAnchor] = []
    for raw_anchor in raw_anchors:
        anchor = _task_anchor(raw_anchor)
        if anchor is not None and anchor not in anchors:
            anchors.append(anchor)
    if not anchors:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_NO_CONCRETE_PATHS: owned_anchors contain no writable workspace paths."
        )

    bindings = _matching_bindings(task, task_id)
    if not bindings:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_BINDING_MISSING: approved custom task has no matching production binding."
        )
    candidates = _binding_symbol_candidates(bindings)
    if len(candidates) != 1:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_PRIMARY_AMBIGUOUS: expected exactly one production-binding "
            f"source symbol, found {len(candidates)}."
        )
    primary_path, primary_symbol = candidates[0]
    primary = next((anchor for anchor in anchors if anchor.path == primary_path), None)
    if primary is None:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_PRIMARY_NOT_OWNED: production-binding source is not a task owned_anchor."
        )
    if not primary_path.startswith("src/main/java/") or not primary_path.endswith(".java"):
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_PRIMARY_NOT_JAVA: custom_java primary must be under src/main/java."
        )
    if primary.status.casefold() != "host_reserved":
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_PRIMARY_NOT_RESERVED: planned custom-Java destination must be host_reserved."
        )

    actions = {
        str(binding.get("reuse_action") or "").strip().casefold()
        for binding in bindings
        if str(binding.get("reuse_action") or "").strip()
    }
    if len(actions) > 1:
        raise TaskCapsuleContractError(
            f"TASK_CAPSULE_REUSE_AMBIGUOUS: matching bindings disagree: {sorted(actions)!r}."
        )
    reuse_action = next(iter(actions), "fresh")
    task_sha = str(task.get("task_sha256") or "").strip()
    digest_input = {
        "task_id": task_id,
        "module_kind": "custom_java",
        "primary_path": primary_path,
        "primary_symbol": primary_symbol,
        "anchors": [anchor.to_dict() for anchor in anchors],
        "reuse_action": reuse_action,
        "task_sha256": task_sha,
    }
    return TaskCapsule(
        task_id=task_id,
        module_kind="custom_java",
        primary_path=primary_path,
        primary_symbol=primary_symbol,
        anchors=tuple(anchors),
        reuse_action=reuse_action,
        task_sha256=task_sha,
        capsule_sha256=_sha256(digest_input),
    )


def compact_task_local_module_contract(module: Any) -> dict[str, Any]:
    """Project PlanIR to task-local facts; remove whole-plan/research provenance noise."""

    task = _evidence_task(module)
    if task is None:
        config = getattr(module, "config", None)
        return {
            "module_id": str(getattr(module, "module_id", "")),
            "kind": str(getattr(module, "kind", "")),
            "config": dict(config) if isinstance(config, Mapping) else {},
            "depends_on": list(getattr(module, "depends_on", ()) or ()),
            "required_gates": list(getattr(module, "required_gates", ()) or ()),
        }
    compact_task = {
        key: copy.deepcopy(task[key]) for key in _COMPACT_TASK_FIELDS if key in task
    }
    return {
        "module_id": str(getattr(module, "module_id", "")),
        "kind": str(getattr(module, "kind", "")),
        "evidence_task": compact_task,
        "depends_on": list(getattr(module, "depends_on", ()) or ()),
        "required_gates": list(getattr(module, "required_gates", ()) or ()),
    }


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    return str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""


def narrow_source_edit_schema(schema: Any, capsule: TaskCapsule) -> Any:
    """Show the model only task-owned paths; the host validator remains authoritative."""

    if _tool_name(schema) != _SOURCE_EDIT_TOOL or not isinstance(schema, Mapping):
        return schema
    narrowed = copy.deepcopy(dict(schema))
    function = narrowed.get("function")
    parameters = function.get("parameters") if isinstance(function, dict) else None
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(properties, dict):
        return narrowed
    path_schema = dict(properties.get("path") or {"type": "string"})
    path_schema.update(
        {
            "type": "string",
            "enum": list(capsule.writable_paths),
            "description": (
                "Host-selected PlanIR destination. Choose one exact path from this enum; "
                "retrieval cannot add write targets."
            ),
        }
    )
    properties["path"] = path_schema
    for alias in _PATH_ALIASES:
        properties.pop(alias, None)
    parameters["required"] = list(
        dict.fromkeys([*(parameters.get("required") or ()), "operation", "path"])
    )
    return narrowed


def narrow_task_tool_schema(schema: Any, capsule: TaskCapsule) -> Any:
    """Expose exact task-owned paths to mutation and diagnostic tools."""

    narrowed = narrow_source_edit_schema(schema, capsule)
    name = _tool_name(narrowed)
    if name not in _JAVA_VERIFY_TOOLS or not isinstance(narrowed, Mapping):
        return narrowed
    result = copy.deepcopy(dict(narrowed))
    function = result.get("function")
    parameters = function.get("parameters") if isinstance(function, dict) else None
    if not isinstance(parameters, dict):
        return result
    properties = parameters.setdefault("properties", {})
    java_paths = [capsule.primary_path]
    properties["relative_files"] = {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "string", "enum": java_paths},
        "description": "Exact host-owned task Java paths.",
    }
    properties["project_root"] = {
        "type": "string",
        "enum": ["."],
        "description": "Host-bound project root relative to the runtime workspace.",
    }
    for alias in (
        "diagnostics_path",
        "file_path",
        "diagnostics_command",
        "diagnostics_config",
    ):
        properties.pop(alias, None)
    parameters["additionalProperties"] = False
    parameters["required"] = ["project_root", "relative_files"]
    return result


def _resolve_model_path(arguments: Mapping[str, Any], capsule: TaskCapsule) -> str:
    raw = ""
    for key in ("path", *_PATH_ALIASES):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.replace("\\", "/").strip()
            break
    while raw.startswith("./"):
        raw = raw[2:]
    if raw in capsule.writable_paths:
        return raw

    basename = PurePosixPath(raw).name if raw else ""
    if basename:
        matches = tuple(
            path for path in capsule.writable_paths if PurePosixPath(path).name == basename
        )
        if len(matches) == 1:
            return matches[0]

    hint = raw.casefold()
    if "src/test/" in hint or "/test/" in hint or basename.casefold().endswith("test.java"):
        if len(capsule.test_paths) == 1:
            return capsule.test_paths[0]

    suffix = PurePosixPath(raw).suffix.casefold() if raw else ""
    if suffix and suffix != ".java":
        matches = tuple(
            path
            for path in capsule.writable_paths
            if PurePosixPath(path).suffix.casefold() == suffix
        )
        if len(matches) == 1:
            return matches[0]
    return capsule.primary_path


def bind_source_edit_arguments(
    arguments: Mapping[str, Any], capsule: TaskCapsule
) -> dict[str, Any]:
    """Treat model path text as a hint and bind the executable host-owned destination."""

    bound = dict(arguments)
    host_path = _resolve_model_path(bound, capsule)
    for alias in _PATH_ALIASES:
        bound.pop(alias, None)
    bound["path"] = host_path

    operation = str(bound.get("operation") or "").strip().casefold()
    if operation in {"create_java_type", "create_type", "create_class"}:
        anchor = capsule.anchor_for_path(host_path)
        relative = ""
        for prefix in ("src/main/java/", "src/test/java/"):
            if host_path.startswith(prefix):
                relative = host_path.removeprefix(prefix)
                break
        if relative and "/" in relative:
            bound["package_name"] = relative.rsplit("/", 1)[0].replace("/", ".")
        if anchor is not None and anchor.symbol:
            declaration = str(bound.get("declaration") or "").strip()
            if not declaration or anchor.symbol not in declaration:
                bound["declaration"] = f"public final class {anchor.symbol}"
    return bound


def bind_verifier_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
    capsule: TaskCapsule,
) -> dict[str, Any]:
    """Replace model diagnostic paths with exact PlanIR-owned Java targets."""

    if tool_name not in _JAVA_VERIFY_TOOLS:
        return dict(arguments)
    java_paths = [capsule.primary_path] if capsule.primary_path.endswith(".java") else []
    if not java_paths:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_VERIFIER_TARGET_MISSING: no task-owned Java path is available."
        )
    timeout = arguments.get("timeout_seconds", 60)
    if type(timeout) is not int or not 1 <= timeout <= 600:
        timeout = 60
    return {
        "project_root": ".",
        "relative_files": java_paths,
        "timeout_seconds": timeout,
    }


def _bind_tool_call(call: Any, capsule: TaskCapsule) -> Any:
    tool_name = str(getattr(call, "name", "") or "")
    if tool_name != _SOURCE_EDIT_TOOL and tool_name not in _JAVA_VERIFY_TOOLS:
        return call
    arguments = getattr(call, "arguments", None)
    if not isinstance(arguments, Mapping):
        arguments = {}
    original_path = next(
        (
            value
            for key in ("path", *_PATH_ALIASES)
            if isinstance((value := arguments.get(key)), str) and value.strip()
        ),
        "",
    )
    bound = (
        bind_source_edit_arguments(arguments, capsule)
        if tool_name == _SOURCE_EDIT_TOOL
        else bind_verifier_arguments(tool_name, arguments, capsule)
    )
    if (
        tool_name == _SOURCE_EDIT_TOOL
        and original_path.replace("\\", "/").strip() != bound["path"]
    ):
        print(
            "task capsule: rebound model path",
            f"task={capsule.task_id}",
            f"hint={original_path!r}",
            f"target={bound['path']!r}",
            flush=True,
        )
    emit_root_cause(
        "task_tool_arguments_bound",
        stage="generation",
        operation=tool_name,
        gate="task_capsule_authority",
        result="PASS",
        reason="model arguments normalized to exact task-owned targets",
        details={
            "task_id": capsule.task_id,
            "primary_path": capsule.primary_path,
            "raw_arguments": getattr(call, "raw_arguments", ""),
            "parsed_arguments": dict(arguments),
            "normalized_arguments": bound,
        },
    )
    try:
        return replace(
            call,
            arguments=bound,
            raw_arguments=json.dumps(bound, ensure_ascii=False, separators=(",", ":")),
        )
    except TypeError as exc:
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_TOOLCALL_UNSUPPORTED: cannot bind source-edit arguments."
        ) from exc


class _TaskBoundAdapter:
    """Proxy generation while preserving the underlying adapter's optional capabilities."""

    def __init__(self, adapter: Any, capsule: TaskCapsule) -> None:
        self._adapter = adapter
        self._capsule = capsule
        # progress_aware_tool_loop deliberately uses inspect.getattr_static to detect
        # exact accounting support.  Do not define a class method that makes an adapter
        # without this capability appear to support it.
        counter = getattr(adapter, "input_context_accounting", None)
        if callable(counter):
            self.input_context_accounting = counter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def generate(self, request: Any) -> Any:
        return self._adapter.generate(request)

    def generate_turn(self, request: Any) -> Any:
        emit_root_cause(
            "task_coder_request",
            stage="generation",
            operation="task_bound_adapter",
            gate="coder_boundary",
            result="START",
            details={
                "task_id": self._capsule.task_id,
                "primary_path": self._capsule.primary_path,
                "request": request,
            },
        )
        response = self._adapter.generate_turn(request)
        emit_root_cause(
            "task_coder_response_raw",
            stage="generation",
            operation="task_bound_adapter",
            gate="coder_boundary",
            result="PASS",
            details={
                "task_id": self._capsule.task_id,
                "content": getattr(response, "content", ""),
                "tool_calls": [
                    {
                        "id": getattr(call, "id", ""),
                        "name": getattr(call, "name", ""),
                        "raw_arguments": getattr(call, "raw_arguments", ""),
                        "parsed_arguments": getattr(call, "arguments", {}),
                    }
                    for call in tuple(getattr(response, "tool_calls", ()) or ())
                ],
            },
        )
        calls = tuple(
            _bind_tool_call(call, self._capsule)
            for call in tuple(getattr(response, "tool_calls", ()) or ())
        )
        try:
            normalized = replace(response, tool_calls=calls)
            emit_root_cause(
                "task_coder_response_normalized",
                stage="generation",
                operation="task_bound_adapter",
                gate="task_capsule_authority",
                result="PASS",
                details={
                    "task_id": self._capsule.task_id,
                    "tool_calls": [
                        {
                            "id": getattr(call, "id", ""),
                            "name": getattr(call, "name", ""),
                            "raw_arguments": getattr(call, "raw_arguments", ""),
                            "arguments": getattr(call, "arguments", {}),
                        }
                        for call in calls
                    ],
                },
            )
            return normalized
        except TypeError as exc:
            raise TaskCapsuleContractError(
                "TASK_CAPSULE_RESPONSE_UNSUPPORTED: generation response is not replaceable."
            ) from exc


def _authority_message(capsule: TaskCapsule) -> dict[str, str]:
    return {
        "role": "developer",
        "content": json.dumps(
            capsule.to_host_authority_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _insert_authority_message(
    messages: Sequence[Mapping[str, Any]], capsule: TaskCapsule
) -> tuple[Mapping[str, Any], ...]:
    result = [dict(message) for message in messages]
    insert_at = (
        1
        if result
        and str(result[0].get("role") or "").strip().casefold() == "system"
        else 0
    )
    result.insert(insert_at, _authority_message(capsule))
    return tuple(result)


def install() -> None:
    """Install last so task authority remains outside all generic generation wrappers."""

    from . import custom_module_generator, progress_aware_tool_loop

    if getattr(progress_aware_tool_loop, _MARKER, False):
        return
    Generator = custom_module_generator.CustomModuleGenerator
    original_generate = Generator.generate
    original_loop = progress_aware_tool_loop.generate_with_tools
    original_contract = custom_module_generator._task_local_module_contract

    @wraps(original_generate)
    def generate(self: Any, *args: Any, **kwargs: Any):
        capsule = compile_task_capsule(kwargs.get("module"))
        token = _CURRENT_CAPSULE.set(capsule)
        emit_root_cause(
            "task_capsule_compiled",
            stage="generation",
            operation="compile_task_capsule",
            gate="planir_task_authority",
            result="PASS" if capsule is not None else "SKIP",
            details={
                "task_id": capsule.task_id if capsule else "",
                "primary_path": capsule.primary_path if capsule else "",
                "writable_paths": capsule.writable_paths if capsule else (),
                "test_paths": capsule.test_paths if capsule else (),
                "required_gates": capsule.required_gates if capsule else (),
            },
        )
        try:
            return original_generate(self, *args, **kwargs)
        except BaseException as exc:
            emit_root_cause(
                "task_capsule_generation_failure",
                stage="generation",
                operation="custom_module_generate",
                gate="task_capsule",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={"task_id": capsule.task_id if capsule else ""},
                exc=exc,
            )
            raise
        finally:
            _CURRENT_CAPSULE.reset(token)

    @wraps(original_loop)
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
        capsule = _CURRENT_CAPSULE.get()
        if capsule is None or stage != "generation" or role not in {"coder", "coder_safe"}:
            return original_loop(
                router,
                config=config,
                adapter=adapter,
                request=request,
                runtime=runtime,
                stage=stage,
                role=role,
            )
        request = replace(
            request,
            messages=_insert_authority_message(request.messages, capsule),
            tools=tuple(
                narrow_task_tool_schema(schema, capsule) for schema in tuple(request.tools)
            ),
        )
        print(
            "task capsule: active",
            f"task={capsule.task_id}",
            f"primary={capsule.primary_path}",
            f"writable={len(capsule.writable_paths)}",
            f"reuse={capsule.reuse_action}",
            flush=True,
        )
        with trace_scope(f"task:{capsule.task_id}"):
            emit_root_cause(
                "task_capsule_activated",
                stage=stage,
                operation="task_capsule",
                gate="coder_input",
                result="PASS",
                details={
                    "task_id": capsule.task_id,
                    "primary_path": capsule.primary_path,
                    "writable_paths": capsule.writable_paths,
                    "test_paths": capsule.test_paths,
                    "reuse_action": capsule.reuse_action,
                    "messages": request.messages,
                    "narrowed_tools": request.tools,
                },
            )
            return original_loop(
                router,
                config=config,
                adapter=_TaskBoundAdapter(adapter, capsule),
                request=request,
                runtime=runtime,
                stage=stage,
                role=role,
            )

    @wraps(original_contract)
    def task_local_module_contract(module: Any) -> dict[str, Any]:
        if _evidence_task(module) is None:
            return original_contract(module)
        return compact_task_local_module_contract(module)

    generate._mmm_small_model_task_capsule = True  # type: ignore[attr-defined]
    generate_with_tools._mmm_small_model_task_capsule = True  # type: ignore[attr-defined]
    task_local_module_contract._mmm_small_model_task_capsule = True  # type: ignore[attr-defined]
    Generator.generate = generate
    progress_aware_tool_loop.generate_with_tools = generate_with_tools
    custom_module_generator._task_local_module_contract = task_local_module_contract
    setattr(progress_aware_tool_loop, _MARKER, True)
    setattr(custom_module_generator, _MARKER, True)


def assert_installed() -> None:
    from . import custom_module_generator, progress_aware_tool_loop

    checks = (
        getattr(progress_aware_tool_loop, _MARKER, False),
        getattr(
            progress_aware_tool_loop.generate_with_tools,
            "_mmm_small_model_task_capsule",
            False,
        ),
        getattr(
            custom_module_generator.CustomModuleGenerator.generate,
            "_mmm_small_model_task_capsule",
            False,
        ),
        getattr(
            custom_module_generator._task_local_module_contract,
            "_mmm_small_model_task_capsule",
            False,
        ),
    )
    if not all(checks):
        raise RuntimeError("Small-model task capsule contract is not final/active.")


__all__ = [
    "TaskAnchor",
    "TaskCapsule",
    "TaskCapsuleContractError",
    "assert_installed",
    "bind_source_edit_arguments",
    "bind_verifier_arguments",
    "compact_task_local_module_contract",
    "compile_task_capsule",
    "install",
    "narrow_source_edit_schema",
    "narrow_task_tool_schema",
]
