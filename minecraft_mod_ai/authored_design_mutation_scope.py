from __future__ import annotations

"""Allow host-authored designs to mutate a bounded source-set instead of one placeholder file.

Ordinary module tasks keep the exact-file mutation authority installed by the PlanIR/direct-task
contracts. A saved authored design is different: the host explicitly asks the coder to choose
the source/resource files needed to realize the design. This post-final contract broadens write
authority only while the trusted direct-task ContextVar proves that the current internal request
is the matching ``implement_authored_design`` task.

The broadened scope is still fail-closed:
* only the four generated source/resource roots are writable;
* absolute paths, drive/URI paths and traversal are rejected;
* deletes remain forbidden;
* no user/model payload can activate the scope without the host's active direct-task authority.
"""

import contextvars
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

_MARKER = "_mmm_authored_design_mutation_scope_v1"
_AUTHORED_PHASE = "implement_authored_design"
_ALLOWED_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
)
_DELETE_OPERATIONS = frozenset({"delete", "delete_file", "remove", "remove_file"})


@dataclass(frozen=True)
class AuthoredMutationScope:
    task_id: str
    allowed_prefixes: tuple[str, ...] = _ALLOWED_PREFIXES


_ACTIVE_SCOPE: contextvars.ContextVar[AuthoredMutationScope | None] = contextvars.ContextVar(
    "mmm_authored_design_mutation_scope",
    default=None,
)


def _structured_payload(content: Any) -> Mapping[str, Any] | None:
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        return None
    raw = content.strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _module_id(module: Any) -> str:
    if not isinstance(module, Mapping):
        return ""
    return str(module.get("module_id") or "").strip()


def _resolve_authored_scope(
    messages: Sequence[Mapping[str, Any]],
    *,
    authority: Any,
    stage: str,
    role: str,
) -> AuthoredMutationScope | None:
    """Resolve scope only from a matching host-held direct-task authority."""

    if authority is None or str(stage) != "generation" or role not in {"coder", "coder_safe"}:
        return None
    task_id = str(getattr(authority, "task_id", "") or "").strip()
    if not task_id:
        return None

    for message in reversed(tuple(messages)):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        payload = _structured_payload(message.get("content"))
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("phase") or "").strip() != _AUTHORED_PHASE:
            continue
        if str(payload.get("workspace_project_root") or "").strip() != ".":
            continue
        module = payload.get("module")
        if not isinstance(module, Mapping) or "authored_plan" not in module:
            continue
        if _module_id(module) != task_id:
            continue
        return AuthoredMutationScope(task_id=task_id)
    return None


def _canonical_scoped_path(value: Any, *, prefixes: Sequence[str] = _ALLOWED_PREFIXES) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw or raw.startswith("/") or ":" in raw:
        return ""
    parts = PurePosixPath(raw).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    path = PurePosixPath(raw).as_posix()
    if not any(path.startswith(prefix) and len(path) > len(prefix) for prefix in prefixes):
        return ""
    return path


def _scoped_mutation_error(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    path_keys: Sequence[str],
    scope: AuthoredMutationScope,
) -> str | None:
    if tool_name != "apply_source_edit":
        return None

    raw_path = ""
    for key in path_keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            raw_path = value
            break
    supplied = _canonical_scoped_path(raw_path, prefixes=scope.allowed_prefixes)
    if not supplied:
        return (
            "PATH_OUTSIDE_WRITABLE_SET: authored design writes are restricted to "
            f"{list(scope.allowed_prefixes)!r}; model-supplied path {raw_path!r} is not authorized."
        )

    operation = str(arguments.get("operation") or "").strip().casefold()
    if operation in _DELETE_OPERATIONS or operation.startswith(("delete_", "remove_")):
        return (
            "WRITE_SCOPE_DELETE_FORBIDDEN: authored design generation may create or edit "
            f"bounded project files but may not delete {supplied!r}."
        )
    return None


def _forced_tool_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    function = tool_choice.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "").strip()
    return str(tool_choice.get("name") or "").strip()


def _scope_instruction(scope: AuthoredMutationScope) -> str:
    roots = ", ".join(scope.allowed_prefixes)
    return (
        "Host-authored design write scope is active for this internal task. "
        f"You may create or edit files only below: {roots}. "
        "Do not delete files and do not write build configuration, .mmm state, or any path "
        "outside those roots. During an ACT turn, emit all independent apply_source_edit "
        "calls needed for the current coherent implementation wave in the same response; "
        "the host executes write calls sequentially before verification."
    )


def install(loop_module: Any | None = None) -> None:
    """Install after runtime finalization so this is the last mutation-scope policy."""

    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if getattr(loop_module, _MARKER, False):
        return

    from . import direct_task_mutation_authority_contract as direct_authority

    original_generate = loop_module.generate_with_tools
    original_target_error = loop_module._mutation_target_error
    original_turn = loop_module._generate_turn_with_context_recovery

    @wraps(original_generate)
    def generate_with_authored_scope(
        router: Any,
        *,
        config: Any,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        authority = direct_authority._CURRENT_AUTHORITY.get()
        scope = _resolve_authored_scope(
            request.messages,
            authority=authority,
            stage=stage,
            role=role,
        )
        if scope is None:
            return original_generate(
                router,
                config=config,
                adapter=adapter,
                request=request,
                runtime=runtime,
                stage=stage,
                role=role,
            )

        instruction = {"role": "developer", "content": _scope_instruction(scope)}
        request = replace(request, messages=tuple(request.messages) + (instruction,))
        token = _ACTIVE_SCOPE.set(scope)
        try:
            return original_generate(
                router,
                config=config,
                adapter=adapter,
                request=request,
                runtime=runtime,
                stage=stage,
                role=role,
            )
        finally:
            _ACTIVE_SCOPE.reset(token)

    @wraps(original_target_error)
    def mutation_target_error(tool_name: str, arguments: Mapping[str, Any], context: Any):
        scope = _ACTIVE_SCOPE.get()
        if scope is None or tool_name != "apply_source_edit":
            return original_target_error(tool_name, arguments, context)
        return _scoped_mutation_error(
            tool_name,
            arguments,
            path_keys=tuple(loop_module._SOURCE_EDIT_PATH_KEYS),
            scope=scope,
        )

    @wraps(original_turn)
    def generate_turn_with_authored_parallelism(*args: Any, **kwargs: Any):
        scope = _ACTIVE_SCOPE.get()
        if scope is not None and _forced_tool_name(kwargs.get("tool_choice")) == "apply_source_edit":
            kwargs["parallel_tool_calls"] = True
            request = kwargs.get("request")
            if request is not None and hasattr(request, "parallel_tool_calls"):
                kwargs["request"] = replace(request, parallel_tool_calls=True)
        return original_turn(*args, **kwargs)

    loop_module.generate_with_tools = generate_with_authored_scope
    loop_module._mutation_target_error = mutation_target_error
    loop_module._generate_turn_with_context_recovery = generate_turn_with_authored_parallelism
    setattr(loop_module, _MARKER, True)


__all__ = [
    "AuthoredMutationScope",
    "_canonical_scoped_path",
    "_resolve_authored_scope",
    "_scoped_mutation_error",
    "install",
]
