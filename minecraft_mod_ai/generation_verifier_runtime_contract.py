from __future__ import annotations

"""Runtime targeting and lifecycle contract for host-owned generation verification."""

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

_TARGET_MARKER = "_mmm_generation_verifier_targeted_turn"
_CLOSE_MARKER = "_mmm_generation_verifier_close"
_MUTATION_TOOLS = frozenset({"apply_source_edit", "apply_source_patch"})
_JAVA_SUFFIX = ".java"


def _safe_source_path(raw: Any) -> str:
    value = str(raw or "").replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or not value.endswith(_JAVA_SUFFIX)
    ):
        return ""
    return value


def _function_arguments(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _paths_from_arguments(name: str, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    if name == "apply_source_edit":
        for key in ("path", "relative_path", "target_path", "file_path"):
            path = _safe_source_path(arguments.get(key))
            if path:
                paths.append(path)
                break
    elif name == "apply_source_patch":
        operations = arguments.get("operations")
        if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes, bytearray)):
            for operation in operations:
                if not isinstance(operation, Mapping):
                    continue
                for key in ("path", "relative_path", "target_path", "file_path"):
                    path = _safe_source_path(operation.get(key))
                    if path:
                        paths.append(path)
                        break
    return tuple(dict.fromkeys(paths))


def latest_mutated_source_files(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return Java files from the most recent assistant mutation turn only."""

    for message in reversed(tuple(messages)):
        if str(message.get("role") or "") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes, bytearray)):
            continue
        found: list[str] = []
        saw_mutation = False
        for raw_call in tool_calls:
            if not isinstance(raw_call, Mapping):
                continue
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                continue
            name = str(function.get("name") or "").strip()
            if name not in _MUTATION_TOOLS:
                continue
            saw_mutation = True
            arguments = _function_arguments(function.get("arguments"))
            found.extend(_paths_from_arguments(name, arguments))
        if saw_mutation:
            return tuple(dict.fromkeys(found))
    return ()


def install(*, agent_tool_runtime_module: Any, verifier_module: Any) -> None:
    """Target verifier work to changed Java files and own JDT service cleanup."""

    current_synth = verifier_module.synthesized_verifier_turn
    if not getattr(current_synth, _TARGET_MARKER, False):

        @wraps(current_synth)
        def synthesized_verifier_turn(messages: list[dict[str, Any]]) -> Any:
            from .model_adapters import GenerationResponse, ToolCall

            relative_files = latest_mutated_source_files(messages)
            arguments: dict[str, Any] = {
                "timeout_seconds": verifier_module.host_jdt_idle_timeout_seconds(),
            }
            if relative_files:
                arguments["relative_files"] = list(relative_files)
            raw_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            identity_material = f"{len(messages)}:{raw_arguments}".encode("utf-8")
            call_id = "host_verify_" + hashlib.sha256(identity_material).hexdigest()[:16]
            return GenerationResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id=call_id,
                        name="java_diagnostics",
                        arguments=arguments,
                        raw_arguments=raw_arguments,
                    ),
                ),
            )

        setattr(synthesized_verifier_turn, _TARGET_MARKER, True)
        synthesized_verifier_turn.__wrapped__ = current_synth
        verifier_module.synthesized_verifier_turn = synthesized_verifier_turn

    runtime_cls = agent_tool_runtime_module.AgentToolRuntime
    current_close = runtime_cls.close
    if not getattr(current_close, _CLOSE_MARKER, False):

        @wraps(current_close)
        def close_with_generation_verifier(self: Any) -> None:
            service = getattr(self, "_mmm_generation_java_service", None)
            if service is not None:
                try:
                    close = getattr(service, "close", None)
                    if callable(close):
                        close()
                finally:
                    try:
                        delattr(self, "_mmm_generation_java_service")
                    except AttributeError:
                        pass
            current_close(self)

        setattr(close_with_generation_verifier, _CLOSE_MARKER, True)
        close_with_generation_verifier.__wrapped__ = current_close
        runtime_cls.close = close_with_generation_verifier


__all__ = ["install", "latest_mutated_source_files"]
