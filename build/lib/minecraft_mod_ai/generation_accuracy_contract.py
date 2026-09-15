from __future__ import annotations

"""Host-owned accuracy gate for atomic small-model source generation.

The small coder is allowed to decide implementation semantics inside an approved atomic
obligation, but its edits are not trusted merely because the model returned a success
summary.  This contract verifies the staged workspace after every atomic coder turn,
feeds exact deterministic diagnostics back for one local repair turn, and fails closed
when the repair still violates the approved target or produces structurally invalid
source/resource output.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

from .artifact_validators.java import JavaValidationError, validate_java_fragment

_INNER_MARKER = "_mmm_generation_accuracy_inner"
_OUTER_MARKER = "_mmm_generation_accuracy_outer"
_ATOMIC_SCHEMA = "mmm/atomic-coder-step"
_SOURCE_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
)
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\bTODO\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"\bUnsupportedOperationException\b"),
    re.compile(r"\bNotImplementedError\b"),
)
_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


class GenerationAccuracyError(RuntimeError):
    """An atomic coder turn could not satisfy deterministic generation checks."""


def _normalize_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise GenerationAccuracyError(f"GENERATION_TARGET_PATH_INVALID: {raw!r}")
    return path.as_posix()


def _implementation_request(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(tuple(messages)):
        if str(message.get("role") or "").strip().casefold() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            request = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(request, dict) and request.get("phase") == "implement_module":
            return request
    return None


def _atomic_contract(request: Mapping[str, Any]) -> dict[str, Any] | None:
    module = request.get("module")
    if not isinstance(module, Mapping):
        return None
    evidence_task = module.get("evidence_task")
    if not isinstance(evidence_task, Mapping):
        return None
    contract = evidence_task.get("coder_execution_contract")
    if not isinstance(contract, Mapping) or contract.get("schema_version") != _ATOMIC_SCHEMA:
        return None
    return dict(contract)


def _approved_paths(contract: Mapping[str, Any]) -> tuple[str, ...]:
    protected = contract.get("protected_boundaries")
    raw = protected.get("writable_paths") if isinstance(protected, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise GenerationAccuracyError("GENERATION_WRITABLE_PATHS_MISSING")
    paths = tuple(dict.fromkeys(_normalize_path(item) for item in raw))
    if not paths:
        raise GenerationAccuracyError("GENERATION_WRITABLE_PATHS_EMPTY")
    return paths


def _target_records(contract: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = contract.get("targets")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise GenerationAccuracyError("GENERATION_TARGETS_MISSING")
    targets: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        path = _normalize_path(item.get("path"))
        locator = str(item.get("locator") or "").strip()
        if not locator:
            raise GenerationAccuracyError(f"GENERATION_TARGET_LOCATOR_MISSING: {path}")
        targets.append(
            {
                "path": path,
                "locator": locator,
                "symbol": str(item.get("symbol") or "").strip(),
                "operation": str(item.get("operation") or "").strip(),
            }
        )
    if not targets:
        raise GenerationAccuracyError("GENERATION_TARGETS_EMPTY")
    return tuple(targets)


def _source_snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for prefix in _SOURCE_PREFIXES:
        base = root / prefix.rstrip("/")
        if not base.exists():
            continue
        if base.is_symlink() or not base.is_dir():
            raise GenerationAccuracyError(f"GENERATION_SOURCE_ROOT_INVALID: {prefix}")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise GenerationAccuracyError(
                    f"GENERATION_SOURCE_SYMLINK_FORBIDDEN: {path.relative_to(root).as_posix()}"
                )
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _target_text(root: Path, path: str) -> str:
    candidate = root / path
    if not candidate.exists() or candidate.is_symlink() or not candidate.is_file():
        return ""
    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _target_texts(root: Path, paths: Sequence[str]) -> dict[str, str]:
    return {path: _target_text(root, path) for path in paths}


def _changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
    )


def _placeholder_counts(text: str) -> tuple[int, ...]:
    return tuple(len(pattern.findall(text)) for pattern in _PLACEHOLDER_PATTERNS)


def _validate_target_file(
    root: Path,
    target: Mapping[str, str],
    *,
    before_text: str,
    changed: bool,
) -> list[dict[str, Any]]:
    path = target["path"]
    candidate = root / path
    diagnostics: list[dict[str, Any]] = []
    if not candidate.exists() or candidate.is_symlink() or not candidate.is_file():
        diagnostics.append({"code": "TARGET_MISSING", "path": path})
        return diagnostics

    text = _target_text(root, path)
    suffix = candidate.suffix.casefold()
    if suffix in {".java", ".kt"}:
        if not text:
            diagnostics.append({"code": "SOURCE_EMPTY_OR_NON_UTF8", "path": path})
        elif suffix == ".java":
            try:
                validate_java_fragment(text, anchor=target["locator"])
            except JavaValidationError as exc:
                diagnostics.append(
                    {"code": "JAVA_STRUCTURE_INVALID", "path": path, "detail": str(exc)}
                )
    elif suffix == ".json":
        if not text:
            diagnostics.append({"code": "JSON_EMPTY_OR_NON_UTF8", "path": path})
        else:
            try:
                json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                diagnostics.append(
                    {"code": "JSON_INVALID", "path": path, "detail": str(exc)[:512]}
                )

    symbol = target.get("symbol", "")
    if symbol and _IDENTIFIER.fullmatch(symbol) and text:
        if re.search(rf"(?<![A-Za-z0-9_$]){re.escape(symbol)}(?![A-Za-z0-9_$])", text) is None:
            diagnostics.append(
                {"code": "TARGET_SYMBOL_MISSING", "path": path, "symbol": symbol}
            )

    if changed and text:
        before_counts = _placeholder_counts(before_text)
        after_counts = _placeholder_counts(text)
        introduced = [
            pattern.pattern
            for pattern, old, new in zip(
                _PLACEHOLDER_PATTERNS, before_counts, after_counts, strict=True
            )
            if new > old
        ]
        if introduced:
            diagnostics.append(
                {
                    "code": "PLACEHOLDER_INTRODUCED",
                    "path": path,
                    "patterns": introduced,
                }
            )
    return diagnostics


def _validate_atomic_workspace(
    root: Path,
    contract: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    before_snapshot: Mapping[str, str],
    before_texts: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    allowed = frozenset(_approved_paths(contract))
    targets = _target_records(contract)
    target_paths = frozenset(target["path"] for target in targets)
    if not target_paths.issubset(allowed):
        return (
            {
                "code": "TARGET_OUTSIDE_WRITABLE_BOUNDARY",
                "paths": sorted(target_paths - allowed),
            },
        )

    after_snapshot = _source_snapshot(root)
    changed = _changed_paths(before_snapshot, after_snapshot)
    diagnostics: list[dict[str, Any]] = []
    outside = sorted(path for path in changed if path not in allowed)
    if outside:
        diagnostics.append({"code": "OUTSIDE_TARGET_MUTATION", "paths": outside})

    deleted = sorted(path for path in changed if path in before_snapshot and path not in after_snapshot)
    if deleted:
        diagnostics.append({"code": "SOURCE_DELETION_FORBIDDEN", "paths": deleted})

    checkpoint = request.get("checkpoint")
    resumed = bool(checkpoint.get("resumed")) if isinstance(checkpoint, Mapping) else False
    changed_targets = tuple(path for path in changed if path in target_paths)
    if not changed_targets and not resumed:
        diagnostics.append(
            {
                "code": "ATOMIC_OBLIGATION_NO_TARGET_MUTATION",
                "targets": sorted(target_paths),
            }
        )

    for target in targets:
        diagnostics.extend(
            _validate_target_file(
                root,
                target,
                before_text=before_texts.get(target["path"], ""),
                changed=target["path"] in changed_targets,
            )
        )
    return tuple(diagnostics)


def _repair_messages(
    messages: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    diagnostic_json = json.dumps(
        list(diagnostics),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    repair = {
        "role": "system",
        "content": (
            "HOST GENERATION-ACCURACY GATE FAILED. Repair only the current atomic obligation "
            "in the already-mutated staged workspace. Do not restart sibling obligations, widen "
            "scope, guess missing deterministic facts, or merely claim success. Resolve every "
            "host diagnostic with the source-edit and bounded evidence tools, then return the "
            "normal coder summary. Exact diagnostics: " + diagnostic_json
        ),
    }
    materialized = tuple(dict(message) for message in messages)
    if materialized and str(materialized[-1].get("role") or "").casefold() == "user":
        return (*materialized[:-1], repair, materialized[-1])
    return (*materialized, repair)


def _raise_accuracy_failure(diagnostics: Sequence[Mapping[str, Any]]) -> None:
    rendered = json.dumps(
        list(diagnostics),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    raise GenerationAccuracyError("GENERATION_ACCURACY_REPAIR_FAILED: " + rendered)


def install_inner(model_router_module: Any) -> None:
    """Install inside the atomic slicer so every obligation is verified independently."""

    Router = model_router_module.ModelRouter
    if getattr(Router.generate_text, _INNER_MARKER, False):
        return
    current = Router.generate_text

    @wraps(current)
    def generate_text(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if str(role).strip().casefold() not in {"coder", "coder_safe"}:
            return current(self, role, messages, *args, **kwargs)
        request = _implementation_request(messages)
        contract = _atomic_contract(request) if request is not None else None
        if request is None or contract is None:
            return current(self, role, messages, *args, **kwargs)

        workspace = getattr(self, "_agent_workspace_root", None)
        if workspace is None:
            raise GenerationAccuracyError("GENERATION_ACCURACY_WORKSPACE_UNBOUND")
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise GenerationAccuracyError("GENERATION_ACCURACY_WORKSPACE_INVALID")

        approved = _approved_paths(contract)
        before_snapshot = _source_snapshot(root)
        before_texts = _target_texts(root, approved)
        result = current(self, role, messages, *args, **kwargs)
        diagnostics = _validate_atomic_workspace(
            root,
            contract,
            request,
            before_snapshot=before_snapshot,
            before_texts=before_texts,
        )
        if not diagnostics:
            return result
        if any(
            item.get("code") in {"OUTSIDE_TARGET_MUTATION", "SOURCE_DELETION_FORBIDDEN"}
            for item in diagnostics
        ):
            _raise_accuracy_failure(diagnostics)

        repair_result = current(
            self,
            role,
            _repair_messages(messages, diagnostics),
            *args,
            **kwargs,
        )
        remaining = _validate_atomic_workspace(
            root,
            contract,
            request,
            before_snapshot=before_snapshot,
            before_texts=before_texts,
        )
        if remaining:
            _raise_accuracy_failure(remaining)
        return repair_result

    setattr(generate_text, _INNER_MARKER, True)
    Router.generate_text = generate_text


def install_outer(model_router_module: Any) -> None:
    """Normalize the complete coder result after the atomic slicer has aggregated turns."""

    Router = model_router_module.ModelRouter
    if getattr(Router.generate_text, _OUTER_MARKER, False):
        return
    current = Router.generate_text

    @wraps(current)
    def generate_text(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = current(self, role, messages, *args, **kwargs)
        if str(role).strip().casefold() not in {"coder", "coder_safe"}:
            return result
        request = _implementation_request(messages)
        if request is None:
            return result
        text = str(result or "").strip()
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping) and isinstance(payload.get("summary"), str):
            return text
        return json.dumps({"summary": text}, ensure_ascii=False, separators=(",", ":"))

    setattr(generate_text, _OUTER_MARKER, True)
    Router.generate_text = generate_text


def assert_inner_installed(model_router_module: Any) -> None:
    current = model_router_module.ModelRouter.generate_text
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, _INNER_MARKER, False):
            return
        current = getattr(current, "__wrapped__", None)
    raise RuntimeError("Generation accuracy inner gate is not installed.")


def assert_outer_installed(model_router_module: Any) -> None:
    if not getattr(model_router_module.ModelRouter.generate_text, _OUTER_MARKER, False):
        raise RuntimeError("Generation accuracy result normalizer is not installed.")


__all__ = [
    "GenerationAccuracyError",
    "assert_inner_installed",
    "assert_outer_installed",
    "install_inner",
    "install_outer",
]
