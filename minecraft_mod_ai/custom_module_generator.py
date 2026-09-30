from __future__ import annotations

"""Direct, whole-file custom-module generation.

Authored designs first compile to a responsibility/dependency implementation graph.
Each admitted source unit owns an exact host-selected file. The coder returns one complete
first-pass source candidate and Gradle is a pass/fail verification gate, never a model-repair
loop. Output exhaustion returns to graph decomposition and cannot retry the exhausted task.
No patch transport is involved.
"""

import hashlib
import inspect
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from .complete_spec import ProductionModule
from .custom_module_errors import AtomicJavaDecisionError, CustomModuleGenerationError
from .generation_implementation_grounding import (
    build_generation_implementation_grounding,
    render_generation_implementation_authority_prompt,
)
from .host_grounding import custom_module_path_allowed
from .implementation_ir import OutputBudgetExhausted
from .llama_finish_reason_contract import OUTPUT_EXHAUSTED, completion_boundary_error
from .model_router import ModelRouter
from .platform_catalog import adapter_for_target, adapter_from_project
from .project_write_lock import project_path_write_locks, project_write_lock
from .runner import GradleRunner
from .scale_policy import ScalePolicy
from .target_contract import TargetContractError, validate_target_coordinates

_LOCATOR = re.compile(r"^(?P<path>[^#]+\.java)#(?P<symbol>[A-Za-z_$][A-Za-z0-9_$]*)$")
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*;\s*$")
_PUBLIC_TYPE = re.compile(
    r"\bpublic\s+final\s+class\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b"
)
_INITIALIZE = re.compile(
    r"\bpublic\s+static\s+void\s+initialize\s*\(\s*\)\s*(?:throws\s+[^{]+)?\{"
)
_SIDE_ONLY = re.compile(r"@Environment\s*\(\s*EnvType\.(?:CLIENT|SERVER)\s*\)")
_FORBIDDEN_ENTRYPOINT = re.compile(
    r"\b(?:implements\s+)?(?:ModInitializer|ClientModInitializer)\b"
)
_BODY_MARKER = "MMM_AUTHORED_FEATURE_BODY"
_ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING = 4096

def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_project_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or raw in {".", ".."}
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise CustomModuleGenerationError(
            f"Custom module target must be a safe project-relative path: {value!r}"
        )
    return candidate.as_posix()


def _task_local_module_contract(module: ProductionModule) -> dict[str, Any]:
    config = module.config if isinstance(module.config, dict) else {}
    task = config.get("evidence_task")
    if not isinstance(task, Mapping):
        raise CustomModuleGenerationError(
            "TASK_LOCAL_CONTRACT_REQUIRED: "
            f"module {module.module_id!r} is missing structured config.evidence_task"
        )
    return dict(task)


def _bounded_execution_feedback(value: Any) -> dict[str, Any] | None:
    """Normalize and deduplicate structured diagnostics without arbitrary truncation."""
    if not isinstance(value, Mapping):
        return None
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(
        diagnostics, (str, bytes, bytearray)
    ):
        return None
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in diagnostics:
        if not isinstance(raw, Mapping):
            continue
        row = {
            "path": str(raw.get("path") or "").strip(),
            "code": str(raw.get("code") or "").strip(),
            "message": " ".join(str(raw.get("message") or "").split()),
        }
        key = (row["path"], row["code"], row["message"])
        if any(row.values()) and key not in seen:
            seen.add(key)
            rows.append(row)
    return {"diagnostics": rows} if rows else None


def _owned_main_java_candidate(anchor: Any) -> tuple[str, str] | None:
    if not isinstance(anchor, Mapping):
        return None
    match = _LOCATOR.fullmatch(str(anchor.get("locator") or "").strip())
    if match is None:
        return None
    path = _normalize_project_path(match.group("path"))
    if not path.startswith("src/main/java/"):
        return None
    return path, match.group("symbol")


def _exact_target(module: ProductionModule) -> tuple[str, str, dict[str, Any]]:
    task = _task_local_module_contract(module)
    anchors = task.get("owned_anchors")
    candidates: list[tuple[str, str]] = []
    if isinstance(anchors, Sequence) and not isinstance(
        anchors, (str, bytes, bytearray)
    ):
        for anchor in anchors:
            candidate = _owned_main_java_candidate(anchor)
            if candidate is not None:
                candidates.append(candidate)
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise CustomModuleGenerationError(
            "DIRECT_CODER_EXACT_TARGET_REQUIRED: each custom Java task must own "
            "exactly one host-selected Java file and top-level symbol."
        )
    path, symbol = unique[0]
    if not path.startswith("src/main/java/"):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_JAVA_TARGET_REQUIRED: {path}"
        )
    return path, symbol, task


def _resolve_generation_adapter(
    root: Path,
    *,
    minecraft_version: str | None,
    loader: str | None,
):
    requested_version = str(minecraft_version or "").strip()
    requested_loader = str(loader or "").strip()
    if requested_version and requested_loader:
        try:
            return adapter_for_target(requested_version, requested_loader)
        except ValueError as exc:
            raise CustomModuleGenerationError(str(exc)) from exc
    try:
        return adapter_from_project(root)
    except ValueError as exc:
        raise CustomModuleGenerationError(
            "Custom generation requires one unambiguous executable platform target."
        ) from exc


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise CustomModuleGenerationError(
            f"Custom module target escapes the project root: {relative}"
        ) from exc
    if target.is_symlink():
        raise CustomModuleGenerationError(
            f"Custom module target may not be a symbolic link: {relative}"
        )
    return target


def _host_reserved(anchor: Mapping[str, Any]) -> bool:
    return str(anchor.get("status") or "").strip() in {
        "host_reserved",
        "host_owned",
        "new",
    }


def _target_anchor(task: Mapping[str, Any], relative: str, symbol: str) -> Mapping[str, Any] | None:
    anchors = task.get("owned_anchors")
    if not isinstance(anchors, Sequence) or isinstance(
        anchors, (str, bytes, bytearray)
    ):
        return None
    locator = f"{relative}#{symbol}"
    for anchor in anchors:
        if isinstance(anchor, Mapping) and str(anchor.get("locator") or "").strip() == locator:
            return anchor
    return None


def _package_from_java_target(relative: str) -> str:
    prefix = "src/main/java/"
    if not relative.startswith(prefix) or not relative.endswith(".java"):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_JAVA_TARGET_REQUIRED: {relative}"
        )
    body = relative[len(prefix):]
    parent = PurePosixPath(body).parent
    if str(parent) in {"", "."}:
        return ""
    return ".".join(parent.parts)


def _materialize_host_scaffold(
    target: Path,
    *,
    relative: str,
    symbol: str,
    task: Mapping[str, Any],
) -> str:
    anchor = _target_anchor(task, relative, symbol)
    if anchor is None or not _host_reserved(anchor):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_HOST_SCAFFOLD_MISSING: {relative}"
        )
    if not custom_module_path_allowed(relative):
        raise CustomModuleGenerationError(
            f"DIRECT_CODER_TARGET_OUTSIDE_WRITE_SCOPE: {relative}"
        )
    package_name = _package_from_java_target(relative)
    package_line = f"package {package_name};\n\n" if package_name else ""
    source = (
        package_line
        + f"public final class {symbol} {{\n"
        + f"    private {symbol}() {{}}\n\n"
        + "    // MMM_AUTHORED_FEATURE_BODY\n"
        + "}\n"
    )
    _atomic_write(target, source)
    return source


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content.encode("utf-8") if isinstance(content, str) else content
    temporary = path.with_name(
        f".{path.name}.mmm-{os.getpid()}-"
        f"{hashlib.sha256(raw).hexdigest()}.tmp"
    )
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_context(root: Path, target: Path, *, relevance_text: str) -> str:
    """Return only project sources explicitly referenced by the active task/scaffold."""
    java_root = root / "src/main/java"
    if not java_root.is_dir():
        return ""

    evidence = relevance_text
    if target.is_file() and not target.is_symlink():
        try:
            evidence += "\n" + target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass

    # Tokenize the evidence once. The previous implementation ran one regex
    # search over the full task/scaffold text for every Java file, making context
    # discovery O(project_files * evidence_size) for every custom generation.
    referenced_symbols = set(
        re.findall(
            r"(?<![A-Za-z0-9_$])[A-Za-z_$][A-Za-z0-9_$]*(?![A-Za-z0-9_$])",
            evidence,
        )
    )
    target_resolved = target.resolve()
    candidates = [
        path
        for path in java_root.rglob("*.java")
        if path.is_file()
        and not path.is_symlink()
        and path.resolve() != target_resolved
    ]
    selected = [
        path
        for path in candidates
        if path.stem in referenced_symbols
    ]
    selected.sort(key=lambda path: path.relative_to(root).as_posix())

    rendered: list[str] = []
    for path in selected:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        relative = path.relative_to(root).as_posix()
        rendered.append(f"\n--- {relative} ---\n{text}\n")
    return "".join(rendered)


def _source_invariant_errors(
    source: str,
    *,
    symbol: str,
    expected_package: str,
    require_initialize: bool,
) -> tuple[str, ...]:
    errors: list[str] = []
    if "```" in source:
        errors.append("source contains Markdown code fences")
    package_match = _PACKAGE.search(source)
    if expected_package and (
        package_match is None or package_match.group(1) != expected_package
    ):
        errors.append(f"package must remain exactly {expected_package}")
    public = _PUBLIC_TYPE.findall(source)
    if public != [symbol]:
        errors.append(
            f"top-level contract must be exactly `public final class {symbol}`"
        )
    if require_initialize and not _INITIALIZE.search(source):
        errors.append("required `public static void initialize()` is missing")
    if _SIDE_ONLY.search(source):
        errors.append(
            "common authored source may not carry @Environment(CLIENT/SERVER)"
        )
    if _FORBIDDEN_ENTRYPOINT.search(source):
        errors.append(
            "feature source may not implement or declare a Fabric mod entrypoint"
        )
    if _BODY_MARKER in source:
        errors.append("host implementation marker was not replaced")
    return tuple(errors)


def _plain_coder_output(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        raise CustomModuleGenerationError(
            "DIRECT_CODER_EMPTY_RESPONSE: coder returned no source text."
        )
    return raw + "\n"

def _supports_kwarg(callable_value: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _dependency_source_context(root: Path, raw: str) -> str:
    """Materialize exact host-selected dependency sources; the model chooses no lookup."""
    try:
        rows = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        rows = []
    if not isinstance(rows, list):
        return raw
    rendered: list[str] = []
    budget = 64 * 1024
    used = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        path = str(row.get("path") or "").replace("\\", "/").strip()
        if not path:
            continue
        try:
            target = _safe_target(root, _normalize_project_path(path))
            source = target.read_text(encoding="utf-8")
        except (CustomModuleGenerationError, OSError, UnicodeError):
            source = ""
        block = json.dumps(
            {
                "symbol": row.get("symbol"),
                "path": path,
                "responsibility": row.get("responsibility"),
                "public_api": row.get("public_api") or [],
                "source": source,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        encoded = block.encode("utf-8")
        if used + len(encoded) > budget:
            break
        rendered.append(block)
        used += len(encoded)
    return "\n".join(rendered) if rendered else raw


def _direct_host_grounding(
    *,
    adapter: Any,
    task: Mapping[str, Any],
    ir_contract: Mapping[str, Any] | None,
    dependency_context: str,
    typed_grounding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build complete host evidence while preserving the established typed contract."""
    version_facts: dict[str, Any] = {}
    try:
        from .host_version_catalog import host_target

        target = host_target(adapter.minecraft_version)
        resolved = target.version_context
        snapshot = resolved.to_dict()["host_facts"]
        version_facts = {
            "context_id": resolved.context_id,
            "capabilities": {
                str(key): bool(value)
                for key, value in snapshot["capabilities"].items()
                if bool(value)
            },
            "api_symbols": dict(snapshot["api_symbols"]),
            "dependency_coordinates": dict(snapshot["dependency_coordinates"]),
        }
    except Exception as exc:  # noqa: BLE001 - optional grounding failures become explicit evidence
        version_facts = {
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    section_role = ""
    if isinstance(ir_contract, Mapping):
        role_by_symbol = {
            "AuthoredStateModel": "state_model",
            "AuthoredBehaviorContract": "behavior_contract",
            "AuthoredAlgorithm": "algorithm",
            "AuthoredAuthorityNetwork": "authority_and_network",
            "AuthoredPersistence": "persistence",
            "AuthoredResourcesUi": "resources_and_ui",
            "AuthoredFailureLimits": "failure_and_limits",
            "AuthoredIntegration": "integration",
        }
        section_role = role_by_symbol.get(str(ir_contract.get("symbol") or ""), "")

    direct_context = {
        "schema_version": "mmm/direct-coder-host-context-v1",
        "phase": "implement_module",
        "workspace_project_root": ".",
        "section_role": section_role,
        "platform": {
            "minecraft_version": str(adapter.minecraft_version),
            "loader": str(adapter.loader),
            "java_version": int(adapter.java_version),
            "mappings": str(getattr(adapter, "yarn_mappings", "") or ""),
        },
        "task": dict(task),
        "implementation_ir_node": dict(ir_contract) if isinstance(ir_contract, Mapping) else None,
        "dependency_context": dependency_context,
        "host_version_facts": version_facts,
    }
    legacy = dict(typed_grounding) if isinstance(typed_grounding, Mapping) else {}
    prior_policy = legacy.get("policy")
    merged_policy = dict(prior_policy) if isinstance(prior_policy, Mapping) else {}
    merged_policy.update({
        "resolved_before_first_coder_decode": True,
        "baseline_grounding_owned_by_host": True,
        "writes_still_require_approved_pipeline": True,
        "model_tool_choice_required": False,
        "fabric_lifecycle_owned_by_host": True,
        "invent_unlisted_platform_api": False,
    })
    # Keep artifact_kind/facts and every established typed-grounding key at the top
    # level so deterministic renderers and model prompts consume one stable contract.
    return {
        **legacy,
        "direct_host_context": direct_context,
        "policy": merged_policy,
    }

_ATOMIC_JAVA_REGION_TOOL = "emit_java_structure"
_JAVA_IDENTIFIER_PATTERN = r"^[A-Za-z_$][A-Za-z0-9_$]*$"
_ATOMIC_METHOD_NAME_PATTERN = r"^(?:<init>|[A-Za-z_$][A-Za-z0-9_$]*)$"
_ATOMIC_PARAMETER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Java type. Use a primitive/java.lang type, a type declared in this same "
                "structured call, a supplied sibling/dependency type, or a fully-qualified "
                "external type. Common JDK collection names may be simple names because the "
                "host qualifies them."
            ),
        },
        "name": {
            "type": "string",
            "pattern": _JAVA_IDENTIFIER_PATTERN,
            "description": (
                "Semantic identifier. Java reserved words are accepted here because the host "
                "canonicalizes them consistently before rendering."
            ),
        },
    },
    "required": ["type", "name"],
    "additionalProperties": True,
}
_ATOMIC_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Java field type only. Concern-owned domain types may be declared in the same "
                "records/enums/classes payload. Declaration modifiers are host-owned."
            ),
        },
        "name": {"type": "string", "pattern": _JAVA_IDENTIFIER_PATTERN},
        "initializer": {
            "type": "string",
            "description": (
                "Initializer expression only, without a trailing semicolon. "
                "Never instantiate an interface or abstract JDK collection directly; "
                "use a concrete implementation or a valid factory."
            ),
        },
    },
    "required": ["type", "name"],
    "additionalProperties": True,
}
_ATOMIC_METHOD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "return_type": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Java return type only. Declaration modifiers are host-owned."
            ),
        },
        "name": {"type": "string", "pattern": _ATOMIC_METHOD_NAME_PATTERN},
        "parameters": {"type": "array", "items": _ATOMIC_PARAMETER_SCHEMA},
        "throws": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "body": {
            "type": "array",
            "items": {
                "type": "string",
                "description": (
                    "One Java statement or one complete control-flow block inside this method."
                ),
            },
        },
    },
    "required": ["return_type", "name"],
    "additionalProperties": True,
}
_ATOMIC_OUTER_METHOD_SCHEMA: dict[str, Any] = deepcopy(_ATOMIC_METHOD_SCHEMA)
_ATOMIC_OUTER_METHOD_SCHEMA["properties"]["name"] = {
    "type": "string",
    "pattern": _JAVA_IDENTIFIER_PATTERN,
    "description": (
        "Ordinary method name in the existing host-selected outer class. "
        "<init> is forbidden here because outer-class construction is host-owned."
    ),
}

_ATOMIC_CONSTRUCTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "parameters": {"type": "array", "items": _ATOMIC_PARAMETER_SCHEMA},
        "throws": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "body": {"type": "array", "items": {"type": "string"}},
    },
    "required": [],
    "additionalProperties": True,
}
_ATOMIC_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": _JAVA_IDENTIFIER_PATTERN},
        "components": {"type": "array", "items": _ATOMIC_PARAMETER_SCHEMA},
        "constructors": {"type": "array", "items": _ATOMIC_CONSTRUCTOR_SCHEMA},
        "methods": {"type": "array", "items": _ATOMIC_METHOD_SCHEMA},
    },
    "required": ["name"],
    "additionalProperties": True,
}
_ATOMIC_ENUM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": _JAVA_IDENTIFIER_PATTERN},
        "constants": {
            "type": "array",
            "items": {"type": "string", "pattern": _JAVA_IDENTIFIER_PATTERN},
            "uniqueItems": True,
        },
    },
    "required": ["name", "constants"],
    "additionalProperties": True,
}
_ATOMIC_CLASS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": _JAVA_IDENTIFIER_PATTERN},
        "fields": {"type": "array", "items": _ATOMIC_FIELD_SCHEMA},
        "constructors": {"type": "array", "items": _ATOMIC_CONSTRUCTOR_SCHEMA},
        "methods": {"type": "array", "items": _ATOMIC_METHOD_SCHEMA},
    },
    "required": ["name"],
    "additionalProperties": True,
}
_ATOMIC_MEMBERS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": _ATOMIC_RECORD_SCHEMA},
        "enums": {"type": "array", "items": _ATOMIC_ENUM_SCHEMA},
        "classes": {"type": "array", "items": _ATOMIC_CLASS_SCHEMA},
        "fields": {"type": "array", "items": _ATOMIC_FIELD_SCHEMA},
        "methods": {"type": "array", "items": _ATOMIC_OUTER_METHOD_SCHEMA},
    },
    "required": [],
    "additionalProperties": True,
}
_ATOMIC_LOGIC_MEMBERS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fields": {"type": "array", "items": _ATOMIC_FIELD_SCHEMA},
        "methods": {"type": "array", "items": _ATOMIC_OUTER_METHOD_SCHEMA},
    },
    "required": [],
    "additionalProperties": False,
}
_ATOMIC_DECLARATION_MEMBERS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": _ATOMIC_RECORD_SCHEMA},
        "enums": {"type": "array", "items": _ATOMIC_ENUM_SCHEMA},
        "classes": {"type": "array", "items": _ATOMIC_CLASS_SCHEMA},
        "fields": {"type": "array", "items": _ATOMIC_FIELD_SCHEMA},
    },
    "required": [],
    "additionalProperties": False,
}
# Static initializer blocks are host-owned lifecycle structure. Concern models
# may emit fields/methods/local helper types, but never class initialization blocks.
_ATOMIC_TYPE_OWNING_CONCERNS = frozenset(
    {"variables", "inputs", "outputs", "stored_state", "payloads"}
)

_ATOMIC_INITIALIZE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
    "required": [],
    "additionalProperties": True,
}


def _atomic_parameters_for_request(
    payload: Mapping[str, Any],
    *,
    response_region: str,
) -> tuple[dict[str, Any], str]:
    if response_region == "initialize":
        return _ATOMIC_INITIALIZE_PARAMETERS, "initialize_statements"
    recipe = payload.get("generation_recipe")
    preferred = (
        str(recipe.get("preferred_shape") or "").strip()
        if isinstance(recipe, Mapping)
        else ""
    )
    concern = payload.get("concern")
    concern_name = (
        str(concern.get("name") or "").strip()
        if isinstance(concern, Mapping)
        else ""
    )
    if concern_name == "stored_state":
        parameters = _ATOMIC_DECLARATION_MEMBERS_PARAMETERS
        shape = preferred or "declarations_only_fields_or_private_nested_types"
    elif concern_name and concern_name not in _ATOMIC_TYPE_OWNING_CONCERNS:
        return (
            _ATOMIC_LOGIC_MEMBERS_PARAMETERS,
            preferred or "logic_fields_methods_only",
        )
    else:
        parameters = _ATOMIC_MEMBERS_PARAMETERS
        shape = preferred or "smallest_components"

    host_symbol = str(payload.get("host_selected_class") or "").strip()
    if host_symbol:
        parameters = deepcopy(parameters)
        for category in ("records", "enums", "classes"):
            category_schema = parameters["properties"].get(category)
            if not isinstance(category_schema, Mapping):
                continue
            name_schema = category_schema["items"]["properties"]["name"]
            name_schema["not"] = {"enum": [host_symbol]}
            name_schema["description"] = (
                f"Name of a nested runtime helper. {host_symbol} is the existing outer class "
                "and must not be declared again. Place outer fields in fields, not in a class wrapper."
            )
    return parameters, shape


_COMMON_JAVA_NAMES = {
    "ArrayDeque": "java.util.ArrayDeque",
    "ArrayList": "java.util.ArrayList",
    "Collection": "java.util.Collection",
    "Collections": "java.util.Collections",
    "Comparator": "java.util.Comparator",
    "Deque": "java.util.Deque",
    "EnumSet": "java.util.EnumSet",
    "HashMap": "java.util.HashMap",
    "HashSet": "java.util.HashSet",
    "LinkedHashMap": "java.util.LinkedHashMap",
    "LinkedHashSet": "java.util.LinkedHashSet",
    "List": "java.util.List",
    "Map": "java.util.Map",
    "Objects": "java.util.Objects",
    "Optional": "java.util.Optional",
    "Queue": "java.util.Queue",
    "Set": "java.util.Set",
    "UUID": "java.util.UUID",
    "ConcurrentHashMap": "java.util.concurrent.ConcurrentHashMap",
    "Lock": "java.util.concurrent.locks.Lock",
    "ReentrantLock": "java.util.concurrent.locks.ReentrantLock",
    "ReadWriteLock": "java.util.concurrent.locks.ReadWriteLock",
    "ReentrantReadWriteLock": "java.util.concurrent.locks.ReentrantReadWriteLock",
    "AtomicBoolean": "java.util.concurrent.atomic.AtomicBoolean",
    "AtomicInteger": "java.util.concurrent.atomic.AtomicInteger",
    "AtomicLong": "java.util.concurrent.atomic.AtomicLong",
}
_KNOWN_JAVA_FQCN_ALIASES = {
    "java.util.concurrent.Lock": "java.util.concurrent.locks.Lock",
    "java.util.concurrent.ReentrantLock": "java.util.concurrent.locks.ReentrantLock",
    "java.util.concurrent.ReadWriteLock": "java.util.concurrent.locks.ReadWriteLock",
    "java.util.concurrent.ReentrantReadWriteLock": "java.util.concurrent.locks.ReentrantReadWriteLock",
}
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")

_JAVA_RESERVED_IDENTIFIERS = frozenset(
    {
        "_",
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "exports",
        "extends",
        "false",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "module",
        "native",
        "new",
        "non-sealed",
        "null",
        "open",
        "opens",
        "package",
        "permits",
        "private",
        "protected",
        "provides",
        "public",
        "record",
        "requires",
        "return",
        "sealed",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "to",
        "transient",
        "transitive",
        "true",
        "try",
        "uses",
        "var",
        "void",
        "volatile",
        "when",
        "while",
        "with",
        "yield",
    }
)


def _canonical_java_identifier(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "$mmm$unnamed"
    cleaned = re.sub(r"[^A-Za-z0-9_$]", "_", raw)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "$mmm$" + cleaned
    if cleaned in _JAVA_RESERVED_IDENTIFIERS:
        cleaned = "$mmm$" + cleaned.replace("-", "_")
    return cleaned


def _java_identifier_renames(decision: Mapping[str, Any]) -> dict[str, str]:
    """Build one deterministic spelling map for model-declared Java identifiers."""

    names: set[str] = set()

    def add(raw: Any) -> None:
        value = str(raw or "").strip()
        if value:
            names.add(value)

    def visit_parameters(raw: Any) -> None:
        for item in raw if isinstance(raw, Sequence) else ():
            if isinstance(item, Mapping):
                add(item.get("name"))

    def visit_methods(raw: Any) -> None:
        for item in raw if isinstance(raw, Sequence) else ():
            if not isinstance(item, Mapping):
                continue
            add(item.get("name"))
            visit_parameters(item.get("parameters") or [])

    for item in decision.get("records") or []:
        if isinstance(item, Mapping):
            add(item.get("name"))
            visit_parameters(item.get("components") or [])
            visit_methods(item.get("methods") or [])
    for item in decision.get("enums") or []:
        if isinstance(item, Mapping):
            add(item.get("name"))
            for constant in item.get("constants") or []:
                add(constant)
    for item in decision.get("classes") or []:
        if not isinstance(item, Mapping):
            continue
        add(item.get("name"))
        for field in item.get("fields") or []:
            if isinstance(field, Mapping):
                add(field.get("name"))
        for constructor in item.get("constructors") or []:
            if isinstance(constructor, Mapping):
                visit_parameters(constructor.get("parameters") or [])
        visit_methods(item.get("methods") or [])
    for field in decision.get("fields") or []:
        if isinstance(field, Mapping):
            add(field.get("name"))
    visit_methods(decision.get("methods") or [])

    renames: dict[str, str] = {}
    for raw in sorted(names):
        canonical = _canonical_java_identifier(raw)
        if canonical != raw:
            renames[raw] = canonical
    return renames


def _rewrite_java_identifiers(value: Any, renames: Mapping[str, str]) -> str:
    """Rewrite declared identifier spellings outside literals/comments."""

    source = str(value or "")
    if not renames:
        return source
    out: list[str] = []
    index = 0
    quote = ""
    line_comment = False
    block_comment = False
    while index < len(source):
        ch = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            out.append(ch)
            if ch == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            out.append(ch)
            if ch == "*" and nxt == "/":
                out.append(nxt)
                index += 2
                block_comment = False
            else:
                index += 1
            continue
        if quote:
            out.append(ch)
            if ch == "\\" and index + 1 < len(source):
                out.append(source[index + 1])
                index += 2
                continue
            if ch == quote:
                quote = ""
            index += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            out.append(ch)
            index += 1
            continue
        if ch == "/" and nxt == "/":
            out.extend((ch, nxt))
            index += 2
            line_comment = True
            continue
        if ch == "/" and nxt == "*":
            out.extend((ch, nxt))
            index += 2
            block_comment = True
            continue
        alias_match = next(
            (
                (alias, canonical)
                for alias, canonical in _KNOWN_JAVA_FQCN_ALIASES.items()
                if source.startswith(alias, index)
            ),
            None,
        )
        if alias_match is not None:
            alias, canonical = alias_match
            out.append(canonical)
            index += len(alias)
            continue
        match = _IDENTIFIER_TOKEN.match(source, index)
        if match is None:
            out.append(ch)
            index += 1
            continue
        token = match.group(0)
        replacement = renames.get(token)
        if replacement is not None:
            # Preserve Java's switch keyword when a semantic field/component happened
            # to use the same source word "default".
            tail = source[match.end():].lstrip()
            if token == "default" and tail.startswith((':', '->')):
                replacement = None
        out.append(replacement if replacement is not None else token)
        index = match.end()
    return "".join(out)




def _atomic_request_payload(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        try:
            value = json.loads(str(message.get("content") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _qualify_common_java_names(value: str) -> str:
    """Qualify common JDK collection/concurrency names outside strings/comments."""

    source = str(value or "")
    out: list[str] = []
    index = 0
    quote = ""
    line_comment = False
    block_comment = False
    while index < len(source):
        ch = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            out.append(ch)
            if ch == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            out.append(ch)
            if ch == "*" and nxt == "/":
                out.append(nxt)
                index += 2
                block_comment = False
            else:
                index += 1
            continue
        if quote:
            out.append(ch)
            if ch == "\\" and index + 1 < len(source):
                out.append(source[index + 1])
                index += 2
                continue
            if ch == quote:
                quote = ""
            index += 1
            continue
        if ch in {'"', "'"}:
            quote = ch
            out.append(ch)
            index += 1
            continue
        if ch == "/" and nxt == "/":
            out.extend((ch, nxt))
            index += 2
            line_comment = True
            continue
        if ch == "/" and nxt == "*":
            out.extend((ch, nxt))
            index += 2
            block_comment = True
            continue
        alias_match = next(
            (
                (alias, canonical)
                for alias, canonical in _KNOWN_JAVA_FQCN_ALIASES.items()
                if source.startswith(alias, index)
            ),
            None,
        )
        if alias_match is not None:
            alias, canonical = alias_match
            out.append(canonical)
            index += len(alias)
            continue
        match = _IDENTIFIER_TOKEN.match(source, index)
        if match is None:
            out.append(ch)
            index += 1
            continue
        token = match.group(0)
        previous = source[:index].rstrip()
        if token in _COMMON_JAVA_NAMES and not previous.endswith("."):
            out.append(_COMMON_JAVA_NAMES[token])
        else:
            out.append(token)
        index = match.end()
    return "".join(out)


def _java_modifiers(raw: Any, *, kind: str) -> str:
    allowed = {
        "field": ("public", "protected", "private", "static", "final", "volatile", "transient"),
        "method": ("public", "protected", "private", "static", "final", "synchronized"),
        "class": ("public", "protected", "private", "static", "final", "abstract"),
        "record": ("public", "protected", "private", "static"),
        "enum": ("public", "protected", "private", "static"),
    }[kind]
    requested = {str(item) for item in raw if str(item) in allowed}
    return " ".join(item for item in allowed if item in requested)



def _java_parameters(raw: Any, *, renames: Mapping[str, str] | None = None) -> str:
    active = renames or {}
    result: list[str] = []
    for item in raw if isinstance(raw, Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        java_type = _qualify_common_java_names(
            _rewrite_java_identifiers(str(item.get("type") or "").strip(), active)
        )
        raw_name = str(item.get("name") or "").strip()
        name = active.get(raw_name, _canonical_java_identifier(raw_name))
        result.append(f"{java_type} {name}")
    return ", ".join(result)


def _java_body_lines(
    raw: Any,
    *,
    indent: str,
    renames: Mapping[str, str] | None = None,
) -> list[str]:
    active = renames or {}
    rows: list[str] = []
    for item in raw if isinstance(raw, Sequence) else ():
        text = _qualify_common_java_names(
            _rewrite_java_identifiers(str(item or "").strip(), active)
        )
        if not text:
            continue
        if (
            "\n" not in text
            and not text.endswith((";", "{", "}", ":"))
            and not text.startswith(("//", "/*", "*"))
        ):
            text += ";"
        for row in text.splitlines():
            rows.append(indent + row.rstrip())
    return rows


def _render_method(
    item: Mapping[str, Any],
    *,
    indent: str = "",
    renames: Mapping[str, str] | None = None,
) -> str:
    active = renames or {}
    modifiers = _java_modifiers(item.get("modifiers") or [], kind="method")
    prefix = (modifiers + " ") if modifiers else ""
    return_type = _qualify_common_java_names(
        _rewrite_java_identifiers(str(item.get("return_type") or "").strip(), active)
    )
    raw_name = str(item.get("name") or "").strip()
    name = active.get(raw_name, _canonical_java_identifier(raw_name))
    params = _java_parameters(item.get("parameters") or [], renames=active)
    throws = [
        _qualify_common_java_names(
            _rewrite_java_identifiers(str(value).strip(), active)
        )
        for value in item.get("throws") or []
        if str(value).strip()
    ]
    header = f"{indent}{prefix}{return_type} {name}({params})"
    if throws:
        header += " throws " + ", ".join(throws)
    body = _java_body_lines(
        item.get("body") or [],
        indent=indent + "    ",
        renames=active,
    )
    return "\n".join([header + " {", *body, indent + "}"])


def _render_field(
    item: Mapping[str, Any],
    *,
    indent: str = "",
    renames: Mapping[str, str] | None = None,
) -> str:
    active = renames or {}
    modifiers = _java_modifiers(item.get("modifiers") or [], kind="field")
    prefix = (modifiers + " ") if modifiers else ""
    java_type = _qualify_common_java_names(
        _rewrite_java_identifiers(str(item.get("type") or "").strip(), active)
    )
    raw_name = str(item.get("name") or "").strip()
    name = active.get(raw_name, _canonical_java_identifier(raw_name))
    initializer = _qualify_common_java_names(
        _rewrite_java_identifiers(str(item.get("initializer") or "").strip(), active)
    )
    if java_type in {"float", "long"} and initializer:
        from .atomic_concern_source import _state_default_literal

        initializer = _state_default_literal(java_type, initializer) or initializer
    suffix = f" = {initializer}" if initializer else ""
    return f"{indent}{prefix}{java_type} {name}{suffix};"


def _erase_java_generic_arguments(value: str) -> str:
    result: list[str] = []
    depth = 0
    for char in str(value or ""):
        if char == "<":
            depth += 1
            continue
        if char == ">" and depth:
            depth -= 1
            continue
        if depth == 0:
            result.append(char)
    return "".join(result)


def _constructor_signature(item: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for parameter in item.get("parameters") or []:
        if not isinstance(parameter, Mapping):
            continue
        java_type = re.sub(
            r"\s+",
            "",
            str(parameter.get("type") or "").strip(),
        )
        result.append(_erase_java_generic_arguments(java_type))
    return tuple(result)


def _constructor_entries(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = [
        value
        for value in item.get("constructors") or []
        if isinstance(value, Mapping)
    ]
    candidates.extend(
        value
        for value in item.get("methods") or []
        if isinstance(value, Mapping)
        and str(value.get("name") or "").strip() == "<init>"
    )
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for value in candidates:
        signature = _constructor_signature(value)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(value)
    return result

def _ordinary_method_entries(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        value
        for value in item.get("methods") or []
        if isinstance(value, Mapping)
        and str(value.get("name") or "").strip() != "<init>"
    ]


def _parameter_pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for value in raw if isinstance(raw, Sequence) else ():
        if not isinstance(value, Mapping):
            continue
        pairs.append(
            (
                str(value.get("type") or "").strip(),
                str(value.get("name") or "").strip(),
            )
        )
    return tuple(pairs)


def _compact_record_body_lines(
    raw: Any,
    *,
    component_names: set[str],
    indent: str,
    renames: Mapping[str, str],
) -> list[str]:
    normalized: list[str] = []
    assignment = re.compile(
        r"^this\.([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(.+?);?$"
    )
    for value in raw if isinstance(raw, Sequence) else ():
        text = str(value or "").strip()
        match = assignment.fullmatch(text)
        if match and match.group(1) in component_names:
            text = f"{match.group(1)} = {match.group(2)}"
        normalized.append(text)
    return _java_body_lines(normalized, indent=indent, renames=renames)


def _render_record_constructor(
    constructor: Mapping[str, Any],
    *,
    record_name: str,
    components: Any,
    renames: Mapping[str, str],
    indent: str = "    ",
) -> str:
    component_pairs = _parameter_pairs(components)
    parameter_pairs = _parameter_pairs(constructor.get("parameters") or [])
    canonical = parameter_pairs == component_pairs
    if canonical:
        component_names = {
            str(name)
            for _java_type, name in component_pairs
            if str(name).strip()
        }
        body = _compact_record_body_lines(
            constructor.get("body") or [],
            component_names=component_names,
            indent=indent + "    ",
            renames=renames,
        )
        return "\n".join(
            [f"{indent}private {record_name} {{", *body, indent + "}"]
        )

    raw_body = [
        str(value or "").strip()
        for value in constructor.get("body") or []
        if str(value or "").strip()
    ]
    if not raw_body or not raw_body[0].startswith("this("):
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: a non-canonical record constructor "
            "must delegate to the canonical constructor with this(...)."
        )
    params = _java_parameters(
        constructor.get("parameters") or [],
        renames=renames,
    )
    throws = [
        _qualify_common_java_names(
            _rewrite_java_identifiers(str(value).strip(), renames)
        )
        for value in constructor.get("throws") or []
        if str(value).strip()
    ]
    header = f"{indent}private {record_name}({params})"
    if throws:
        header += " throws " + ", ".join(throws)
    body = _java_body_lines(
        constructor.get("body") or [],
        indent=indent + "    ",
        renames=renames,
    )
    return "\n".join([header + " {", *body, indent + "}"])


def _render_class_constructor(
    constructor: Mapping[str, Any],
    *,
    class_name: str,
    renames: Mapping[str, str],
    indent: str = "    ",
) -> str:
    params = _java_parameters(
        constructor.get("parameters") or [],
        renames=renames,
    )
    throws = [
        _qualify_common_java_names(
            _rewrite_java_identifiers(str(value).strip(), renames)
        )
        for value in constructor.get("throws") or []
        if str(value).strip()
    ]
    header = f"{indent}{class_name}({params})"
    if throws:
        header += " throws " + ", ".join(throws)
    body = _java_body_lines(
        constructor.get("body") or [],
        indent=indent + "    ",
        renames=renames,
    )
    return "\n".join([header + " {", *body, indent + "}"])


def _render_record(
    item: Mapping[str, Any],
    *,
    renames: Mapping[str, str] | None = None,
) -> str:
    active = renames or {}
    prefix = "private "
    raw_name = str(item.get("name") or "").strip()
    name = active.get(raw_name, _canonical_java_identifier(raw_name))
    components = _java_parameters(item.get("components") or [], renames=active)
    constructors = [
        _render_record_constructor(
            constructor,
            record_name=name,
            components=item.get("components") or [],
            renames=active,
        )
        for constructor in _constructor_entries(item)
    ]
    methods = [
        _render_method(method, indent="    ", renames=active)
        for method in _ordinary_method_entries(item)
    ]
    body = [*constructors, *methods]
    if not body:
        return f"{prefix}record {name}({components}) {{}}"
    return "\n".join([f"{prefix}record {name}({components}) {{", *body, "}"])

def _render_enum(
    item: Mapping[str, Any],
    *,
    renames: Mapping[str, str] | None = None,
) -> str:
    active = renames or {}
    prefix = "private "
    raw_name = str(item.get("name") or "").strip()
    name = active.get(raw_name, _canonical_java_identifier(raw_name))
    constants = ", ".join(
        active.get(str(value), _canonical_java_identifier(value))
        for value in item.get("constants") or []
    )
    return f"{prefix}enum {name} {{ {constants} }}"


def _render_nested_class(
    item: Mapping[str, Any],
    *,
    renames: Mapping[str, str] | None = None,
) -> str:
    active = renames or {}
    prefix = "private static "
    raw_name = str(item.get("name") or "").strip()
    name = active.get(raw_name, _canonical_java_identifier(raw_name))
    rows = [f"{prefix}class {name} {{"]
    rows.extend(
        _render_field(field, indent="    ", renames=active)
        for field in item.get("fields") or []
        if isinstance(field, Mapping)
    )
    rows.extend(
        _render_class_constructor(
            constructor,
            class_name=name,
            renames=active,
        )
        for constructor in _constructor_entries(item)
    )
    rows.extend(
        _render_method(method, indent="    ", renames=active)
        for method in _ordinary_method_entries(item)
    )
    rows.append("}")
    return "\n".join(rows)

def _validate_atomic_type_namespace(
    decision: Mapping[str, Any],
    *,
    reserved_type_names: Sequence[str] = (),
) -> None:
    reserved = {
        _canonical_java_identifier(value)
        for value in reserved_type_names
        if str(value or "").strip()
    }
    seen: dict[str, str] = {}
    for category, kind in (
        ("records", "record"),
        ("enums", "enum"),
        ("classes", "class"),
    ):
        for item in decision.get(category) or []:
            if not isinstance(item, Mapping):
                continue
            name = _canonical_java_identifier(item.get("name"))
            if name in reserved:
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_SCOPE_ESCAPE: nested type "
                    f"{name!r} collides with the host-selected outer class."
                )
            previous = seen.get(name)
            if previous is not None:
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_RESPONSE_INVALID: duplicate nested type "
                    f"{name!r} emitted as both {previous} and {kind}."
                )
            seen[name] = kind


_HOST_JAVA_MODIFIER_ORDER = (
    "public",
    "protected",
    "private",
    "static",
    "final",
    "synchronized",
    "volatile",
    "transient",
    "abstract",
)


def _implementation_api_modifiers(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[str, ...]]:
    grounding = payload.get("host_grounding")
    contract = (
        grounding.get("implementation_contract")
        if isinstance(grounding, Mapping)
        else None
    )
    declarations = (
        contract.get("public_api")
        if isinstance(contract, Mapping)
        else ()
    )
    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for raw in declarations or ():
        text = " ".join(str(raw or "").replace("{ ... }", "").split()).strip()
        if not text:
            continue
        kind = "method" if "(" in text else "field"
        head = text.split("(", 1)[0] if kind == "method" else text.split("=", 1)[0]
        head = head.rstrip(";").strip()
        match = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*$", head)
        if match is None:
            continue
        name = match.group(1)
        modifiers = tuple(
            token
            for token in _HOST_JAVA_MODIFIER_ORDER
            if re.search(rf"\b{re.escape(token)}\b", head)
        )
        if modifiers:
            result[(kind, name)] = modifiers
    return result


def _host_owned_member_modifiers(
    item: Mapping[str, Any],
    *,
    kind: str,
    outer: bool,
    public_api_modifiers: Mapping[tuple[str, str], tuple[str, ...]],
) -> dict[str, Any]:
    result = dict(item)
    name = str(result.get("name") or "").strip()
    if outer:
        modifiers = list(public_api_modifiers.get((kind, name), ("private",)))
    else:
        modifiers = ["private"]
    result["modifiers"] = modifiers
    return result


def _materialize_atomic_java_modifiers(
    decision: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(decision))
    public_api_modifiers = _implementation_api_modifiers(payload)

    result["fields"] = [
        _host_owned_member_modifiers(
            item,
            kind="field",
            outer=True,
            public_api_modifiers=public_api_modifiers,
        )
        if isinstance(item, Mapping)
        else item
        for item in result.get("fields") or []
    ]
    result["methods"] = [
        _host_owned_member_modifiers(
            item,
            kind="method",
            outer=True,
            public_api_modifiers=public_api_modifiers,
        )
        if isinstance(item, Mapping)
        else item
        for item in result.get("methods") or []
    ]

    for category in ("records", "classes"):
        for owner in result.get(category) or []:
            if not isinstance(owner, dict):
                continue
            if "fields" in owner:
                owner["fields"] = [
                    _host_owned_member_modifiers(
                        item,
                        kind="field",
                        outer=False,
                        public_api_modifiers=public_api_modifiers,
                    )
                    if isinstance(item, Mapping)
                    else item
                    for item in owner.get("fields") or []
                ]
            if "methods" in owner:
                owner["methods"] = [
                    _host_owned_member_modifiers(
                        item,
                        kind="method",
                        outer=False,
                        public_api_modifiers=public_api_modifiers,
                    )
                    if isinstance(item, Mapping)
                    else item
                    for item in owner.get("methods") or []
                ]
    return result


def _force_outer_static(item: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(item)
    modifiers = [
        str(value)
        for value in result.get("modifiers") or []
        if str(value).strip()
    ]
    if "static" not in modifiers:
        modifiers.append("static")
    result["modifiers"] = modifiers
    return result


def _assignment_to_name(text: str, name: str) -> bool:
    token = re.escape(str(name or "").strip())
    if not token:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_$]){token}\s*(?:\+\+|--|(?:>>>|>>|<<|[+\-*/%&|^])?=(?!=))",
            str(text or ""),
        )
    )


def _decision_body_strings(decision: Mapping[str, Any]) -> tuple[str, ...]:
    rows: list[str] = []

    def visit_methods(raw: Any) -> None:
        for method in raw if isinstance(raw, Sequence) else ():
            if not isinstance(method, Mapping):
                continue
            rows.extend(str(item or "") for item in method.get("body") or ())

    visit_methods(decision.get("methods") or ())
    for initializer in decision.get("static_initializers") or ():
        if isinstance(initializer, Mapping):
            rows.extend(str(item or "") for item in initializer.get("body") or ())
    for category in ("records", "classes"):
        for item in decision.get(category) or ():
            if not isinstance(item, Mapping):
                continue
            visit_methods(item.get("methods") or ())
            for constructor in item.get("constructors") or ():
                if isinstance(constructor, Mapping):
                    rows.extend(str(value or "") for value in constructor.get("body") or ())
    return tuple(rows)


def _field_rows(decision: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = [
        item
        for item in decision.get("fields") or ()
        if isinstance(item, Mapping)
    ]
    for category in ("classes",):
        for owner in decision.get(category) or ():
            if not isinstance(owner, Mapping):
                continue
            rows.extend(
                item
                for item in owner.get("fields") or ()
                if isinstance(item, Mapping)
            )
    return tuple(rows)


def _authority_method_return_types(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return only unambiguous authoritative method return types.

    Tree-sitter-derived contracts are preferred. Text parsing remains a compatibility
    fallback for dependency rows produced before typed_public_api existed.
    """

    candidates: dict[str, set[str]] = {}

    def record(name: Any, return_type: Any) -> None:
        symbol = str(name or "").strip()
        declared = str(return_type or "").strip()
        if symbol and declared:
            candidates.setdefault(symbol, set()).add(declared)

    def ingest(declaration: Any) -> None:
        text = " ".join(str(declaration or "").replace("{ ... }", "").split())
        if not text or "(" not in text:
            return
        match = re.search(
            r"(?:^|\s)([A-Za-z_$][A-Za-z0-9_$<>?,.\[\] ]*)\s+"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
            text,
        )
        if match is None:
            return
        return_type = re.sub(
            r"^(?:(?:public|protected|private|static|final|synchronized|abstract|native)\s+)+",
            "",
            match.group(1).strip(),
        )
        record(match.group(2), return_type)

    for row in payload.get("available_sibling_api") or ():
        if not isinstance(row, Mapping) or row.get("kind") != "method":
            continue
        if row.get("return_type"):
            record(row.get("symbol"), row.get("return_type"))
        else:
            ingest(row.get("declaration"))

    for row in payload.get("dependency_api") or ():
        if not isinstance(row, Mapping):
            continue
        typed = [
            item
            for item in row.get("typed_public_api") or ()
            if isinstance(item, Mapping) and item.get("kind") == "method"
        ]
        if typed:
            for item in typed:
                record(item.get("symbol"), item.get("return_type"))
            continue
        for declaration in row.get("public_api") or ():
            ingest(declaration)

    return {
        name: next(iter(types))
        for name, types in candidates.items()
        if len(types) == 1
    }


def _is_object_type(java_type: str) -> bool:
    return re.sub(r"\s+", "", str(java_type or "")) in {"Object", "java.lang.Object"}


def _validate_atomic_java_decision(
    decision: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    response_region: str,
) -> None:
    if response_region == "initialize":
        return

    body_text = "\n".join(_decision_body_strings(decision))
    final_names: set[str] = set()

    for field in _field_rows(decision):
        modifiers = {str(value) for value in field.get("modifiers") or ()}
        name = str(field.get("name") or "").strip()
        initializer = str(field.get("initializer") or "").strip()
        if "final" in modifiers:
            if not initializer:
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_RESPONSE_INVALID: final field "
                    f"{name!r} must be initialized at its declaration in structured generation."
                )
            if name:
                final_names.add(name)

    for row in payload.get("available_sibling_api") or ():
        if (
            isinstance(row, Mapping)
            and row.get("kind") == "field"
            and row.get("mutable") is False
            and str(row.get("symbol") or "").strip()
        ):
            final_names.add(str(row["symbol"]).strip())

    for name in sorted(final_names):
        if _assignment_to_name(body_text, name):
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: final field "
                f"{name!r} is reassigned by generated executable code."
            )

    authoritative_returns = _authority_method_return_types(payload)
    if not authoritative_returns:
        return
    for method in decision.get("methods") or ():
        if not isinstance(method, Mapping):
            continue
        target_type = str(method.get("return_type") or "").strip()
        if not target_type or target_type == "void" or _is_object_type(target_type):
            continue
        for statement in method.get("body") or ():
            match = re.fullmatch(
                r"\s*return\s+(?:[A-Za-z_$][A-Za-z0-9_$.]*\.)?"
                r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^;]*\)\s*;?\s*",
                str(statement or ""),
                re.DOTALL,
            )
            if match is None:
                continue
            source_type = authoritative_returns.get(match.group(1), "")
            if _is_object_type(source_type):
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_RESPONSE_INVALID: method "
                    f"{method.get('name')!r} returns {target_type} but directly returns "
                    f"authoritative Object-valued call {match.group(1)}(...); "
                    "narrow the runtime value with an explicit type check first."
                )


def _render_atomic_java_structure(
    decision: Mapping[str, Any],
    *,
    response_region: str,
    host_symbol: str = "",
) -> str:
    if response_region == "initialize":
        return "\n".join(
            _java_body_lines(decision.get("statements") or [], indent="")
        ).strip()

    _validate_atomic_type_namespace(
        decision,
        reserved_type_names=(host_symbol,) if host_symbol else (),
    )
    renames = _java_identifier_renames(decision)
    rows: list[str] = []
    rows.extend(
        _render_record(item, renames=renames)
        for item in decision.get("records") or []
        if isinstance(item, Mapping)
    )
    rows.extend(
        _render_enum(item, renames=renames)
        for item in decision.get("enums") or []
        if isinstance(item, Mapping)
    )
    rows.extend(
        _render_nested_class(item, renames=renames)
        for item in decision.get("classes") or []
        if isinstance(item, Mapping)
    )
    rows.extend(
        _render_field(_force_outer_static(item), renames=renames)
        for item in decision.get("fields") or []
        if isinstance(item, Mapping)
    )
    outer_constructors = [
        item
        for item in decision.get("methods") or []
        if isinstance(item, Mapping)
        and str(item.get("name") or "").strip() == "<init>"
    ]
    if outer_constructors:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: <init> is valid only inside a "
            "structured record/class; outer-class construction is host-owned."
        )
    rows.extend(
        _render_method(_force_outer_static(item), renames=renames)
        for item in decision.get("methods") or []
        if isinstance(item, Mapping)
    )
    for initializer in decision.get("static_initializers") or []:
        if not isinstance(initializer, Mapping):
            continue
        rows.append(
            "\n".join(
                [
                    "static {",
                    *_java_body_lines(
                        initializer.get("body") or [],
                        indent="    ",
                        renames=renames,
                    ),
                    "}",
                ]
            )
        )
    return "\n\n".join(row for row in rows if row.strip()).strip()

def _call_atomic_java_region(
    router: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    output_token_ceiling: int | None,
) -> str:
    callback = getattr(router, "generate_tool_decision", None)
    if not callable(callback):
        raise CustomModuleGenerationError(
            "ATOMIC_STRUCTURED_CODER_REQUIRED: router has no generate_tool_decision()."
        )
    multi_callback = getattr(router, "generate_tool_decisions", None)
    if not callable(multi_callback):
        multi_callback = None
    from .atomic_java_assembly import JavaStructureAssembly

    payload = _atomic_request_payload(messages)
    response_region = str(payload.get("response_region") or "members").strip()
    parameters, _ = _atomic_parameters_for_request(payload, response_region=response_region)
    registry = getattr(router, "registry", None)
    config = registry.role(router.profile, "coder") if registry is not None else None
    decision = JavaStructureAssembly(
        callback,
        payload,
        output_token_ceiling=output_token_ceiling,
        config=config,
        multi_callback=multi_callback,
    ).run(parameters)
    decision = _materialize_atomic_java_modifiers(decision, payload=payload)
    from .atomic_concern_source import _validate_region_text

    try:
        _validate_atomic_java_decision(
            decision,
            payload=payload,
            response_region=response_region,
        )
        source = _render_atomic_java_structure(
            decision, response_region=response_region,
            host_symbol=str(payload.get("host_selected_class") or "").strip(),
        )
        _validate_region_text(source, initialize_region=response_region == "initialize")
    except CustomModuleGenerationError as exc:
        raise AtomicJavaDecisionError(
            f"ATOMIC_JAVA_ASSEMBLY_INVALID: {exc}", response=decision,
        ) from exc
    return source


def _call_coder(
    router: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    output_token_ceiling: int | None = None,
    force_non_thinking: bool = False,
    structured_java_region: bool = False,
    tool_stage: str = "generation",
) -> str:
    if structured_java_region:
        return _call_atomic_java_region(
            router,
            messages,
            output_token_ceiling=output_token_ceiling,
        )

    callback = getattr(router, "generate_text", None)
    if not callable(callback):
        raise CustomModuleGenerationError(
            "DIRECT_CODER_ROUTER_REQUIRED: router has no generate_text()."
        )
    kwargs: dict[str, Any] = {}
    for key, value in (
        ("response_format", "text"),
        ("enable_tools", False),
        ("tool_stage", tool_stage),
    ):
        if _supports_kwarg(callback, key):
            kwargs[key] = value
    if output_token_ceiling is not None and _supports_kwarg(
        callback, "output_token_ceiling"
    ):
        kwargs["output_token_ceiling"] = max(1, int(output_token_ceiling))
    if force_non_thinking and _supports_kwarg(callback, "force_non_thinking"):
        kwargs["force_non_thinking"] = True
    native_format_replays = 0
    while True:
        try:
            text = callback("coder", messages, **kwargs)
        except Exception as exc:
            boundary = completion_boundary_error(exc)
            if boundary is not None and boundary.kind == OUTPUT_EXHAUSTED:
                raise OutputBudgetExhausted(
                    "OUTPUT_BUDGET_EXHAUSTED: return to implementation decomposition; "
                    f"completion_tokens={boundary.completion_tokens}, max_tokens={boundary.max_tokens}"
                ) from exc

            from .llama_sse_protocol import is_recoverable_native_format_error

            if (
                native_format_replays < 1
                and is_recoverable_native_format_error(exc)
            ):
                native_format_replays += 1
                print(
                    "custom generation: native response format retry 1/1",
                    flush=True,
                )
                continue
            raise
        return _plain_coder_output(text)

def _compile_log(report: Any) -> str:
    """Extract compiler diagnostics by structure instead of truncating raw logs."""
    fallback = str(getattr(report, "error", "") or "")
    commands = tuple(getattr(report, "commands", ()) or ())
    if not commands:
        return fallback
    path = Path(str(getattr(commands[-1], "log_path", "") or ""))
    if not path.is_file() or path.is_symlink():
        return fallback
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback

    lines = text.splitlines()
    header = re.compile(r"(?i)(?:\.java:\d+:\s*(?:error|warning):|^\s*(?:error|failure):)")
    stop = re.compile(r"^\s*(?:> Task |BUILD (?:FAILED|SUCCESSFUL)|FAILURE:)")
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if header.search(line):
            if current:
                blocks.append("\n".join(current).strip())
            current = [line]
            continue
        if current:
            if stop.search(line):
                blocks.append("\n".join(current).strip())
                current = []
            elif line.strip():
                current.append(line)
            else:
                blocks.append("\n".join(current).strip())
                current = []
    if current:
        blocks.append("\n".join(current).strip())

    diagnostics = "\n\n".join(block for block in blocks if block)
    return diagnostics or fallback or text


def _compile_failure_measure(log: str) -> tuple[int, int]:
    """Well-founded source-repair measure; smaller is objectively closer to compile."""
    error_lines = {
        line.strip()
        for line in log.splitlines()
        if "error" in line.casefold() and line.strip()
    }
    return (0, len(error_lines) or 1)


@dataclass(frozen=True)
class _AtomicGenerationContext:
    root: Path
    module: ProductionModule
    target: Path
    relative: str
    symbol: str
    task: Mapping[str, Any]
    original: str
    original_bytes: bytes | None
    target_existed: bool
    expected_package: str
    require_initialize: bool
    ir_contract: Mapping[str, Any]
    host_grounding: Mapping[str, Any]
    dependency_context: str
    compiler: Any
    before_sha: str
    section: str
    concerns: tuple[dict[str, Any], ...]


def _atomic_ir_plan(
    module: ProductionModule, ir_contract: Any
) -> tuple[str, tuple[dict[str, Any], ...]] | None:
    if not isinstance(ir_contract, Mapping):
        return None
    section = str(module.config.get("implementation_section") or "").strip()
    raw = module.config.get("implementation_atomic_concerns")
    if not section or not isinstance(raw, Sequence) or isinstance(
        raw, (str, bytes, bytearray)
    ):
        return None
    concerns = tuple(dict(item) for item in raw if isinstance(item, Mapping))
    return (section, concerns) if concerns else None


def _atomic_result_receipt(
    context: _AtomicGenerationContext,
    atomic: Mapping[str, Any],
    candidate: str,
    *,
    compile_deferred: bool = False,
) -> dict[str, Any]:
    after_sha = _sha256_text(candidate)
    return {
        "schema_version": "mmm/custom-module-result-v3",
        "module_id": context.module.module_id,
        "kind": context.module.kind,
        "status": "SOURCE_GENERATED",
        "patch_receipt": {
            "schema_version": "mmm/direct-source-write-v1",
            "status": "APPLIED",
            "operations": [{
                "operation": "replace" if context.target_existed else "create",
                "path": context.relative,
                "before_sha256": context.before_sha if context.target_existed else "",
                "after_sha256": after_sha,
            }],
            "touched_paths": [context.relative],
        },
        "operation_count": 1,
        "runtime_tests": [
            "Build the real project and execute the requested GameTest/runtime gates."
        ],
        "source_observation_receipt": {
            "path": context.relative,
            "sha256": context.before_sha,
        },
        "touched_paths": [context.relative],
        "discarded_out_of_scope_paths": [],
        "agent_summary": str(atomic["summary"]).strip(),
        "generation_verification": {
            "status": "PASS",
            "mode": (
                "host_semantic_validation_deferred_to_implementation_graph"
                if compile_deferred
                else "gradle_compile_java_semantic_concerns"
            ),
            "compile_deferred": compile_deferred,
            "target_path": context.relative,
            "atomic_concern_count": int(atomic["concern_count"]),
            "atomic_repair_count": int(atomic["repair_count"]),
            "atomic_first_pass_rejection_count": int(
                atomic.get("first_pass_rejection_count", 0)
            ),
            "atomic_repair_region_rejection_count": int(
                atomic.get("repair_region_rejection_count", 0)
            ),
            "atomic_first_compile_failure_count": int(
                atomic.get("first_compile_failure_count", 0)
            ),
        },
        "output_exhaustion_continuations": 0,
        "generation_checkpoint_resumed": False,
        "required_gates": list(context.module.required_gates),
    }


def _restore_atomic_target(context: _AtomicGenerationContext) -> None:
    if context.target_existed and context.original_bytes is not None:
        _atomic_write(context.target, context.original_bytes)
    else:
        context.target.unlink(missing_ok=True)


def _run_atomic_ir_generation(
    generator: Any, context: _AtomicGenerationContext
) -> dict[str, Any]:
    from .atomic_concern_source import AtomicConcernExecutor
    from .implementation_graph_execution import public_api_errors

    compile_deferred = (
        context.module.config.get("implementation_graph_deferred_compile") is True
        and context.module.module_id.startswith("ir_")
        and isinstance(context.ir_contract, Mapping)
    )
    compile_java = (
        (lambda _root: SimpleNamespace(status="PASS", error="", commands=()))
        if compile_deferred
        else context.compiler.compile_java
    )

    def write_atomic_source(path: Path, source: str) -> None:
        if compile_deferred:
            # Implementation-graph leaves have unique host-selected targets and
            # compile once at the graph transaction boundary. Keep slow model
            # decoding entirely outside the coarse project mutation gate; only the
            # exact final target write needs exclusion against same-path writers.
            with project_path_write_locks(context.root, (context.relative,)):
                _atomic_write(path, source)
            return
        _atomic_write(path, source)

    executor = AtomicConcernExecutor(
        root=context.root,
        target=context.target,
        relative=context.relative,
        symbol=context.symbol,
        original=context.original,
        task=context.task,
        section=context.section,
        concerns=context.concerns,
        grounding=context.host_grounding,
        dependency_source=context.dependency_context,
        require_initialize=context.require_initialize,
        # AtomicConcernExecutor already owns the concern boundary, sibling
        # declaration inventory, scope validation, and compile admission. Production
        # is first-pass only; do not put a second syntax-level JavaStructureAssembly between
        # that host contract and the coder: scalar slots such as type/modifier/body
        # are exactly what caused valid Java intent to be misrouted across schema
        # fields (for example type="final"). The coder emits one complete,
        # concern-local Java region and the host parses/adjudicates its declarations.
        call_coder=lambda messages: _call_coder(
            generator.router,
            messages,
            output_token_ceiling=_ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
            # Atomic production is a source-materialization turn, not a reasoning
            # turn. Qwen thinking can otherwise consume the entire 4096-token page
            # while drafting/reconsidering multiple implementations before Java.
            force_non_thinking=True,
            structured_java_region=False,
            tool_stage="atomic_java",
        ),
        compile_java=compile_java,
        compile_log=_compile_log,
        write_source=write_atomic_source,
        # Production is fail-fast regardless of diagnostic environment variables.
        # One concern decode, one assembled-source compile, no model repair loop.
        region_attempt_limit=1,
        compile_repair_limit=0,
    )

    def execute() -> dict[str, Any]:
        try:
            atomic = executor.run()
            candidate = str(atomic["source"])
            errors = _source_invariant_errors(
                candidate,
                symbol=context.symbol,
                expected_package=context.expected_package,
                require_initialize=context.require_initialize,
            )
            errors += public_api_errors(candidate, context.ir_contract)
            if errors:
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_FINAL_CONTRACT_FAILED: " + "; ".join(errors)
                )
            return _atomic_result_receipt(
                context,
                atomic,
                candidate,
                compile_deferred=compile_deferred,
            )
        except BaseException:
            _restore_atomic_target(context)
            raise

    if compile_deferred:
        return execute()
    with project_write_lock(context.root):
        return execute()

class CustomModuleGenerator:
    """One exact task -> one first-pass source candidate -> host compile gate."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        policy: ScalePolicy | None = None,
        fast_mode: bool = False,
        project_index: Any | None = None,
        checkpoint_root: str | Path | None = None,
    ) -> None:
        self.router = router
        self.policy = policy or ScalePolicy.from_environment()
        self.fast_mode = bool(fast_mode)
        self._cached_index = project_index
        self._cached_root = (
            Path(project_index.root).resolve()
            if project_index is not None and getattr(project_index, "root", None)
            else None
        )
        self._checkpoint_root = (
            Path(checkpoint_root).expanduser().resolve()
            if checkpoint_root is not None
            else None
        )

    def _cache_dir(self, root: Path) -> Path:
        if self._checkpoint_root is not None:
            run_root = self._checkpoint_root.parent.parent
            if run_root != self._checkpoint_root:
                return run_root / ".cache" / "gradle"
        return root / ".minecraft_ai" / "gradle-cache"

    def generate(
        self,
        project_root: str | Path,
        *,
        module: ProductionModule,
        research_modules: Iterable[ProductionModule] = (),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if "implementation_graph_request" in module.config:
            from .implementation_graph_execution import execute_implementation_graph

            return execute_implementation_graph(
                self, project_root, module=module, execution_feedback=execution_feedback,
            )
        del research_modules
        module.validate(policy=self.policy)
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise CustomModuleGenerationError(
                "Custom module target must be a regular project directory."
            )

        bind_workspace = getattr(self.router, "bind_agent_workspace", None)
        if callable(bind_workspace):
            if _supports_kwarg(bind_workspace, "require_fresh_evidence"):
                # Direct coder receives a complete host-owned grounding bundle below.
                # It must not enter the model-driven retrieve/tool-choice loop.
                bind_workspace(root, require_fresh_evidence=False)
            else:
                bind_workspace(root)

        adapter = _resolve_generation_adapter(
            root,
            minecraft_version=minecraft_version,
            loader=loader,
        )
        requested_mappings = (
            str(getattr(adapter, "yarn_mappings", "") or "")
            if mappings is None
            else str(mappings).strip()
        )
        try:
            coordinates = validate_target_coordinates(
                adapter.minecraft_version,
                adapter.loader,
                requested_mappings,
                declared_mappings_applicable=getattr(
                    adapter, "mappings_applicable", None
                ),
            )
        except TargetContractError as exc:
            raise CustomModuleGenerationError(str(exc)) from exc
        adapter_mapping = str(getattr(adapter, "yarn_mappings", "") or "")
        if (
            coordinates.mappings_applicable
            and adapter_mapping
            and coordinates.mappings != adapter_mapping
        ):
            raise CustomModuleGenerationError(
                "TARGET_MAPPINGS_ALIAS: requested mappings disagree with the "
                "resolved executable platform target."
            )

        relative, symbol, task = _exact_target(module)
        target = _safe_target(root, relative)
        target_existed = target.is_file()
        original_bytes: bytes | None = None
        if target.exists() and not target_existed:
            raise CustomModuleGenerationError(
                f"DIRECT_CODER_TARGET_NOT_REGULAR: {relative}"
            )
        if target_existed:
            try:
                original_bytes = target.read_bytes()
                original = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise CustomModuleGenerationError(
                    f"Could not read host-owned source {relative}: {exc}"
                ) from exc
        else:
            original = _materialize_host_scaffold(
                target,
                relative=relative,
                symbol=symbol,
                task=task,
            )

        package_match = _PACKAGE.search(original)
        expected_package = package_match.group(1) if package_match else ""
        require_initialize = bool(_INITIALIZE.search(original))
        grounding = build_generation_implementation_grounding(
            module,
            minecraft_version=adapter.minecraft_version,
        )
        authority_prompt = render_generation_implementation_authority_prompt(
            grounding
        )
        task_text = json.dumps(task, ensure_ascii=False, sort_keys=True)
        ir_contract = module.config.get("implementation_ir_node")
        if isinstance(ir_contract, Mapping):
            raw_dependency_context = str(
                module.config.get("implementation_dependency_context") or ""
            )
            context = _dependency_source_context(root, raw_dependency_context)
        else:
            context = _project_context(
                root,
                target,
                relevance_text=task_text + "\n" + original,
            )
        host_grounding = _direct_host_grounding(
            adapter=adapter,
            task=task,
            ir_contract=ir_contract if isinstance(ir_contract, Mapping) else None,
            dependency_context=context,
            typed_grounding=grounding,
        )
        grounding_text = json.dumps(
            host_grounding,
            ensure_ascii=False,
            sort_keys=True,
        )
        bounded_feedback = _bounded_execution_feedback(execution_feedback)
        system = (
            "You implement exactly one host-owned Minecraft Java source file. "
            "Return only the complete Java source text. "
            "No JSON, no prose, no Markdown fences, no patch or diff. "
            "Do not change the package, public final top-level class name, or "
            "declared public API (including initialize() when required). Do not create "
            "another mod entrypoint. The host already resolved project/platform evidence; "
            "do not search for tools or invent unlisted Minecraft/Fabric APIs. "
            "Only the integration section may perform lifecycle/registration wiring; other "
            "authored sections implement bounded domain logic. This is the only production "
            "decode. The host compiles it as a pass/fail gate and never sends compiler errors "
            "back to the model for repair."
            + ("\n\n" + authority_prompt if authority_prompt else "")
        )
        initial_user = (
            f"Platform: Minecraft {adapter.minecraft_version}; "
            f"loader {adapter.loader}; Java {adapter.java_version}; "
            f"mappings {adapter.yarn_mappings or '<none>'}.\n"
            f"Exact target: {relative}#{symbol}\n\n"
            f"Approved task:\n{task_text}\n\n"
            f"Host implementation grounding:\n{grounding_text}\n\n"
            f"Current host scaffold:\n{original}\n\n"
            f"Relevant existing project source:\n{context or '<none>'}"
        )
        if bounded_feedback:
            initial_user += (
                "\n\nDownstream execution feedback from the previous attempt:\n"
                + json.dumps(
                    bounded_feedback,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        before_sha = (
            "sha256:" + hashlib.sha256(original_bytes).hexdigest()
            if original_bytes is not None else _sha256_text(original)
        )
        summary = ""
        last_failure = ""
        compiler = GradleRunner(self._cache_dir(root))

        atomic_plan = _atomic_ir_plan(module, ir_contract)
        if atomic_plan is not None:
            section, concerns = atomic_plan
            context_state = _AtomicGenerationContext(
                root, module, target, relative, symbol, task, original, original_bytes,
                target_existed, expected_package, require_initialize, dict(ir_contract),
                host_grounding, context, compiler, before_sha, section, concerns,
            )
            return _run_atomic_ir_generation(self, context_state)
        attempt = 0

        with project_write_lock(root):
            while True:
                attempt += 1
                if attempt != 1:
                    raise CustomModuleGenerationError(
                        "DIRECT_CODER_INTERNAL_RETRY_FORBIDDEN: production source generation "
                        "must use exactly one model decode."
                    )
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": initial_user},
                ]

                try:
                    payload = _call_coder(self.router, messages)
                except OutputBudgetExhausted:
                    if target_existed:
                        _atomic_write(target, original_bytes)
                    else:
                        target.unlink(missing_ok=True)
                    raise
                except Exception as exc:
                    boundary = completion_boundary_error(exc)
                    if boundary is not None:
                        if target_existed:
                            _atomic_write(target, original_bytes)
                        else:
                            target.unlink(missing_ok=True)
                        raise
                    last_failure = (
                        "DIRECT_CODER_RESPONSE_FAILED: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
                candidate = payload.replace("\r\n", "\n").replace("\r", "\n")
                summary = f"host-validated source for {relative}#{symbol}"
                invariant_errors = _source_invariant_errors(
                    candidate,
                    symbol=symbol,
                    expected_package=expected_package,
                    require_initialize=require_initialize,
                )
                if isinstance(ir_contract, Mapping):
                    from .implementation_graph_execution import public_api_errors
                    invariant_errors += public_api_errors(candidate, ir_contract)
                if invariant_errors:
                    if target_existed:
                        _atomic_write(target, original_bytes)
                    else:
                        target.unlink(missing_ok=True)
                    raise CustomModuleGenerationError(
                        "DIRECT_CODER_FIRST_PASS_CONTRACT_FAILED: "
                        f"{relative}#{symbol} violated the host source contract on its only "
                        "production decode:\n"
                        + "\n".join(f"- {error}" for error in invariant_errors)
                    )

                _atomic_write(target, candidate)
                report = compiler.compile_java(root)
                if getattr(report, "status", "") == "PASS":
                    after_sha = _sha256_text(candidate)
                    return {
                        "schema_version": "mmm/custom-module-result-v3",
                        "module_id": module.module_id,
                        "kind": module.kind,
                        "status": "SOURCE_GENERATED",
                        "patch_receipt": {
                            "schema_version": "mmm/direct-source-write-v1",
                            "status": "APPLIED",
                            "operations": [
                                {
                                    "operation": "replace" if target_existed else "create",
                                    "path": relative,
                                    "before_sha256": before_sha if target_existed else "",
                                    "after_sha256": after_sha,
                                }
                            ],
                            "touched_paths": [relative],
                        },
                        "operation_count": 1,
                        "runtime_tests": [
                            "Build the real project and execute the requested GameTest/runtime gates."
                        ],
                        "source_observation_receipt": {
                            "path": relative,
                            "sha256": before_sha,
                        },
                        "touched_paths": [relative],
                        "discarded_out_of_scope_paths": [],
                        "agent_summary": summary.strip(),
                        "generation_verification": {
                            "status": "PASS",
                            "mode": "gradle_compile_java",
                            "target_path": relative,
                            "attempt": attempt,
                        },
                        "output_exhaustion_continuations": 0,
                        "generation_checkpoint_resumed": False,
                        "required_gates": list(module.required_gates),
                    }

                last_failure = _compile_log(report) or str(
                    getattr(report, "error", "")
                    or "Gradle compileJava failed."
                )
                if target_existed:
                    _atomic_write(target, original_bytes)
                else:
                    target.unlink(missing_ok=True)
                raise CustomModuleGenerationError(
                    "DIRECT_CODER_FIRST_PASS_COMPILE_FAILED: exact whole-file generation "
                    f"did not compile on its only production decode for {relative}. "
                    "Production does not invoke model repair.\n"
                    + last_failure
                )

            if target_existed:
                _atomic_write(target, original_bytes)
            else:
                target.unlink(missing_ok=True)
            raise CustomModuleGenerationError(
                "DIRECT_CODER_FIRST_PASS_FAILED: exact whole-file generation failed "
                f"before compile for {relative}.\n"
                + last_failure
            )

    def ensure_generation_live_commit(
        self,
        result: Any,
        *,
        project_root: str | Path,
    ) -> bool:
        if not isinstance(result, Mapping):
            return False
        paths = result.get("touched_paths")
        receipt = result.get("patch_receipt")
        if (
            not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes, bytearray))
            or not paths
            or not isinstance(receipt, Mapping)
        ):
            return False
        operations = receipt.get("operations")
        if not isinstance(operations, Sequence) or len(operations) != len(paths):
            return False
        if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
            return False
        for path, operation in zip(paths, operations, strict=True):
            if not isinstance(operation, Mapping) or operation.get("path") != path:
                return False
            expected = str(operation.get("after_sha256") or "")
            try:
                target = _safe_target(
                    Path(project_root).expanduser().resolve(), _normalize_project_path(path),
                )
                content = target.read_text(encoding="utf-8")
            except (CustomModuleGenerationError, OSError, UnicodeError):
                return False
            if _sha256_text(content) != expected:
                return False
        return True

    def finalize_committed_generation_checkpoint(
        self,
        result: Any,
        *,
        project_root: str | Path,
    ) -> bool:
        return self.ensure_generation_live_commit(
            result,
            project_root=project_root,
        )

    def acknowledge_generation_checkpoint(self, result: Any) -> bool:
        return isinstance(result, Mapping)

    def release_generation_checkpoint(self, result: Any) -> bool:
        return isinstance(result, Mapping)

    def discard_generation_checkpoint(self, result: Any) -> bool:
        return isinstance(result, Mapping)


def _normalized_operation_path(item: Mapping[str, Any]) -> str:
    return PurePosixPath(
        str(item.get("path", "")).replace("\\", "/")
    ).as_posix()


_agent_mutable_path = custom_module_path_allowed

__all__ = [
    "CustomModuleGenerationError",
    "CustomModuleGenerator",
]
