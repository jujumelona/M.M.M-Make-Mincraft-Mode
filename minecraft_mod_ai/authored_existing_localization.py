from __future__ import annotations

"""Read-only localization before a small coder receives exact edit authority."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .complete_spec import ProductionModule
from .project_index import ProjectIndex
from .root_cause_trace import emit_root_cause

_SCHEMA = "mmm/authored-existing-localization-v1"
_MAX_CANDIDATES = 24
_MAX_SNIPPET_CANDIDATES = 8
_SNIPPET_CHARS = 900
_MAX_SUPPORTING_PATHS = 5
_ALLOWED_PREFIXES = (
    "src/main/java/", "src/client/java/", "src/main/resources/",
    "src/client/resources/", "src/test/java/", "src/gametest/",
)
_ALLOWED_SUFFIXES = frozenset({
    ".java", ".json", ".mcmeta", ".mcfunction", ".properties",
    ".accesswidener", ".mixins", ".toml", ".yaml", ".yml",
})
_JAVA_PREFIXES = ("src/main/java/", "src/client/java/")
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class AuthoredExistingLocalizationError(RuntimeError):
    pass


def needs_authored_existing_localization(module: Any) -> bool:
    config = getattr(module, "config", None)
    return bool(
        isinstance(config, Mapping)
        and isinstance(config.get("authored_plan"), Mapping)
        and not isinstance(config.get("evidence_task"), Mapping)
    )


def _sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    path = candidate.as_posix()
    if not any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        return ""
    if candidate.suffix.casefold() not in _ALLOWED_SUFFIXES:
        return ""
    return path


def _java(path: str) -> bool:
    return path.endswith(".java") and path.startswith(_JAVA_PREFIXES)


def _source_set(path: str) -> str:
    if path.startswith("src/client/"):
        return "client"
    if path.startswith("src/test/"):
        return "test"
    if path.startswith("src/gametest/"):
        return "gametest"
    return "main"


def _scope(authored: Mapping[str, Any]) -> dict[str, Any]:
    text = str(authored.get("text") or "")
    raw = text.encode("utf-8")
    headings = [
        line.lstrip("#").strip()
        for line in text.splitlines()
        if line.lstrip().startswith("#") and line.lstrip("#").strip()
    ][:16]
    outline = headings or [text[:1200], text[-600:] if len(text) > 1200 else ""]
    semantic_excerpt = text
    if len(semantic_excerpt.encode("utf-8")) > 3 * 1024:
        semantic_excerpt = text[:2200] + "\n…\n" + text[-800:]
    return {
        "requested_prompt": str(authored.get("requested_prompt") or "").strip(),
        "source_text_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "outline": [value for value in outline if value],
        "semantic_excerpt": semantic_excerpt,
    }


def _search_terms(router: Any, scope: Mapping[str, Any]) -> tuple[str, ...]:
    schema = {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string", "minLength": 2, "maxLength": 64},
                "minItems": 1,
                "maxItems": 8,
                "uniqueItems": True,
            }
        },
        "required": ["terms"],
        "additionalProperties": False,
    }
    result = router.generate_tool_decision(
        "coder",
        [
            {
                "role": "system",
                "content": (
                    "You are only a repository retrieval-query role for a small coder. "
                    "Do not design or implement. Return concrete source identifier/search "
                    "terms for the authored scope; translate concepts when useful."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(scope, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        tool_name="derive_authored_repository_search_terms",
        parameters=schema,
        description="Derive bounded repository search terms only.",
    )
    raw = result.get("terms") if isinstance(result, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_TERMS_INVALID")
    terms = tuple(dict.fromkeys(str(x).strip() for x in raw if str(x).strip()))
    if not terms:
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_TERMS_EMPTY")
    return terms[:8]


def _candidates(root: Path, scope: Mapping[str, Any], terms: Sequence[str]) -> list[dict[str, Any]]:
    index = ProjectIndex(root)
    query = " ".join([
        str(scope.get("requested_prompt") or ""),
        str(scope.get("semantic_excerpt") or ""),
        *[str(x) for x in scope.get("outline", ())],
        *terms,
    ])
    context = index.select(query=query, byte_budget=20 * 1024)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    selected = context.get("files")
    if isinstance(selected, list):
        for raw in selected:
            if not isinstance(raw, Mapping):
                continue
            path = _path(raw.get("path"))
            target = root / path if path else root
            if not path or path in seen or not target.is_file() or target.is_symlink():
                continue
            rows.append({
                "path": path,
                "sha256": str(raw.get("sha256") or ""),
                "size_bytes": int(raw.get("size_bytes") or 0),
                "snippet": str(raw.get("content") or "")[:_SNIPPET_CHARS],
            })
            seen.add(path)
            if len(rows) >= _MAX_CANDIDATES:
                break
    for item in index.files:
        if len(rows) >= _MAX_CANDIDATES:
            break
        path = _path(item.path)
        target = root / path if path else root
        if not path or path in seen or not _java(path) or not target.is_file() or target.is_symlink():
            continue
        rows.append({
            "path": path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "snippet": (
                target.read_text(encoding="utf-8", errors="replace")[:_SNIPPET_CHARS]
                if len(rows) < _MAX_SNIPPET_CANDIDATES else ""
            ),
        })
        seen.add(path)
    if not rows:
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_NO_CANDIDATES")
    if not any(_java(str(row["path"])) for row in rows):
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_NO_JAVA_CANDIDATE")
    return rows


def _select(
    router: Any,
    scope: Mapping[str, Any],
    terms: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    paths = [str(row["path"]) for row in candidates]
    java_paths = [path for path in paths if _java(path)]
    schema = {
        "type": "object",
        "properties": {
            "primary_path": {"type": "string", "enum": java_paths},
            "supporting_paths": {
                "type": "array",
                "items": {"type": "string", "enum": paths},
                "maxItems": _MAX_SUPPORTING_PATHS,
                "uniqueItems": True,
            },
        },
        "required": ["primary_path", "supporting_paths"],
        "additionalProperties": False,
    }
    payload = {
        "authored_scope": dict(scope),
        "search_terms": list(terms),
        "candidates": [
            {
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "snippet": row["snippet"] if i < _MAX_SNIPPET_CANDIDATES else "",
            }
            for i, row in enumerate(candidates)
        ],
    }
    result = router.generate_tool_decision(
        "coder",
        [
            {
                "role": "system",
                "content": (
                    "You are only the read-only localization role. Select the minimum "
                    "existing files for the approved authored scope. Do not implement, "
                    "invent files, redesign architecture, or choose outside candidates."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        tool_name="freeze_existing_authored_targets",
        parameters=schema,
        description="Freeze minimum exact existing-project edit targets.",
    )
    if not isinstance(result, Mapping):
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_DECISION_INVALID")
    primary = _path(result.get("primary_path"))
    if not primary or primary not in java_paths:
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_PRIMARY_INVALID")
    raw_support = result.get("supporting_paths")
    if not isinstance(raw_support, Sequence) or isinstance(raw_support, (str, bytes, bytearray)):
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_SUPPORT_INVALID")
    allowed = set(paths)
    support: list[str] = []
    for value in raw_support:
        path = _path(value)
        if not path or path not in allowed:
            raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_PATH_DRIFT")
        if path != primary and path not in support:
            support.append(path)
    if len(support) > _MAX_SUPPORTING_PATHS:
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_TOO_WIDE")
    return primary, tuple(support)


def _receipt_path(root: Path, task_id: str) -> Path:
    if _SAFE_TASK_ID.fullmatch(task_id) is None:
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_TASK_ID_INVALID")
    return root / ".minecraft_ai" / "authored-localization" / f"{task_id}.json"


def _load(root: Path, task_id: str, authored_sha: str) -> dict[str, Any] | None:
    path = _receipt_path(root, task_id)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != _SCHEMA:
        return None
    if value.get("task_id") != task_id or value.get("authored_plan_sha256") != authored_sha:
        return None
    core = dict(value)
    supplied = str(core.pop("localization_sha256", "") or "")
    if supplied != _sha(core):
        return None
    primary = _path(value.get("primary_path"))
    selected = value.get("selected_paths")
    if (
        not primary or not _java(primary) or not isinstance(selected, list)
        or primary not in selected or len(selected) > 1 + _MAX_SUPPORTING_PATHS
    ):
        return None
    normalized = [_path(x) for x in selected]
    if any(not x for x in normalized) or len(normalized) != len(set(normalized)):
        return None
    if any(not (root / x).is_file() or (root / x).is_symlink() for x in normalized):
        return None
    return value


def _write(root: Path, receipt: Mapping[str, Any]) -> None:
    path = _receipt_path(root, str(receipt.get("task_id") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_RECEIPT_UNSAFE")
    path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _localized(module: ProductionModule, receipt: Mapping[str, Any]) -> ProductionModule:
    config = dict(module.config)
    selected = [str(x) for x in receipt["selected_paths"]]
    primary = str(receipt["primary_path"])
    anchors: list[dict[str, Any]] = []
    for path in selected:
        is_java = _java(path)
        anchors.append({
            "kind": "symbol" if is_java else "resource",
            "locator": path + (f"#{PurePosixPath(path).stem}" if is_java else ""),
            "status": "existing",
            "ownership": "host_existing_project_localization",
            "module_id": module.module_id,
            "source_set": _source_set(path),
        })
    primary_anchor = next(
        a for a in anchors if str(a["locator"]).split("#", 1)[0] == primary
    )
    gates = tuple(dict.fromkeys(("target_compile", *module.required_gates)))
    target = {
        key: config[key]
        for key in ("minecraft_version", "loader", "mappings")
        if key in config
    }
    task: dict[str, Any] = {
        "task_id": module.module_id,
        "task_sha256": "",
        "execution_role": "coder",
        "semantic_outcome": (
            "Implement the approved authored design only inside the frozen exact target set."
        ),
        "implementation_obligations": [
            "Implement only the currently host-scheduled authored fragment in frozen targets."
        ],
        "engineering_worksheet": {
            "objective": "Apply authored behavior to host-localized existing targets.",
            "localization": {
                "localization_sha256": receipt["localization_sha256"],
                "primary_path": primary,
                "selected_paths": selected,
            },
        },
        "target_cell": target,
        "depends_on": list(module.depends_on),
        "consumes": [
            f"{dependency}_ready"
            for dependency in module.depends_on
        ],
        "provides": [f"{module.module_id}_ready"],
        "owned_anchors": anchors,
        "production_bindings": [{
            "task_ref": module.module_id,
            "reuse_action": "fresh",
            "owned_anchors": [primary_anchor],
        }],
        "required_gates": list(gates),
        "acceptance": [
            "Only frozen exact existing-project targets are mutated.",
            "The edit passes target compilation and project build verification.",
        ],
    }
    task["task_sha256"] = _sha({k: v for k, v in task.items() if k != "task_sha256"})
    config["evidence_task"] = task
    config["_authored_localization"] = {
        "schema_version": _SCHEMA,
        "localization_sha256": receipt["localization_sha256"],
        "primary_path": primary,
        "selected_paths": selected,
    }
    return ProductionModule(
        module_id=module.module_id,
        kind=module.kind,
        config=config,
        depends_on=module.depends_on,
        required_gates=gates,
    )


def localize_existing_authored_module(
    router: Any, project_root: str | Path, module: ProductionModule
) -> ProductionModule:
    if not needs_authored_existing_localization(module):
        return module
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_PROJECT_INVALID")
    authored = module.config.get("authored_plan")
    if not isinstance(authored, Mapping):
        raise AuthoredExistingLocalizationError("AUTHORED_LOCALIZATION_PLAN_MISSING")
    authored_sha = _sha(dict(authored))
    cached = _load(root, module.module_id, authored_sha)
    if cached is not None:
        emit_root_cause(
            "authored_existing_localization",
            stage="generation",
            operation="localize_existing_authored",
            gate="exact_target_freeze",
            result="PASS",
            reason="reused persisted exact localization",
            details={
                "task_id": module.module_id,
                "primary_path": cached["primary_path"],
                "selected_paths": cached["selected_paths"],
            },
        )
        return _localized(module, cached)
    scope = _scope(authored)
    terms = _search_terms(router, scope)
    candidates = _candidates(root, scope, terms)
    primary, support = _select(router, scope, terms, candidates)
    core: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "task_id": module.module_id,
        "authored_plan_sha256": authored_sha,
        "source_text_sha256": scope["source_text_sha256"],
        "search_terms": list(terms),
        "candidate_set_sha256": _sha([
            {"path": row["path"], "sha256": row["sha256"]} for row in candidates
        ]),
        "primary_path": primary,
        "selected_paths": [primary, *support],
    }
    receipt = {**core, "localization_sha256": _sha(core)}
    _write(root, receipt)
    emit_root_cause(
        "authored_existing_localization",
        stage="generation",
        operation="localize_existing_authored",
        gate="exact_target_freeze",
        result="PASS",
        details={
            "task_id": module.module_id,
            "primary_path": primary,
            "selected_paths": receipt["selected_paths"],
            "candidate_count": len(candidates),
        },
    )
    return _localized(module, receipt)


__all__ = [
    "AuthoredExistingLocalizationError",
    "localize_existing_authored_module",
    "needs_authored_existing_localization",
]
