"""Pure runtime-record views and materialized source context, without model dependencies."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

TYPED_UI_OWNER = "__typed_ui__"


def feature_module_id(feature_id: str) -> str:
    return "feature_" + re.sub(r"[^a-z0-9_]", "_", feature_id.casefold())


def _runtime_records(sections):
    for section in ("behavior_contract", "algorithm", "failure_and_limits", "state_model", "integration"):
        value = (sections or {}).get(section, {})
        specification = value.get("specification", {}) if isinstance(value, Mapping) else {}
        if not isinstance(specification, Mapping):
            continue
        for concern, rows in specification.items():
            if not isinstance(rows, list) or concern == "inapplicable_concerns":
                continue
            if section == "state_model" and concern not in {"transitions", "updates"}:
                continue
            if section == "integration" and concern != "entry_points":
                continue
            for index, row in enumerate(rows):
                if isinstance(row, Mapping):
                    yield f"{section}.{concern}[{index}]", deepcopy(dict(row))


def owned_record_refs(sections, owners, *, include_typed_ui=False):
    known = dict(_runtime_records(sections))
    result, seen = set(), set()
    for owner in owners:
        for ref in owner.get("worksheet_record_refs", ()):
            if ref not in known:
                raise ValueError("RUNTIME_OWNER_RECORD_UNKNOWN: " + ref)
            if ref in seen:
                raise ValueError("RUNTIME_OWNER_RECORD_DUPLICATE: " + ref)
            seen.add(ref)
            if include_typed_ui or owner["feature_id"] != TYPED_UI_OWNER:
                result.add(ref)
    return result


def runtime_sections(sections, owners, *, for_ui=False):
    """Project only the records implemented by this Typed dispatcher."""
    moved = owned_record_refs(sections, owners, include_typed_ui=True)
    ui = {ref for owner in owners if owner["feature_id"] == TYPED_UI_OWNER
          for ref in owner["worksheet_record_refs"]}
    removed = (set(dict(_runtime_records(sections))) - ui
               if for_ui and owners else moved)
    result = deepcopy(dict(sections or {}))
    for section, value in result.items():
        spec = value.get("specification", {})
        for concern, rows in list(spec.items()):
            if concern == "inapplicable_concerns" or not isinstance(rows, list):
                continue
            spec[concern] = [row for index, row in enumerate(rows)
                             if f"{section}.{concern}[{index}]" not in removed]
    return result


def owner_records(sections, owners, feature_id):
    records = dict(_runtime_records(sections))
    return [{"record_ref": ref, "record": records[ref],
             **({"ui_action": owner["ui_action"]} if owner.get("ui_action") else {})}
            for owner in owners if owner["feature_id"] == feature_id
            for ref in owner["worksheet_record_refs"]]


def ui_record_action_guards(sections, owners):
    """Retain exact action identity after callback-owned rows are projected out."""
    assigned = {ref: owner["ui_action"] for owner in owners
                if owner["feature_id"] == TYPED_UI_OWNER
                for ref in owner["worksheet_record_refs"]}
    projected = {}
    indices = {}
    for ref, _record in _runtime_records(sections):
        if ref not in assigned:
            continue
        concern = ref.split("[", 1)[0]
        index = indices.get(concern, 0)
        projected[f"{concern}[{index}]"] = assigned[ref]
        indices[concern] = index + 1
    return projected


def generated_owner_source_context(project_root, spec, *, required_paths=()):
    """Read only materialized approved sources; required graph edges never vanish."""
    root = Path(project_root).resolve()
    bindings = spec.get("bindings", {})
    optional_paths = json.loads(bindings.get("content_dependency_paths_json", "[]"))
    source_required = json.loads(bindings.get("required_content_source_paths_json", "[]"))
    if not isinstance(source_required, list) or any(
        not isinstance(path, str) for path in source_required
    ):
        raise ValueError("RUNTIME_OWNER_REQUIRED_SOURCE_LIST_INVALID")
    paths = [*optional_paths, *source_required, *required_paths]
    required = {*source_required, *required_paths}
    readable = []
    for relative in dict.fromkeys(paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not str(relative).startswith(("src/main/java/", "src/client/java/")):
            raise ValueError("RUNTIME_OWNER_SOURCE_PATH_INVALID")
        if not path.is_file() or path.stat().st_size > 512 * 1024:
            if relative in required:
                raise ValueError("ARTIFACT_REQUIRED_JAVA_SOURCE_NOT_MATERIALIZED: " + relative)
            continue  # Only explicitly optional sources may be omitted.
        readable.append((relative, path))
    result = []
    per_file = max(1, 9000 // max(1, len(readable)))
    for relative, path in readable:
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        # Exact declarations include methods below a long implementation body;
        # an arbitrary prefix alone would still force the coder to guess APIs.
        declarations = "\n".join(match.group(0) for match in re.finditer(
            r"\b(?:public|protected)\s+[^{};]+[;{]", source))
        declarations = declarations[:min(1800, per_file // 2)]
        excerpt = source[:min(2500, per_file - len(declarations))]
        result.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                       "source": excerpt, "public_declarations": declarations,
                       "truncated": len(excerpt) < len(source),
                       "authority": "materialized_prerequisite_source"})
    return result


def verified_graph_java_source_paths(job, *, context, port_registry, project_root):
    """Consume exact admitted producer ports instead of trusting model-supplied paths.

    The registry of typed ports comes from completed ArtifactJobs. A producer
    must have actually published a code fragment identical to its materialized
    source; mere presence of a filename does not satisfy the dependency.
    """
    from .artifact_ports import PortKind

    mapping = context.get("_artifact_java_source_ports") or {}
    declared = tuple(name for name in job.requires if name in mapping)
    if job.canonical_leaf == "minecraft/entity/registry" and not declared:
        raise ValueError("ARTIFACT_ENTITY_TYPE_SOURCE_DEPENDENCY_MISSING: " + job.job_id)
    root = Path(project_root).resolve()
    paths = []
    for name in declared:
        port = port_registry.get(name) if port_registry is not None else None
        if port is None or port.port_kind is not PortKind.GENERIC or port.target_type != "code_fragment":
            raise ValueError("ARTIFACT_REQUIRED_SOURCE_PORT_MISSING: " + name)
        try:
            source = json.loads(port.value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("ARTIFACT_REQUIRED_SOURCE_PORT_INVALID: " + name) from exc
        if not isinstance(source, str) or not source.strip():
            raise ValueError("ARTIFACT_REQUIRED_SOURCE_PORT_INVALID: " + name)
        relative = mapping[name]
        if not isinstance(relative, str) or not relative.startswith(("src/main/java/", "src/client/java/")):
            raise ValueError("ARTIFACT_REQUIRED_SOURCE_PATH_INVALID: " + name)
        path = root / relative
        # Reject symlink components so we cannot accept a different on-disk owner.
        current = root
        for segment in Path(relative).parts:
            current = current / segment
            if current.is_symlink():
                raise ValueError("ARTIFACT_REQUIRED_SOURCE_PATH_SYMLINK: " + name)
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("ARTIFACT_REQUIRED_SOURCE_NOT_MATERIALIZED: " + name)
        if resolved.read_text(encoding="utf-8") != source:
            raise ValueError("ARTIFACT_REQUIRED_SOURCE_CONTENT_MISMATCH: " + name)
        paths.append(relative)
    return tuple(dict.fromkeys(paths))
