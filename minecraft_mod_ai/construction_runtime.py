"""Prepare one real, version-locked compiler/API inventory for MiMo construction."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .construction_context import build_target_symbols, java_source_units
from .target_contract import target_symbol_namespace


def _jdk_image(java_version: int) -> str:
    from .java_lsp import _java_major_version, _resolve_project_java_home

    # Preflight, javac and construction must use the same exact target JDK.
    # Colab's PATH can still refer to an older system Java after provisioning.
    root = _resolve_project_java_home(java_version, require_compiler=True).resolve()
    if _java_major_version(root) != java_version:
        raise ValueError("CONSTRUCTION_JDK_VERSION_MISMATCH: requested " + str(java_version))
    image = root / "lib/modules"
    if not image.is_file():
        raise ValueError(
            f"CONSTRUCTION_JDK_RUNTIME_IMAGE_MISSING: java={java_version} home={root} expected={image}"
        )
    return str(image)


@lru_cache(maxsize=2)
def _symbol_inventory(paths: tuple[str, ...], java_version: int, namespace: str,
                      image_signature: tuple[int, int]):
    # Parse verified .class/.jar members, not model-invented Fabric APIs.
    # The runtime image fingerprint is part of the cache identity, so replacing
    # a JDK at the same home cannot reuse an older standard-library inventory.
    return build_target_symbols(paths, java_version=java_version, namespace=namespace,
                                classpath_sides={})


def prepare_construction_context(spec, *, java_classpath, project_root, source_context=(), target_override=None):
    bindings = spec.get("bindings") or {}
    target = json.loads(bindings.get("target_contract_json") or "{}")
    if target_override is not None:
        if target and target != target_override:
            raise ValueError("CONSTRUCTION_TARGET_CONTRACT_MISMATCH")
        target = target_override
    if not target:
        raise ValueError("CONSTRUCTION_IMMUTABLE_TARGET_REQUIRED")
    java_version = int(target.get("java_version") or 0)
    namespace = target_symbol_namespace(target)
    if not java_version or not java_classpath:
        raise ValueError("CONSTRUCTION_CLASSPATH_REQUIRED")
    jdk_image = _jdk_image(java_version)
    paths = tuple(dict.fromkeys([jdk_image, *(str(Path(p).resolve()) for p in java_classpath)]))
    if any(not Path(p).exists() for p in paths):
        raise ValueError("CONSTRUCTION_CLASSPATH_PATH_MISSING")
    image_state = Path(jdk_image).stat()
    symbols = dict(_symbol_inventory(
        paths, java_version, namespace, (image_state.st_size, image_state.st_mtime_ns),
    ))
    root = Path(project_root).resolve()
    pages = {}
    prerequisites = {}
    for source in source_context:
        relative = source["path"]
        absolute = (root / relative).resolve()
        if not absolute.is_relative_to(root) or not absolute.is_file():
            raise ValueError("CONSTRUCTION_PREREQUISITE_MISSING: " + relative)
        data = absolute.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != source.get("sha256"):
            raise ValueError("CONSTRUCTION_PREREQUISITE_HASH_MISMATCH: " + relative)
        prerequisites[relative] = sha
        if absolute.suffix == ".java":
            pages[relative] = java_source_units(data.decode("utf-8"), path=relative)
    from .construction_source_symbols import inspected_project_source_symbols
    project_symbols, rejected = inspected_project_source_symbols(pages, symbols)
    if set(project_symbols) & set(symbols):
        raise ValueError("CONSTRUCTION_PROJECT_SYMBOL_COLLISION")
    symbols.update(project_symbols)
    # Pinned old-version references provide porting evidence, never callable
    # target API authority or permission to install outdated code unchanged.
    reference_members = []
    for item in json.loads(bindings.get("verified_source_code_json") or "[]"):
        if item.get("authority") != "reference_only_requires_target_adaptation":
            continue
        if not item.get("source_commit") or not item.get("adapted_sha256"):
            raise ValueError("CONSTRUCTION_PORT_REFERENCE_UNPINNED")
        ref_id = "reference:" + hashlib.sha256((
            item["repository"] + "@" + item["source_commit"] + ":" +
            item["path"] + ":" + str(item["start_line"])).encode()).hexdigest()[:20]
        pages[ref_id] = [{
            "code": item["code"], "signature": item.get("signature", ""),
            "source_ref": ref_id, "authority": "reference_only_port_required",
            "repository": item["repository"], "source_commit": item["source_commit"],
            "source_sha256": item["adapted_sha256"],
            "migration_diagnostics": item.get("migration_diagnostics", ""),
        }]
        reference_members.append({
            "source_ref": ref_id, "signature": item.get("signature", ""),
            "migration_diagnostics": item.get("migration_diagnostics", ""),
        })
    donor_members = {}
    for row in source_context:
        if row.get("authority") != "installed_target_adapted_source":
            continue
        for capability in row.get("capabilities", ()):
            keys = [key for key, record in project_symbols.items()
                    if record.get("source_path") == row["path"]
                    and record.get("kind") in {"METHOD", "CONSTRUCTOR", "FIELD"}]
            donor_members.setdefault(capability, set()).update(keys)
    missing = [cap for cap, keys in donor_members.items() if not keys]
    if missing:
        raise ValueError("CONSTRUCTION_INSTALLED_DONOR_CALLABLE_API_MISSING: " +
                         ",".join(sorted(missing)))
    return {
        "reference_port_obligations": reference_members,
        "donor_callable_symbols": {cap: sorted(keys) for cap, keys in donor_members.items()},
        "project_root": str(root),
        "source_symbol_rejections": list(rejected),
        "project_symbols": project_symbols,
        "symbols": symbols,
        "source_pages": pages,
        "prerequisite_hashes": prerequisites,
        "target": target,
        "namespace": namespace,
    }
