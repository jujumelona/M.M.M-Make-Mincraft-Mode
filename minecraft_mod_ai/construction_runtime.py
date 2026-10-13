"""Prepare one real, version-locked compiler/API inventory for MiMo construction."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .construction_context import build_target_symbols, java_source_units


def _jdk_base(java_version: int) -> str:
    java = shutil.which("java")
    if not java:
        raise ValueError("CONSTRUCTION_JDK_REQUIRED")
    completed = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=15)
    version_text = completed.stderr + completed.stdout
    match = re.search(r'version\s+"(\d+)', version_text)
    if not match or int(match.group(1)) != java_version:
        raise ValueError("CONSTRUCTION_JDK_VERSION_MISMATCH: requested " + str(java_version))
    root = Path(java).resolve().parent.parent
    jmod = root / "jmods/java.base.jmod"
    if not jmod.is_file():
        raise ValueError("CONSTRUCTION_JDK_MODULE_MISSING")
    return str(jmod)


@lru_cache(maxsize=2)
def _symbol_inventory(paths: tuple[str, ...], java_version: int, namespace: str):
    # Parse verified .class/.jar members, not model-invented Fabric APIs.
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
    mapping = str(target.get("mappings_kind") or "").lower()
    if target.get("mappings_applicable") is False or mapping in {"mojang", "official"}:
        namespace = "OFFICIAL"
    elif mapping in {"yarn", "named"}:
        namespace = "NAMED"
    elif mapping == "intermediary":
        namespace = "INTERMEDIARY"
    else:
        raise ValueError("CONSTRUCTION_MAPPING_NAMESPACE_REQUIRED")
    if not java_version or not java_classpath:
        raise ValueError("CONSTRUCTION_CLASSPATH_REQUIRED")
    java_base = _jdk_base(java_version)
    paths = tuple(dict.fromkeys([java_base, *(str(Path(p).resolve()) for p in java_classpath)]))
    if any(not Path(p).exists() for p in paths):
        raise ValueError("CONSTRUCTION_CLASSPATH_PATH_MISSING")
    symbols = dict(_symbol_inventory(paths, java_version, namespace))
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
    return {
        "project_root": str(root),
        "symbols": symbols,
        "source_pages": pages,
        "prerequisite_hashes": prerequisites,
        "target": target,
        "namespace": namespace,
    }
