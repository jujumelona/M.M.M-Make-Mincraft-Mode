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
                      _classpath_signature: tuple):
    # Parse verified .class/.jar members, not model-invented Fabric APIs.
    # Every archive/image/class-directory member participates in cache identity.
    return build_target_symbols(paths, java_version=java_version, namespace=namespace,
                                classpath_sides={})


def _reconcile_project_compiled_sources(symbols, pages, root):
    """Prefer verified current project source over its own stale compiled outputs.

    Fabric split-client compileClasspath legitimately contains build/classes/java/main.
    That directory must remain available to javac for linking, but its old
    AuthoredProgram.class must not be treated as a conflicting external API when
    the exact current AuthoredProgram.java is included as a materialized source.
    External jar conflicts remain errors. The cached bytecode index is not mutated.
    """
    from .java_symbol_authority import is_project_compiled_output

    declarations = {
        unit['owner'] for collection in pages.values() for unit in collection
        if unit.get('kind') in {'class', 'interface', 'enum', 'record'}
    }
    local_owners = set()
    for owner in declarations:
        existing = symbols.get(owner)
        if not existing:
            continue
        source_path = existing.get('source_path', '')
        if source_path and is_project_compiled_output(source_path, root):
            local_owners.add(owner)
    if not local_owners:
        return symbols
    return {
        key: row for key, row in symbols.items()
        if not (row.get('owner') in local_owners
                and row.get('source_path')
                and is_project_compiled_output(row['source_path'], root))
    }


def normalized_construction_target(target):
    """Normalize the host-pinned JDK major once for all construction consumers.

    Platform release metadata can provide the major as the decimal string
    "25"; the classpath preflight already accepted it, while FeatureWorkOrder
    required an int. Preserve the selected version; never substitute defaults.
    """
    if not isinstance(target, dict):
        raise ValueError("CONSTRUCTION_IMMUTABLE_TARGET_REQUIRED")
    raw = target.get("java_version")
    if type(raw) is int:
        java_version = raw
    elif isinstance(raw, str) and raw.isascii() and raw.isdecimal() and raw == raw.strip():
        java_version = int(raw)
    else:
        raise ValueError("CONSTRUCTION_JAVA_VERSION_INVALID")
    if java_version < 1:
        raise ValueError("CONSTRUCTION_JAVA_VERSION_INVALID")
    return {**target, "java_version": java_version}


def prepare_construction_context(spec, *, java_classpath, project_root, source_context=(), target_override=None):
    source_context = tuple(source_context)
    bindings = spec.get("bindings") or {}
    target = json.loads(bindings.get("target_contract_json") or "{}")
    if target_override is not None:
        if target and target != target_override:
            raise ValueError("CONSTRUCTION_TARGET_CONTRACT_MISMATCH")
        target = target_override
    if not target:
        raise ValueError("CONSTRUCTION_IMMUTABLE_TARGET_REQUIRED")
    target = normalized_construction_target(target)
    java_version = target["java_version"]
    namespace = target_symbol_namespace(target)
    if not java_version or not java_classpath:
        raise ValueError("CONSTRUCTION_CLASSPATH_REQUIRED")
    jdk_image = _jdk_image(java_version)
    classpath_paths = tuple(str(Path(p).resolve()) for p in java_classpath)
    paths = tuple(dict.fromkeys([jdk_image, *classpath_paths]))
    if any(not Path(p).exists() for p in paths):
        raise ValueError("CONSTRUCTION_CLASSPATH_PATH_MISSING")
    classpath_signature = []
    for raw_path in paths:
        path = Path(raw_path)
        members = sorted(path.rglob('*.class')) if path.is_dir() else (path,)
        for member in members:
            state = member.stat()
            classpath_signature.append((str(member), state.st_dev, state.st_ino,
                                        state.st_size, state.st_mtime_ns, state.st_ctime_ns))
    symbols = dict(_symbol_inventory(
        paths, java_version, namespace, tuple(classpath_signature),
    ))
    # Classless dependency archives are valid, but the target dependency group
    # must still expose bytecode. The JDK alone cannot certify a target API.
    dependency_paths = set(classpath_paths) - {str(Path(jdk_image).resolve())}
    if not any(row.get("kind") == "CLASS" and row.get("source_path") in dependency_paths
               for row in symbols.values()):
        raise ValueError("CONSTRUCTION_DEPENDENCY_CLASSES_MISSING: " + json.dumps(sorted(dependency_paths)))
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
    symbols = _reconcile_project_compiled_sources(symbols, pages, root)
    project_symbols, rejected = inspected_project_source_symbols(pages, symbols)
    if set(project_symbols) & set(symbols):
        raise ValueError("CONSTRUCTION_PROJECT_SYMBOL_COLLISION")
    symbols.update(project_symbols)
    from .construction_inheritance import project_inherited_api
    project_inherited_api(symbols)
    from .construction_type_contracts import annotate_functional_interfaces
    annotate_functional_interfaces(symbols)
    # Pinned same-version references provide partial porting evidence. The
    # original loader/mappings and unresolved dependencies still need adaptation.
    reference_members = []
    reference_ids = set()
    for item in json.loads(bindings.get("verified_source_code_json") or "[]"):
        if item.get("authority") != "reference_only_requires_target_adaptation":
            continue
        if not item.get("source_commit") or not item.get("adapted_sha256"):
            raise ValueError("CONSTRUCTION_PORT_REFERENCE_UNPINNED")
        if item.get('source_minecraft_version') != target['minecraft_version']:
            raise ValueError('CONSTRUCTION_PORT_REFERENCE_VERSION_MISMATCH')
        ref_id = "reference:" + hashlib.sha256((
            item["repository"] + "@" + item["source_commit"] + ":" +
            item["path"] + ":" + str(item["start_line"])).encode()).hexdigest()[:20]
        if ref_id in reference_ids:
            continue
        reference_ids.add(ref_id)
        pages[ref_id] = [{
            "code": item["code"], "signature": item.get("signature", ""),
            "source_ref": ref_id, "authority": "reference_only_port_required",
            "repository": item["repository"], "source_commit": item["source_commit"],
            "source_sha256": item["adapted_sha256"],
            "migration_diagnostics": item.get("migration_diagnostics", ""),
            "source_minecraft_version": item['source_minecraft_version'],
            "source_closure_complete": item.get('source_closure_complete', True),
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
    # Explicitly distinguish empty donor discovery/handoff from a failing
    # bytecode lookup. A valid target API inventory is not reused mod source.
    print('CONSTRUCTION SOURCE HANDOFF: ' + json.dumps({
        'source_context_count': len(source_context),
        'java_source_unit_count': sum(len(rows) for rows in pages.values()),
        'installed_donor_capabilities': sorted(donor_members),
        'reference_port_count': len(reference_members),
        'requested_verified_source_records': len(json.loads(
            bindings.get('verified_source_code_json') or '[]')),
    }, ensure_ascii=False), flush=True)
    return {
        "reference_port_obligations": reference_members,
        "donor_callable_symbols": {cap: sorted(keys) for cap, keys in donor_members.items()},
        "project_root": str(root),
        # Direct Java members must prove their references against the SAME
        # selected version and Gradle classpath before any checkpoint commit.
        "java_classpath": classpath_paths,
        "source_symbol_rejections": list(rejected),
        "project_symbols": project_symbols,
        "symbols": symbols,
        "source_pages": pages,
        "prerequisite_hashes": prerequisites,
        "target": target,
        "namespace": namespace,
    }
