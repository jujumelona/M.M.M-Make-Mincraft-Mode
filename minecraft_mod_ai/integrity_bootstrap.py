"""Central registration authority, constructed completely before publication."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from functools import cached_property
from pathlib import Path
from threading import RLock
from types import MappingProxyType

from .canonical_schema_compiler import canonical_leaves, compile_all_schemas
from .implementation_identity import ValidatorType, compute_content_hash
from .implementation_registry import ImplementationRegistry
from .type_registry import TypeRegistry
from .task_template_catalog import RUNTIME_TEMPLATE_ROOT as TEMPLATE_DIR


def _file_hash(path):
    # Git text files have LF identity on every supported checkout platform.
    return compute_content_hash(Path(path).read_bytes().replace(b"\r\n", b"\n"))


@dataclass(frozen=True)
class IntegrityAuthority:
    implementations: ImplementationRegistry
    types: TypeRegistry
    executors: dict
    validators: dict
    files: dict[str, str]

    @cached_property
    def content_hash(self):
        from .implementation_identity import compute_json_schema_hash
        root = Path(__file__).parent.resolve()
        return compute_json_schema_hash({Path(p).relative_to(root).as_posix(): h for p, h in self.files.items()})

    def verify_live(self):
        for name, expected in self.files.items():
            if _file_hash(name) != expected:
                raise ValueError(f"INTEGRITY_SOURCE_CHANGED: {name}")


_lock = RLock()
_authority = None


def build_integrity_authority() -> IntegrityAuthority:
    from . import integrity_validators as v
    from .canonical_generators import generate_canonical_leaf
    from .populate_version_artifact_rules import LEAF_TEMPLATES
    import yaml

    implementations, types = ImplementationRegistry(), TypeRegistry()
    validators = {
        "semantic_contract": (v.validate_semantic_contract, ValidatorType.CUSTOM),
        "java_syntax": (v.validate_java_syntax, ValidatorType.JAVA_SYNTAX),
        "java_parse": (v.validate_java_syntax, ValidatorType.JAVA_SYNTAX),
        "json_schema": (v.validate_json_schema, ValidatorType.JSON_SCHEMA),
        "mod_integration_test": (v.validate_mod_integration, ValidatorType.MOD_INTEGRATION),
        "client_side_only": (v.validate_side, ValidatorType.CUSTOM),
    }
    for name, (function, kind) in validators.items():
        implementations.register_validator(name, function, kind)
    for type_id, schema in compile_all_schemas().items():
        types.register_json_schema(type_id, schema)
        implementations.register_json_schema(type_id, schema, "canonical_schema_compiler:" + type_id)
    paths = set()
    # All declared templates are parsed and registered, including canonical model molds.
    for path in sorted(TEMPLATE_DIR.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        identifier = path.relative_to(TEMPLATE_DIR).with_suffix("").as_posix()
        if not isinstance(raw, dict) or raw.get("id") != identifier:
            raise ValueError(f"TEMPLATE_ID_MISMATCH: {path}")
        implementations.register_template(identifier, path)
        paths.add(path.resolve())
    for identifier in LEAF_TEMPLATES:
        implementations.get_implementation(identifier)
    executors = {}
    for leaf in canonical_leaves():
        identifier = "python_generator:" + leaf
        implementations.register_python_executor(identifier, generate_canonical_leaf)
        executors[identifier] = generate_canonical_leaf
    # Bind helper modules as well as callable bodies. Changes invalidate the snapshot.
    package = Path(__file__).parent
    paths.update(package.glob("*.py"))
    paths.add(Path(inspect.getfile(generate_canonical_leaf)))
    files = {str(p.resolve()): _file_hash(p) for p in sorted(paths)}
    return IntegrityAuthority(implementations, types, MappingProxyType(executors),
                              MappingProxyType({k: value[0] for k, value in validators.items()}), MappingProxyType(files))


def get_integrity_authority() -> IntegrityAuthority:
    global _authority
    with _lock:
        if _authority is None:
            candidate = build_integrity_authority()
            _authority = candidate
        return _authority


def bootstrap_integrity() -> IntegrityAuthority:
    """Publish both compatibility registries only after successful construction."""
    from .implementation_registry import install_global_registry
    from .type_registry import install_global_type_registry
    authority = get_integrity_authority()
    with _lock:
        install_global_registry(authority.implementations)
        install_global_type_registry(authority.types)
    return authority
