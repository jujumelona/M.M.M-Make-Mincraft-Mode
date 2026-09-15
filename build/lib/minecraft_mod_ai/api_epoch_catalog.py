"""Loader-separated API epochs derived from inspected target JAR declarations."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from .jar_api_extractor import inspect_jar
from .implementation_identity import compute_content_hash, compute_json_schema_hash

@dataclass(frozen=True)
class APIEpoch:
    epoch_id: str
    minecraft_versions: list[str]
    description: str
    key_api_changes: list[str]

@dataclass(frozen=True)
class InspectedAPI:
    minecraft_version: str
    loader: str
    namespace: str
    java_version: int
    jar_hashes: dict[str, str]
    classes: dict

    @property
    def epoch_id(self):
        declarations = {name: {"access": c.access, "side": c.side, "superclass": c.superclass,
                                "interfaces": c.interfaces, "members": [asdict(m) for m in c.members]}
                        for name, c in sorted(self.classes.items())}
        digest = compute_json_schema_hash({"loader": self.loader, "namespace": self.namespace,
                                           "java": self.java_version, "declarations": declarations})
        return self.loader + ":" + digest


def inspect_api_epoch(minecraft_version, *, loader, namespace, java_version, jars,
                      mappings=None, source_namespace=None):
    if loader not in {"fabric", "neoforge", "forge"} or not namespace:
        raise ValueError("EXPLICIT_LOADER_AND_NAMESPACE_REQUIRED")
    from dataclasses import replace
    classes, hashes = {}, {}
    for raw in jars:
        path = Path(raw)
        hashes[str(path.resolve())] = compute_content_hash(path.read_bytes())
        for name, item in inspect_jar(path, java_version=java_version).items():
            if mappings is not None:
                if not source_namespace:
                    raise ValueError("SOURCE_NAMESPACE_REQUIRED")
                translated = []
                for m in item.members:
                    owner, member_name, descriptor = mappings.member(m.owner, m.name, m.descriptor,
                        "f" if m.kind == "FIELD" else "m", source_namespace, namespace)
                    translated.append(replace(m, owner=owner, name=member_name, descriptor=descriptor))
                name = mappings.class_name(name, source_namespace, namespace)
                item = replace(item, name=name,
                    superclass=mappings.class_name(item.superclass, source_namespace, namespace),
                    interfaces=tuple(mappings.class_name(i, source_namespace, namespace) for i in item.interfaces),
                    members=tuple(translated))
            if name in classes and classes[name] != item:
                raise ValueError("CONFLICTING_API_CLASS: " + name)
            classes[name] = item
    if not hashes:
        raise ValueError("API_JARS_REQUIRED")
    if loader == "fabric" and any(n.startswith(("net/neoforged/", "net/minecraftforge/")) for n in classes):
        raise ValueError("MIXED_LOADER_CLASSPATH")
    if loader != "fabric" and any(n.startswith("net/fabricmc/") for n in classes):
        raise ValueError("MIXED_LOADER_CLASSPATH")
    return InspectedAPI(minecraft_version, loader, namespace, java_version, hashes, classes)


def determine_api_epoch(minecraft_version: str, *, inspection: InspectedAPI | None = None) -> str:
    if inspection is None or inspection.minecraft_version != minecraft_version:
        raise ValueError("INSPECTED_API_EPOCH_REQUIRED")
    return inspection.epoch_id


def get_template_for_leaf(canonical_leaf, minecraft_version, *, inspection=None, templates=None):
    epoch = determine_api_epoch(minecraft_version, inspection=inspection)
    matches = [t for t in (templates or ()) if t.get("canonical_leaf") == canonical_leaf
               and t.get("api_epoch") == epoch and t.get("loader") == inspection.loader
               and t.get("namespace") == inspection.namespace]
    if len(matches) != 1:
        raise ValueError("EXACT_EPOCH_TEMPLATE_REQUIRED")
    template = matches[0]
    for required in template.get("api_requirements", ()):
        owner = inspection.classes.get(required["owner"])
        if owner is None or not any(m.name == required["name"] and m.descriptor == required["descriptor"]
                                    and m.is_static == required["static"] for m in owner.members):
            raise ValueError("EPOCH_TEMPLATE_API_MISSING")
    return template["id"]


def requires_compile_evidence(epoch_id, minecraft_version):
    return True


def validate_template_compatibility(template_id, minecraft_version, compile_evidence=None,
                                    *, inspection=None, templates=None, store=None, expected=None):
    try:
        from .integrity_evidence import verify_execution_evidence
        if not isinstance(compile_evidence, dict) or not store or not expected:
            raise ValueError("COMPILE_EVIDENCE_REQUIRED")
        verify_execution_evidence(store, compile_evidence["evidence_id"], expected=expected)
        match = next(t for t in (templates or ()) if t["id"] == template_id)
        actual = get_template_for_leaf(match["canonical_leaf"], minecraft_version,
                                      inspection=inspection, templates=templates)
        if actual != template_id or compile_evidence.get("epoch_id") != inspection.epoch_id:
            raise ValueError("EPOCH_EVIDENCE_MISMATCH")
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        return False, str(exc) or "TEMPLATE_NOT_REGISTERED"
    return True, "OK"


def list_supported_versions():
    # No version is inferred compatible without an inspected classpath.
    return {}


def compile_epoch_templates(inspection, *, declarations, output_dir, registry):
    """Materialize exact epoch variants from explicit executable template declarations.

    Each declaration supplies a canonical leaf, a render mold and exact API member
    requirements. No API names are guessed or rewritten using version thresholds.
    """
    import yaml
    from copy import deepcopy
    output_dir = Path(output_dir).resolve()
    compiled = []
    for declaration in declarations:
        template = deepcopy(declaration)
        if not template.get("render") or not template.get("api_requirements"):
            raise ValueError("EXECUTABLE_EPOCH_TEMPLATE_CONTRACT_REQUIRED")
        leaf = template["canonical_leaf"]
        from .canonical_schema_compiler import canonical_leaves
        if leaf not in canonical_leaves():
            raise ValueError("UNKNOWN_EPOCH_LEAF")
        identifier = f"epochs/{inspection.epoch_id.split(':')[-1]}/{leaf}"
        template.update(id=identifier, api_epoch=inspection.epoch_id,
                        loader=inspection.loader, namespace=inspection.namespace)
        get_template_for_leaf(leaf, inspection.minecraft_version, inspection=inspection, templates=[template])
        path = output_dir / (identifier + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(template, sort_keys=True)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("EPOCH_TEMPLATE_CONFLICT")
        path.write_text(content, encoding="utf-8")
        registry.register_template(identifier, path)
        compiled.append(template)
    return compiled
