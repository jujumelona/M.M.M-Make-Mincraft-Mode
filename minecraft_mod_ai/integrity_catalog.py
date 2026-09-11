"""Evidence-backed catalog admission and complete canonical reachability audit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .canonical_schema_compiler import canonical_leaves
from .implementation_identity import compute_content_hash
from .integrity_evidence import binding_expectations, verify_execution_evidence


def admit_evidenced_leaves(facts, *, target, inspection, evidence_ids, store, authority):
    """Return a new catalog; missing evidence remains explicitly not reviewed."""
    authority.verify_live()
    if inspection.minecraft_version != target["minecraft_version"] or inspection.java_version != int(target["java_version"]):
        raise ValueError("API_INSPECTION_TARGET_MISMATCH")
    if inspection.loader != target.get("loader", "fabric"):
        raise ValueError("API_INSPECTION_LOADER_MISMATCH")
    for path, digest in inspection.jar_hashes.items():
        if compute_content_hash(Path(path).read_bytes()) != digest:
            raise ValueError("API_INSPECTION_JAR_CHANGED")
    result = deepcopy(facts)
    result["api_inspection"] = {"epoch_id": inspection.epoch_id, "namespace": inspection.namespace,
                                "loader": inspection.loader, "jar_hashes": inspection.jar_hashes}
    # Declarations have exact identities, including overload descriptors.
    result["api_inspection"]["symbols"] = {
        f"{m.owner}#{m.name}{m.descriptor}": {**asdict(m), "static": m.is_static, "namespace": inspection.namespace}
        for c in inspection.classes.values() for m in c.members
    }
    for leaf, evidence_id in evidence_ids.items():
        binding = result["leaf_bindings"][leaf]
        impl = binding["implementation"]
        expected = binding_expectations(leaf, impl, target)
        record = verify_execution_evidence(store, evidence_id, expected=expected)
        if not record.get("candidate"):
            raise ValueError("CANDIDATE_EXECUTION_REQUIRED")
        if record["classpath"] != inspection.jar_hashes:
            raise ValueError("EVIDENCE_INSPECTION_CLASSPATH_MISMATCH")
        if impl["authority_sha256"] != authority.content_hash:
            raise ValueError("CATALOG_AUTHORITY_CHANGED")
        registry = authority.implementations
        if not registry.verify_implementation_hash(impl["implementation_id"], impl["implementation_sha256"]):
            raise ValueError("CATALOG_IMPLEMENTATION_CHANGED")
        if not registry.verify_validator_hash(impl["validator_profile"], impl["validator_sha256"]):
            raise ValueError("CATALOG_VALIDATOR_CHANGED")
        for direction in ("input", "output"):
            if authority.types.get_schema_hash(f"{leaf}:{direction}") != impl[f"{direction}_schema_sha256"]:
                raise ValueError("CATALOG_SCHEMA_CHANGED")
        impl["evidence_id"] = evidence_id
        binding["state"] = "admitted"
        binding.pop("reason", None)
    return result


def audit_reachability(facts, *, authority, target=None, store=None, production=False):
    failures = []
    bindings = facts.get("leaf_bindings", {})
    leaves = canonical_leaves()
    for leaf in leaves:
        try:
            binding = bindings[leaf]
            impl = binding["implementation"]
            registered = authority.implementations.get_implementation(impl["implementation_id"])
            if registered.content_sha256 != impl["implementation_sha256"]:
                raise ValueError("IMPLEMENTATION_HASH_MISMATCH")
            if impl["executor_type"] == "python_generator" and not callable(authority.executors.get(impl["implementation_id"])):
                raise ValueError("GENERATOR_NOT_CALLABLE")
            validator = authority.implementations.get_validator(impl["validator_profile"])
            if validator.source_hash != impl["validator_sha256"]:
                raise ValueError("VALIDATOR_HASH_MISMATCH")
            for direction in ("input", "output"):
                type_id = leaf + ":" + direction
                if authority.types.get_schema_hash(type_id) != impl[direction + "_schema_sha256"]:
                    raise ValueError("SCHEMA_HASH_MISMATCH")
            if production:
                if binding["state"] != "admitted":
                    raise ValueError("NOT_ADMITTED")
                verify_execution_evidence(store, impl["evidence_id"], expected=binding_expectations(leaf, impl, target))
        except (ValueError, KeyError, OSError) as exc:
            failures.append({"leaf": leaf, "reason": str(exc)})
    return {"status": "FAIL" if failures else "PASS", "leaf_count": len(leaves),
            "mode": "production" if production else "registration", "failures": failures}
