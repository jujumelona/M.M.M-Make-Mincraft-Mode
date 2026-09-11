"""Enforce exact registered identities before execution and before checkpoint reuse."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .implementation_identity import compute_content_hash


def canonical_contract(job, resolved, context, authority):
    from .task_template_catalog import load_template
    manifest = load_template(job.canonical_leaf)
    inputs = context.get("canonical_inputs", {}).get(job.job_id)
    if not isinstance(inputs, dict):
        raise ValueError("RUNTIME_CANONICAL_INPUTS_REQUIRED")
    authority.types.validate_input(job.canonical_leaf + ":input", inputs)
    spec = inputs[next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")]
    if spec["context_id"] != resolved.context_id:
        raise ValueError("RUNTIME_CANONICAL_CONTEXT_MISMATCH")
    return manifest, spec


def verify_candidate_content(job, resolved, context, spec, source=None):
    from .evidence_store import EvidenceStore
    from .integrity_evidence import _read_blob, contract_hash
    import json
    impl = resolved.require_leaf_binding(job.canonical_leaf)["implementation"]
    record = json.loads(_read_blob(EvidenceStore(Path(context["evidence_store"])), impl["evidence_id"]))
    candidate = record.get("candidate")
    if not candidate or candidate["contract_sha256"] != contract_hash(spec):
        raise ValueError("RUNTIME_CANDIDATE_CONTRACT_MISMATCH")
    if candidate["target_path"] != spec["target_path"]:
        raise ValueError("RUNTIME_CANDIDATE_TARGET_MISMATCH")
    if source is not None and compute_content_hash(source.encode()) != candidate["output_sha256"]:
        raise ValueError("RUNTIME_OUTPUT_NOT_EVIDENCED")


def validate_canonical_output(job, source, *, resolved, context, authority, target_path, anchor):
    from .implementation_identity import compute_json_schema_hash
    from .integrity_validators import validate_semantic_contract, validate_java_syntax, validate_side
    manifest, spec = canonical_contract(job, resolved, context, authority)
    verify_candidate_content(job, resolved, context, spec, source)
    if target_path != spec["target_path"] or anchor != spec.get("anchor", ""):
        raise ValueError("CANONICAL_OUTPUT_TARGET_MISMATCH")
    receipts = [validate_semantic_contract(source, contract=spec, context_id=resolved.context_id, leaf_id=job.canonical_leaf)]
    if spec["language"] == "java":
        java_version = str(resolved.to_dict()["target"]["java_version"])
        prefix, suffix = spec.get("java_prefix", ""), spec.get("java_suffix", "")
        receipts.append(validate_java_syntax(source, java_version=java_version,
            filename=spec.get("java_filename", Path(target_path).name), prefix=prefix, suffix=suffix))
        receipts.append(validate_side(prefix+source+suffix, leaf_id=job.canonical_leaf, side=spec["side"],
            classpath=context["java_classpath"], java_version=java_version, classpath_sides=context.get("classpath_sides")))
    output = {p["name"]: source if p["type"] == "code_fragment" else {
        "leaf_id": job.canonical_leaf, "context_id": resolved.context_id,
        "content_sha256": compute_content_hash(source.encode()), "contract_sha256": compute_json_schema_hash(spec)}
        for p in manifest["outputs"]}
    authority.types.validate_output(job.canonical_leaf + ":output", output)
    return receipts


def verify_job_binding(job, resolved, context):
    from .integrity_bootstrap import get_integrity_authority
    from .integrity_evidence import binding_expectations, verify_execution_evidence
    from .evidence_store import EvidenceStore
    authority = get_integrity_authority()
    authority.verify_live()
    leaf = job.canonical_leaf
    binding = resolved.require_leaf_binding(leaf)
    impl = binding["implementation"]
    actual = authority.implementations.get_implementation(impl["implementation_id"])
    checks = {
        "implementation_sha256": actual.content_sha256,
        "validator_sha256": authority.implementations.get_validator(impl["validator_profile"]).source_hash,
        "input_schema_sha256": authority.types.get_schema_hash(leaf + ":input"),
        "output_schema_sha256": authority.types.get_schema_hash(leaf + ":output"),
        "authority_sha256": authority.content_hash,
    }
    if any(impl.get(k) != v for k, v in checks.items()) or job.implementation_id != actual.implementation_id:
        raise ValueError("RUNTIME_INTEGRITY_BINDING_MISMATCH")
    if job.template_id and job.template_id != impl.get("template") and job.template_id not in impl.get("prerequisite_templates", ()):
        raise ValueError("RUNTIME_TEMPLATE_BINDING_MISMATCH")
    registry = authority.implementations
    if not registry.verify_implementation_hash(actual.implementation_id, actual.content_sha256):
        raise ValueError("RUNTIME_EXECUTOR_CHANGED")
    if not registry.verify_validator_hash(impl["validator_profile"], impl["validator_sha256"]):
        raise ValueError("RUNTIME_VALIDATOR_CHANGED")
    store_path = context.get("evidence_store")
    if not store_path:
        raise ValueError("RUNTIME_EVIDENCE_STORE_REQUIRED")
    expected = binding_expectations(leaf, impl, resolved.to_dict()["target"])
    record = verify_execution_evidence(EvidenceStore(Path(store_path)), impl["evidence_id"], expected=expected)
    actual_classpath = {str(Path(p).resolve()): compute_content_hash(Path(p).read_bytes())
                        for p in context.get("java_classpath", ())}
    if record["classpath"] != actual_classpath:
        raise ValueError("RUNTIME_CLASSPATH_EVIDENCE_MISMATCH")
    _, spec = canonical_contract(job, resolved, context, authority)
    verify_candidate_content(job, resolved, context, spec)
    return authority


def execute_generator_job(job, *, context, router, port_registry=None, base_dir=None):
    from .resolved_version_context import execution_context
    from .task_template_catalog import load_template
    from .integrity_validators import validate_java_syntax, validate_side
    from .artifact_materializer import materialize_job_output
    resolved = execution_context(context, job)
    if resolved is None:
        raise ValueError("GENERATOR_HOST_CONTEXT_REQUIRED")
    authority = verify_job_binding(job, resolved, context)
    inputs = context.get("canonical_inputs", {}).get(job.job_id)
    if not isinstance(inputs, dict):
        raise ValueError("GENERATOR_CANONICAL_INPUTS_REQUIRED")
    manifest = load_template(job.canonical_leaf)
    spec = inputs[next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")]
    if spec["context_id"] != resolved.context_id:
        raise ValueError("GENERATOR_CONTEXT_MISMATCH")
    if job.produces and len(job.produces) != len(manifest["outputs"]):
        raise ValueError("GENERATOR_OUTPUT_PORT_ARITY")
    if port_registry is not None and any(port_registry.has(name) for name in job.produces):
        raise ValueError("GENERATOR_OUTPUT_PORT_CONFLICT")
    generator = authority.executors[job.implementation_id]
    output = generator(inputs, leaf_id=job.canonical_leaf, router=router, authority=authority)
    source = output[next(p["name"] for p in manifest["outputs"] if p["type"] == "code_fragment")]
    verify_candidate_content(job, resolved, context, spec, source)
    validations = []
    if spec["language"] == "java":
        target = resolved.to_dict()["target"]
        java_version = str(target["java_version"])
        prefix, suffix = spec.get("java_prefix", ""), spec.get("java_suffix", "")
        validations.append(validate_java_syntax(source, java_version=java_version,
            filename=spec.get("java_filename", Path(spec["target_path"]).name), prefix=prefix, suffix=suffix))
        validations.append(validate_side(prefix+source+suffix, leaf_id=job.canonical_leaf, side=spec["side"],
            classpath=context["java_classpath"], java_version=java_version,
            classpath_sides=context.get("classpath_sides")))
    target_job = replace(job, target_path=spec["target_path"], anchor=spec.get("anchor", ""),
                         operation=spec["operation"], expected_sha256=spec.get("expected_sha256", "").removeprefix("sha256:"))
    materialization = None
    if base_dir is not None:
        receipt = materialize_job_output(target_job, source, base_dir=base_dir)
        materialization = {"path": receipt.target_path, "status": receipt.status,
                           "after_sha256": receipt.after_sha256}
    if job.produces:
        # Canonical output ports carry the actual validated output values.
        from .artifact_ports import TypedPort, PortKind
        if len(job.produces) != len(manifest["outputs"]):
            raise ValueError("GENERATOR_OUTPUT_PORT_ARITY")
        ports = {name: TypedPort(name, PortKind.GENERIC, p["type"], json.dumps(output[p["name"]], sort_keys=True), context_id=resolved.context_id)
                 for name, p in zip(job.produces, manifest["outputs"], strict=True)}
    else:
        ports = {}
    if port_registry is not None:
        for port in ports.values():
            port_registry.publish(port)
    job.rendered_output = source
    job.validation_receipts = validations
    job.status = "SUCCESS"
    return {"status": "PASS", "job_id": job.job_id, "context_id": resolved.context_id,
            "rendered_output": source, "canonical_output": output, "validations": validations,
            "materialization": materialization, "ports_published": {k: v.to_dict() for k, v in ports.items()}}
