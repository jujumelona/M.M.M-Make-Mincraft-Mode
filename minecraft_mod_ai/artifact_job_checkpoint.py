"""Per-job receipts bound to templates, inputs, dependencies and current file hashes."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .artifact_ports import TypedPort
from .task_template_catalog import load_template

_LOCK = RLock()
_JOB_LOCKS = {}
_TARGET_LOCKS = {}


def digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()



def _record_path(root, namespace, key):
    filename = sha256(str(key).encode("utf-8")).hexdigest() + ".json"
    return root / ".mmm/artifact_jobs" / namespace / filename


def _read_record(root, namespace, key):
    path = _record_path(root, namespace, key)
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = {"key": envelope.get("key"), "value": envelope.get("value")}
    if payload["key"] != key or envelope.get("sha256") != digest(payload):
        raise ValueError("ARTIFACT_CHECKPOINT_CORRUPT")
    return payload["value"]


def _write_record(root, namespace, key, value):
    path = _record_path(root, namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key, "value": value}
    envelope = {**payload, "sha256": digest(payload)}
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(envelope, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_job_record(root, job_id, target_relative=None):
    indexed = _read_record(root, "jobs", job_id)
    relative = target_relative or (indexed or {}).get('target_path')
    if relative:
        path_record = _read_record(root, 'paths', relative)
        if isinstance(path_record, dict) and path_record.get('job_id') == job_id:
            checkpoint = path_record.get('job_checkpoint')
            if (not isinstance(checkpoint, dict) or checkpoint.get('target_path') != relative
                    or checkpoint.get('after_hash') != path_record.get('sha256')):
                raise ValueError('ARTIFACT_CHECKPOINT_CORRUPT')
            return checkpoint
    return indexed


def _read_path_hash(root, relative):
    value = _read_record(root, "paths", relative)
    return value.get('sha256') if isinstance(value, dict) else value


def _locks_for(root, job_id, relative):
    project_key = str(root)
    job_key = (project_key, str(job_id))
    target_key = (project_key, relative)
    with _LOCK:
        job_lock = _JOB_LOCKS.setdefault(job_key, RLock())
        target_lock = _TARGET_LOCKS.setdefault(target_key, RLock())
    return job_lock, target_lock


def _current_sha256(target):
    return sha256(target.read_bytes()).hexdigest() if target.is_file() else None


def _restore_checkpoint(job, receipt, registry):
    restored = deepcopy(receipt)
    for port in restored["ports_published"].values():
        registry.publish(TypedPort.from_dict(port))
    restored["reuse"] = "VERIFIED_CHECKPOINT"
    job.status = "SUCCESS"
    job.rendered_output = restored["rendered_output"]
    job.validation_receipts = restored["validations"]
    return restored


def execute_checkpointed_job(job, *, context, router, registry, base_dir, execute):
    from .resolved_version_context import execution_context

    resolved = execution_context(context, job)
    if resolved is not None:
        from .integrity_dispatcher import verify_job_binding
        verify_job_binding(job, resolved, context or {})
        registry.bind_context(resolved.context_id)
    from .implementation_identity import ExecutorType
    is_generator = job.executor_type == ExecutorType.PYTHON_GENERATOR
    implementation_fingerprint = None
    if is_generator:
        from .construction_implementation import construction_implementation_fingerprint
        implementation_fingerprint = construction_implementation_fingerprint()
    if base_dir is None:
        return execute(job, context=context, router=router, port_registry=registry, base_dir=base_dir)
    root = Path(base_dir).resolve()
    if is_generator:
        from .integrity_dispatcher import _canonical_inputs_for_job
        inputs = _canonical_inputs_for_job(job, context or {})
        if not isinstance(inputs, dict):
            raise ValueError("GENERATOR_CANONICAL_INPUTS_REQUIRED")
        manifest = load_template(job.canonical_leaf)
        spec_port = next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")
        target_relative = inputs[spec_port]["target_path"]
    else:
        target_relative = job.target_path
    target = (root / target_relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("ARTIFACT_TARGET_ESCAPE")
    relative = target.relative_to(root).as_posix()
    definition = job.to_dict()
    for name in ("status", "validation_receipts", "rendered_output"):
        definition.pop(name, None)
    dependencies = {}
    for name in job.requires:
        port = registry.get(name)
        if port is None:
            raise ValueError(
                f"ARTIFACT_REQUIRED_PORT_MISSING: {job.job_id!r} requires {name!r}"
            )
        dependencies[name] = digest(port.to_dict())
    source_hashes = {}
    verified_sources = dict((context or {}).get('_verified_prerequisite_source_hashes') or {})
    if is_generator:
        spec = inputs[spec_port]
        bindings = spec.get("bindings") or {}
        raw_paths = bindings.get("content_dependency_paths_json")
        if raw_paths:
            paths = json.loads(raw_paths)
            if not isinstance(paths, list):
                raise ValueError("ARTIFACT_SOURCE_DEPENDENCY_PATHS_INVALID")
            for relative_source in paths:
                path = (root / str(relative_source)).resolve()
                if not path.is_relative_to(root) or not str(relative_source).startswith(
                    ("src/main/java/", "src/client/java/")
                ):
                    raise ValueError("ARTIFACT_SOURCE_DEPENDENCY_PATH_INVALID")
                source_hashes[relative_source] = _current_sha256(path)
        donor_caps = json.loads(bindings.get("construction_donor_capabilities_json") or "[]")
        if donor_caps:
            from .authored_reuse_bridge import installed_donor_source_context
            for row in installed_donor_source_context(root, donor_caps):
                source_hashes[row["path"]] = row["sha256"]
                verified_sources[row['path']] = row['sha256']
    approved_context = dict(context or {})
    approved_context.pop('_verified_prerequisite_source_hashes', None)
    approved_binding = digest({'job': definition,
                              'template': manifest if is_generator else load_template(job.template_id),
                              'context': approved_context})
    binding = digest(
        {
            "job": definition,
            "template": manifest if is_generator else load_template(job.template_id),
            "dependency_hashes": dependencies,
            "materialized_source_hashes": source_hashes,
            "context": approved_context,
        }
    )
    job_lock, target_lock = _locks_for(root, job.job_id, relative)

    # Job identity and target identity are the only checkpoint collision domains.
    # Different jobs writing different targets do not share a project-wide lock, so
    # expensive model/render/template execution and checkpoint commits can overlap.
    with job_lock, target_lock:
        current = _current_sha256(target)
        prior = _read_job_record(root, job.job_id, relative)
        recorded = _read_path_hash(root, relative)
        if prior is not None:
            if resolved is not None:
                resolved.assert_context(prior["receipt"].get("context_id"), code="REUSE_VERSION_INCOMPATIBLE")
            if prior["input_hash"] != binding:
                if prior.get('approved_input_hash') != approved_binding:
                    raise ValueError(f"ARTIFACT_ADAPT_REQUIRED: changed or unbound approved inputs for {job.job_id}")
                for source, value in source_hashes.items():
                    if value == prior.get('materialized_source_hashes', {}).get(source):
                        continue
                    proof = _read_path_hash(root, source) or verified_sources.get(source)
                    if value is None or proof != value:
                        raise ValueError('ARTIFACT_PREREQUISITE_REFRESH_UNVERIFIED: ' + source)
            if current is None or recorded != current:
                raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
            refreshing = (prior['input_hash'] != binding
                          or prior.get('implementation_fingerprint') != implementation_fingerprint)
            if (refreshing and is_generator and spec.get('operation') in {'CREATE_FILE', 'REPLACE_FILE'}
                    and current != prior.get('after_hash')):
                raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
            if (prior['input_hash'] == binding
                    and prior.get('implementation_fingerprint') == implementation_fingerprint):
                return _restore_checkpoint(job, prior["receipt"], registry)
            # Same approved inputs and verified output, new implementation:
            # regenerate under the normal target lock. Preserve old evidence
            # until a complete new receipt and target hash are committed.
        if recorded is not None and current != recorded:
            raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
        before = current
        execution_inputs = context
        if prior is not None and is_generator and spec.get('operation') in {'CREATE_FILE', 'REPLACE_FILE'}:
            execution_inputs = dict(context or {})
            execution_inputs['_artifact_owned_replacement'] = {
                'job_id': job.job_id, 'target_path': relative,
                'before_sha256': before, 'input_hash': prior['input_hash'],
            }

        try:
            receipt = execute(
                job,
                context=execution_inputs,
                router=router,
                port_registry=registry,
                base_dir=root,
            )
        except Exception:
            job.status = "FAILED"
            raise

        after = _current_sha256(target)
        if after is None:
            raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_MISSING: {relative}")
        if is_generator:
            materialized = receipt.get('materialization') or {}
            if materialized.get('after_sha256') != after:
                raise ValueError(f'ARTIFACT_CHECKPOINT_OUTPUT_DRIFT: {relative}')

        # Re-read the per-target record after execution. This detects any mutation
        # that escaped the in-process target lock without serializing unrelated jobs.
        recorded = _read_path_hash(root, relative)
        if recorded is not None and recorded != before:
            raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")

        checkpoint = {
            "template_id": job.template_id,
            "input_hash": binding,
            "implementation_fingerprint": implementation_fingerprint,
            'approved_input_hash': approved_binding,
            "dependency_hashes": dependencies,
            'materialized_source_hashes': source_hashes,
            "target_path": relative,
            "before_hash": before,
            "after_hash": after,
            "status": "SUCCESS",
            "receipt": receipt,
            "verification_level": "artifact_materialized_not_behavior_proven",
        }
        # One atomic envelope publishes the target identity and this job's
        # complete receipt together. The jobs table is only an index: a crash
        # before updating it cannot detach refreshed bytes from their evidence.
        _write_record(root, "paths", relative, {
            'schema_version': 'mmm/artifact-path-checkpoint-v2',
            'sha256': after, 'job_id': job.job_id, 'job_checkpoint': checkpoint,
        })
        _write_record(root, "jobs", job.job_id, checkpoint)
        return receipt
