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


def _load(path):
    """Load the legacy monolithic checkpoint for backward-compatible reuse only."""
    if not path.exists():
        return {"jobs": {}, "paths": {}}
    envelope = json.loads(path.read_text(encoding="utf-8"))
    state = envelope["state"]
    if envelope.get("sha256") != digest(state):
        raise ValueError("ARTIFACT_CHECKPOINT_CORRUPT")
    return state


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


def _legacy_state(root):
    return _load(root / ".mmm/artifact_jobs.json")


def _read_job_record(root, job_id):
    record = _read_record(root, "jobs", job_id)
    if record is not None:
        return record
    return _legacy_state(root)["jobs"].get(job_id)


def _read_path_hash(root, relative):
    record = _read_record(root, "paths", relative)
    if record is not None:
        return record
    return _legacy_state(root)["paths"].get(relative)


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
    if base_dir is None:
        return execute(job, context=context, router=router, port_registry=registry)
    root = Path(base_dir).resolve()
    target = (root / job.target_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("ARTIFACT_TARGET_ESCAPE")
    relative = target.relative_to(root).as_posix()
    definition = job.to_dict()
    for name in ("status", "validation_receipts", "rendered_output"):
        definition.pop(name, None)
    dependencies = {name: digest(registry.get(name).to_dict()) for name in job.requires}
    binding = digest(
        {
            "job": definition,
            "template": load_template(job.template_id),
            "dependency_hashes": dependencies,
            "context": context or {},
        }
    )
    job_lock, target_lock = _locks_for(root, job.job_id, relative)

    # Job identity and target identity are the only checkpoint collision domains.
    # Different jobs writing different targets do not share a project-wide lock, so
    # expensive model/render/template execution and checkpoint commits can overlap.
    with job_lock, target_lock:
        current = _current_sha256(target)
        prior = _read_job_record(root, job.job_id)
        recorded = _read_path_hash(root, relative)
        if prior is not None:
            if prior["input_hash"] != binding:
                raise ValueError(
                    f"ARTIFACT_ADAPT_REQUIRED: changed inputs for {job.job_id}"
                )
            if current is None or recorded != current:
                raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
            return _restore_checkpoint(job, prior["receipt"], registry)
        if recorded is not None and current != recorded:
            raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
        before = current

        try:
            receipt = execute(
                job,
                context=context,
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

        # Re-read the per-target record after execution. This detects any mutation
        # that escaped the in-process target lock without serializing unrelated jobs.
        recorded = _read_path_hash(root, relative)
        if recorded is not None and recorded != before:
            raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")

        checkpoint = {
            "template_id": job.template_id,
            "input_hash": binding,
            "dependency_hashes": dependencies,
            "target_path": relative,
            "before_hash": before,
            "after_hash": after,
            "status": "SUCCESS",
            "receipt": receipt,
        }
        # Commit the target hash first. If the process dies before the job record is
        # written, the next run safely re-executes the job against the verified target
        # instead of accepting an incomplete checkpoint as reusable.
        _write_record(root, "paths", relative, after)
        _write_record(root, "jobs", job.job_id, checkpoint)
        return receipt
