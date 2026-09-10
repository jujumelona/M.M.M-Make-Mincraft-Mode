"""Per-job receipts bound to templates, inputs, dependencies and current file hashes."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock

from .artifact_ports import TypedPort
from .task_template_catalog import load_template

_LOCK = RLock()
_PROJECT_LOCKS = {}
_TARGET_LOCKS = {}


def digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load(path):
    if not path.exists():
        return {"jobs": {}, "paths": {}}
    envelope = json.loads(path.read_text(encoding="utf-8"))
    state = envelope["state"]
    if envelope.get("sha256") != digest(state):
        raise ValueError("ARTIFACT_CHECKPOINT_CORRUPT")
    return state


def _locks_for(root, relative):
    project_key = str(root)
    target_key = (project_key, relative)
    with _LOCK:
        project_lock = _PROJECT_LOCKS.setdefault(project_key, RLock())
        target_lock = _TARGET_LOCKS.setdefault(target_key, RLock())
    return project_lock, target_lock


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


def _write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {"state": state, "sha256": digest(state)}, sort_keys=True, default=str
        ),
        encoding="utf-8",
    )
    temp.replace(path)


def execute_checkpointed_job(job, *, context, router, registry, base_dir, execute):
    if base_dir is None:
        return execute(job, context=context, router=router, port_registry=registry)
    root = Path(base_dir).resolve()
    target = (root / job.target_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("ARTIFACT_TARGET_ESCAPE")
    path = root / ".mmm/artifact_jobs.json"
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
    project_lock, target_lock = _locks_for(root, relative)

    # A target lock prevents duplicate/same-path jobs from mutating one file at
    # the same time. Different targets intentionally do not share this lock, so
    # expensive model/render/template execution can proceed concurrently.
    with target_lock:
        with project_lock:
            state = _load(path)
            current = _current_sha256(target)
            prior = state["jobs"].get(job.job_id)
            if prior is not None:
                if prior["input_hash"] != binding:
                    raise ValueError(
                        f"ARTIFACT_ADAPT_REQUIRED: changed inputs for {job.job_id}"
                    )
                if current is None or state["paths"].get(relative) != current:
                    raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
                return _restore_checkpoint(job, prior["receipt"], registry)
            if relative in state["paths"] and current != state["paths"][relative]:
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

        # Merge into the latest checkpoint state under the short project lock.
        # Reloading here is required so concurrently completed jobs on different
        # targets cannot overwrite each other's checkpoint records.
        with project_lock:
            state = _load(path)
            prior = state["jobs"].get(job.job_id)
            if prior is not None:
                if prior["input_hash"] != binding:
                    raise ValueError(
                        f"ARTIFACT_ADAPT_REQUIRED: changed inputs for {job.job_id}"
                    )
                recorded = state["paths"].get(relative)
                if recorded != after:
                    raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
                return _restore_checkpoint(job, prior["receipt"], registry)
            recorded = state["paths"].get(relative)
            if recorded is not None and recorded != before:
                raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
            state["paths"][relative] = after
            state["jobs"][job.job_id] = {
                "template_id": job.template_id,
                "input_hash": binding,
                "dependency_hashes": dependencies,
                "target_path": relative,
                "before_hash": before,
                "after_hash": after,
                "status": "SUCCESS",
                "receipt": receipt,
            }
            _write_state(path, state)
        return receipt
