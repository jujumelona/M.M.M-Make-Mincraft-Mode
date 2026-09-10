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


def execute_checkpointed_job(job, *, context, router, registry, base_dir, execute):
    if base_dir is None:
        return execute(job, context=context, router=router, port_registry=registry)
    root = Path(base_dir).resolve()
    target = (root / job.target_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("ARTIFACT_TARGET_ESCAPE")
    path = root / ".mmm/artifact_jobs.json"
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
    # Only host file work is serialized; model calls elsewhere retain their own scheduling.
    with _LOCK:
        project_lock = _PROJECT_LOCKS.setdefault(str(root), RLock())
    with project_lock:
        state = _load(path)
        relative = target.relative_to(root).as_posix()
        current = sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        prior = state["jobs"].get(job.job_id)
        if prior is not None:
            if prior["input_hash"] != binding:
                raise ValueError(
                    f"ARTIFACT_ADAPT_REQUIRED: changed inputs for {job.job_id}"
                )
            if current is None or state["paths"].get(relative) != current:
                raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
            receipt = deepcopy(prior["receipt"])
            for port in receipt["ports_published"].values():
                registry.publish(TypedPort.from_dict(port))
            receipt["reuse"] = "VERIFIED_CHECKPOINT"
            job.status = "SUCCESS"
            job.rendered_output = receipt["rendered_output"]
            job.validation_receipts = receipt["validations"]
            return receipt
        if relative in state["paths"] and current != state["paths"][relative]:
            raise ValueError(f"ARTIFACT_CHECKPOINT_TARGET_DRIFT: {relative}")
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
        after = sha256(target.read_bytes()).hexdigest()
        state["paths"][relative] = after
        state["jobs"][job.job_id] = {
            "template_id": job.template_id,
            "input_hash": binding,
            "dependency_hashes": dependencies,
            "target_path": relative,
            "before_hash": current,
            "after_hash": after,
            "status": "SUCCESS",
            "receipt": receipt,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {"state": state, "sha256": digest(state)}, sort_keys=True, default=str
            ),
            encoding="utf-8",
        )
        temp.replace(path)
        return receipt
