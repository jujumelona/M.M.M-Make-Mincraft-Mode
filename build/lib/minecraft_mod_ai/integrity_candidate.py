"""Generate and test a candidate in isolation before any production admission."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from .implementation_identity import compute_content_hash


def build_candidate_evidence(*, project_root, leaf_id, inputs, target, router, store,
                             classpath, gametest_task, report_glob):
    from .integrity_bootstrap import bootstrap_integrity
    from .populate_version_artifact_rules import make_implementation
    from .task_template_catalog import load_template
    from .artifact_materializer import materialize_job_output
    from .artifact_job import ArtifactJob
    from .integrity_validators import validate_java_syntax, validate_side
    from .integrity_evidence import run_gradle_evidence, binding_expectations, contract_hash
    authority = bootstrap_integrity()
    authority.verify_live()
    manifest = load_template(leaf_id)
    authority.types.validate_input(leaf_id + ":input", inputs)
    spec = inputs[next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")]
    implementation = make_implementation(leaf_id, target["minecraft_version"])
    generated = authority.executors[implementation["implementation_id"]](
        inputs, leaf_id=leaf_id, router=router, authority=authority)
    source = generated[next(p["name"] for p in manifest["outputs"] if p["type"] == "code_fragment")]
    if spec["language"] == "java":
        prefix, suffix = spec.get("java_prefix", ""), spec.get("java_suffix", "")
        validate_java_syntax(source, java_version=str(target["java_version"]),
            filename=spec.get("java_filename", Path(spec["target_path"]).name), prefix=prefix, suffix=suffix)
        validate_side(prefix + source + suffix, leaf_id=leaf_id, side=spec["side"],
                      classpath=classpath, java_version=str(target["java_version"]))
    project_root = Path(project_root).resolve()
    if project_root.is_symlink() or any(p.is_symlink() for p in project_root.rglob("*")):
        raise ValueError("CANDIDATE_PROJECT_SYMLINK")
    with tempfile.TemporaryDirectory(prefix="mmm-candidate-") as temp:
        root = Path(temp) / "project"
        shutil.copytree(project_root, root, ignore=shutil.ignore_patterns(".git", ".gradle", "build", ".mmm", ".venv"))
        job = ArtifactJob("candidate", "", "candidate", target_path=spec["target_path"],
                          operation=spec["operation"], anchor=spec.get("anchor", ""),
                          expected_sha256=spec.get("expected_sha256", "").removeprefix("sha256:"))
        receipt = materialize_job_output(job, source, base_dir=root)
        candidate = {"contract_sha256": contract_hash(spec), "target_path": spec["target_path"],
                     "output_sha256": compute_content_hash(source.encode()),
                     "materialized_sha256": "sha256:" + receipt.after_sha256}
        evidence_id = run_gradle_evidence(root, store=store,
            expected=binding_expectations(leaf_id, implementation, target), classpath=classpath,
            gametest_task=gametest_task, report_glob=report_glob, candidate=candidate)
    return {"implementation": implementation, "evidence_id": evidence_id, "output": generated}
