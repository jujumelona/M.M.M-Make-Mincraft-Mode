"""Composed registry patches must reach startup with their final byte proof."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.artifact_graph_executor import (
    ArtifactGraphError,
    _wire_validated_java_startup,
)
from minecraft_mod_ai.artifact_materializer import materialize_java_patch


def _patched_registry(root):
    relative = "src/main/java/proof/Things.java"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text("""package proof;
public final class Things {
    /* MMM:members */
    public static void register() {}
}
""")
    meta = root / "src/main/resources/fabric.mod.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"entrypoints": {"main": []}}))
    receipts = []
    for name in ("first", "second", "third"):
        materialized = materialize_java_patch(
            relative, "/* MMM:members */", f"public static int {name} = 1;",
            base_dir=root,
        )
        receipts.append({"status": "PASS", "materialization": {
            **asdict(materialized), "path": materialized.target_path,
        }})
    jobs = [SimpleNamespace(canonical_leaf="minecraft/item/registry") for _ in receipts]
    return target, meta, jobs, receipts


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_startup_accepts_composed_patch_receipts_in_any_completion_order(tmp_path, order):
    target, meta, jobs, receipts = _patched_registry(tmp_path)
    before = target.read_bytes()
    for _ in range(2):
        _wire_validated_java_startup(
            [jobs[i] for i in order], [receipts[i] for i in order],
            context={"project_root": str(tmp_path)}, base_dir=tmp_path,
        )
        assert target.read_bytes() == before
        entries = json.loads(meta.read_text())["entrypoints"]["main"]
        assert len(entries) == 1
        bootstrap = tmp_path / "src/main/java" / (entries[0].replace(".", "/") + ".java")
        assert "proof.Things.register();" in bootstrap.read_text()


@pytest.mark.parametrize("corruption", ["missing_link", "overwrite", "edit", "bad_hash"])
def test_startup_rejects_unproven_shared_file_history(tmp_path, corruption):
    target, meta, jobs, receipts = _patched_registry(tmp_path)
    if corruption == "missing_link":
        del receipts[1]
        del jobs[1]
    elif corruption == "overwrite":
        receipts[1]["materialization"]["operation"] = "whole_file"
    elif corruption == "edit":
        target.write_text(target.read_text() + "\n// unreceipted edit\n")
    else:
        receipts[0]["materialization"]["after_sha256"] = "f" * 64
    with pytest.raises(ArtifactGraphError, match="ARTIFACT_JAVA_STARTUP_SOURCE_UNVERIFIED"):
        _wire_validated_java_startup(
            jobs, receipts, context={"project_root": str(tmp_path)}, base_dir=tmp_path,
        )
    assert json.loads(meta.read_text())["entrypoints"]["main"] == []


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("change", ["patch", "external_edit", "whole_file"])
def test_earlier_shard_checkpoint_reuses_final_shared_registry_proof(tmp_path, monkeypatch, legacy, change):
    from minecraft_mod_ai.artifact_job import ArtifactJob
    from minecraft_mod_ai.artifact_job_checkpoint import (
        _read_record,
        _write_record,
        execute_checkpointed_job,
    )
    from minecraft_mod_ai.artifact_ports import PortRegistry

    monkeypatch.setattr("minecraft_mod_ai.resolved_version_context.execution_context", lambda *a: None)
    monkeypatch.setattr("minecraft_mod_ai.artifact_job_checkpoint.load_template", lambda _: {})
    target, meta, _, _ = _patched_registry(tmp_path)
    relative = target.relative_to(tmp_path).as_posix()
    calls = []

    def job(name):
        return ArtifactJob(job_id=name, owner_module=name, template_id="test/patch", target_path=relative)

    def execute(job, **kwargs):
        calls.append(job.job_id)
        if job.job_id == 'replacement':
            from minecraft_mod_ai.artifact_materializer import materialize_whole_file
            mat = materialize_whole_file(target, 'package proof; public class Things { '
                'public static void register() {} }')
        else:
            mat = materialize_java_patch(relative, "/* MMM:members */",
                f"public static int {job.job_id} = 1;", base_dir=tmp_path)
        return {"status": "PASS", "rendered_output": job.job_id, "validations": [],
                "ports_published": {}, "materialization": {**asdict(mat), "path": mat.target_path}}

    def run(name):
        return execute_checkpointed_job(job(name), context={}, router=None, registry=PortRegistry(),
                                        base_dir=tmp_path, execute=execute)

    first = run("shardA")
    run("shardB")
    if legacy:
        path_record = _read_record(tmp_path, "paths", relative)
        path_record.pop("java_patch_history", None)
        _write_record(tmp_path, "paths", relative, path_record)
    if change != 'patch':
        if change == 'whole_file':
            run('replacement')
        else:
            target.write_text(target.read_text() + '\n// unreceipted edit')
        with pytest.raises(ArtifactGraphError, match='ARTIFACT_JAVA_STARTUP_SOURCE_UNVERIFIED'):
            _wire_validated_java_startup([SimpleNamespace(canonical_leaf='minecraft/item/registry')],
                [first], context={'project_root': str(tmp_path)}, base_dir=tmp_path)
        assert json.loads(meta.read_text())['entrypoints']['main'] == []
        return
    before = target.read_bytes()
    reused = run("shardA")
    assert reused["reuse"] == "VERIFIED_CHECKPOINT"
    _wire_validated_java_startup([SimpleNamespace(canonical_leaf="minecraft/item/registry")],
        [reused], context={"project_root": str(tmp_path)}, base_dir=tmp_path)
    assert calls == ["shardA", "shardB"]
    assert target.read_bytes() == before
    assert len(json.loads(meta.read_text())["entrypoints"]["main"]) == 1
