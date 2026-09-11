from __future__ import annotations

import json

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_ports import PortRegistry


def test_extended_directory_count_does_not_parse_unrelated_records(tmp_path):
    from minecraft_mod_ai.extended_content_generator import _directory_catalog_count
    records = tmp_path / ".minecraft_ai/extended-module-records"
    records.mkdir(parents=True)
    for index in range(128):
        (records / f"m{index}.json").write_text("{broken", encoding="utf-8")
    (tmp_path / ".minecraft_ai/extended-modules.json").write_text(json.dumps({
        "schema_version": "mmm/extended-module-directory-v1",
        "module_count": 128,
        "directory": ".minecraft_ai/extended-module-records",
    }), encoding="utf-8")
    assert _directory_catalog_count(tmp_path) == 128


def test_single_artifact_graph_avoids_executor_creation(monkeypatch):
    import minecraft_mod_ai.artifact_graph_executor as executor
    job = ArtifactJob(job_id="one", template_id="test/one", owner_module="m")
    monkeypatch.setattr(executor, "_execute_one", lambda job, **_kwargs: {"status": "PASS", "job_id": job.job_id})
    class ForbiddenExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("single artifact graph created a thread pool")
    monkeypatch.setattr(executor, "ThreadPoolExecutor", ForbiddenExecutor)
    result = executor.execute_artifact_graph([job], port_registry=PortRegistry())
    assert result["completed_jobs"] == ["one"]


def test_source_patch_reuses_process_commit_pool(tmp_path, monkeypatch):
    import minecraft_mod_ai.source_patch as source_patch
    monkeypatch.setenv("MMM_SOURCE_PATCH_WORKERS", "2")
    source_patch._COMMIT_POOL = None
    created = 0
    real = source_patch.ThreadPoolExecutor
    class CountingExecutor(real):
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(source_patch, "ThreadPoolExecutor", CountingExecutor)
    try:
        for batch in range(2):
            operations = []
            for index in range(2):
                target = tmp_path / f"{batch}-{index}.txt"
                target.write_text("old", encoding="utf-8")
                operations.append({"operation": "replace", "path": target.name, "expected_sha256": source_patch.sha256_bytes(b"old"), "content": "new"})
            source_patch.TransactionalSourcePatcher(tmp_path).apply(operations)
        assert created == 1
    finally:
        if source_patch._COMMIT_POOL is not None:
            source_patch._COMMIT_POOL.shutdown(wait=True)
        source_patch._COMMIT_POOL = None
