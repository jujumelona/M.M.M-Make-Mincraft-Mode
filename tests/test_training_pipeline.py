import json
from pathlib import Path

from minecraft_mod_ai.training import TrainingTraceStore


def _trace() -> dict:
    return {
        "trace_id": "",
        "task": "repair_compile",
        "prompt": "Fix the Fabric 1.20.1 item registration error.",
        "response": "Apply the minimal Registry.register patch.",
        "patch": "@@ -1 +1 @@",
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "java_version": 17,
        "mappings": "1.20.1+build.1",
        "fabric_api": "0.92.11+1.20.1",
        "source_license": "Apache-2.0",
        "source_commit": "deadbeef",
        "gradle_exit_code": 0,
        "diagnostics_error_count": 0,
        "request_fidelity_passed": True,
        "gametest_passed": True,
        "jar_validation_passed": True,
        "registry_references_valid": True,
        "requested_feature_deleted": False,
        "cross_loader_api": False,
        "wrong_version_symbol": False,
        "approval_scope_escape": False,
    }


def test_verified_trace_records_and_exports_sft(tmp_path: Path) -> None:
    store = TrainingTraceStore(tmp_path / "traces")
    recorded = store.record(_trace())
    assert recorded["reward"] > 0
    output = tmp_path / "data.jsonl"
    exported = store.export_sft(output)
    assert exported["records"] == 1
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["metadata"]["minecraft_version"] == "1.20.1"
    assert record["messages"][-1]["role"] == "assistant"
