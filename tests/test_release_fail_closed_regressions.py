from __future__ import annotations

from dataclasses import dataclass

import pytest

from minecraft_mod_ai import final_artifact
from minecraft_mod_ai import generation_evidence_controller as evidence


@dataclass(frozen=True)
class _RejectedCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str


def _schema_with_nullable_required(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": None,
            },
        },
    }


def test_rejected_retriever_translation_accepts_nullable_required_schema() -> None:
    rejected = _RejectedCall(
        id="nullable-required",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "search_project_rag",
            "raw_arguments": '{"query":"item registry 26.2","limit":8}',
        },
        raw_arguments="{}",
    )

    normalized = evidence.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(_schema_with_nullable_required("search_code_rag"),),
        forced_evidence_tool="search_code_rag",
    )

    assert normalized is not None
    assert normalized[0].name == "search_code_rag"
    assert normalized[0].arguments == {
        "query": "item registry 26.2",
        "limit": 8,
    }


def _authored_manifest(text: str) -> dict:
    encoded = text.encode("utf-8")
    manifest = {
        "schema_version": "mmm/authored-execution-manifest-v2",
        "source_text_sha256": "sha256:" + __import__("hashlib").sha256(encoded).hexdigest(),
        "source_bytes": len(encoded),
        "unit_count": 1,
        "policy": "host_exact_task_queue_no_coder_file_planning",
        "units": [
            {
                "module_id": "authored_feature_001",
                "path": "src/main/java/demo/AuthoredFeature001.java",
                "symbol": "AuthoredFeature001",
                "start_byte": 0,
                "end_byte": len(encoded),
                "text_sha256": "sha256:" + __import__("hashlib").sha256(encoded).hexdigest(),
                "provides": "authored_feature_001_ready",
                "section": "Economy",
            }
        ],
        "entrypoint": {
            "owner": "host_scaffold",
            "path": "src/main/java/demo/DemoMod.java",
            "symbol": "DemoMod",
            "feature_symbols": ["AuthoredFeature001"],
        },
    }
    manifest["manifest_sha256"] = final_artifact._canonical_sha256(manifest)
    return manifest


def _coverage(tmp_path, text: str) -> dict:
    return final_artifact.build_authored_design_coverage_receipt(
        proposal_hash="sha256:" + "1" * 64,
        requested_prompt="space economy",
        authored_plan={
            "schema_version": "mmm/authored-plan-v1",
            "requested_prompt": "space economy",
            "text": text,
        },
        authored_manifest=_authored_manifest(text),
        module_ids=("authored_feature_001",),
        project_root=tmp_path,
        artifact_sha256="sha256:" + "2" * 64,
        source_validation={"status": "PASS", "checks_run": 1, "findings": []},
        build_report={"status": "PASS"},
        jar_validation={"status": "PASS", "checks_run": 1, "findings": []},
        gametest_passed=True,
        unresolved_gates=(),
    )


def test_saved_authored_release_rejects_noop_initialize(tmp_path) -> None:
    text = "# Economy\nImplement credits and trading.\n"
    source = tmp_path / "src/main/java/demo/AuthoredFeature001.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package demo;\n"
        "public final class AuthoredFeature001 {\n"
        "  public static void initialize() { return; }\n"
        "}\n",
        encoding="utf-8",
    )

    blocked = _coverage(tmp_path, text)

    assert blocked["status"] == "BLOCKED"
    assert any(
        "initialize() has no executable behavior" in finding
        for finding in blocked["findings"]
    )

    source.write_text(
        "package demo;\n"
        "public final class AuthoredFeature001 {\n"
        "  private static int credits;\n"
        "  public static void initialize() { credits = 1; }\n"
        "}\n",
        encoding="utf-8",
    )

    assert _coverage(tmp_path, text)["status"] == "PASS"


def test_download_bundle_rejects_stale_proposal_binding(tmp_path) -> None:
    with pytest.raises(final_artifact.FinalArtifactError, match="proposal"):
        final_artifact.write_downloadable_bundle(
            tmp_path / "bundle",
            artifact_receipt={"status": "PASS"},
            requirement_coverage={
                "status": "PASS",
                "proposal_hash": "sha256:" + "a" * 64,
            },
            reuse_manifest={},
            build_receipt={"status": "PASS"},
            runtime_receipt={"status": "NOT_REQUIRED"},
            proposal_hash="sha256:" + "b" * 64,
        )
