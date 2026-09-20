from __future__ import annotations

import inspect
import json
import zipfile
from pathlib import Path

import pytest
from types import SimpleNamespace

from minecraft_mod_ai.final_artifact import (
    sha256_file,
    write_build_artifact_bundle,
    write_downloadable_bundle,
)
from minecraft_mod_ai.mcp_tools import MMMToolService

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _attach_verified_release_artifact,
    _blocking_jdt_errors,
    _final_validation_failure,
    _gametest_attestation_status,
    _jdt_verification_attempts,
    _jdt_verification_timeout_seconds,
    _retryable_jdt_bootstrap_failure,
    _generation_receipt_sort_key,
    _persisted_runtime_evidence,
    _refresh_runtime_receipt_status,
    _runtime_verification_passed,
    _runtime_visual_download_artifacts,
    _replace_stale_directory_target,
    _replace_stale_file_target,
    _stable_payload_sha256,
    _validate_external_execution_preflight,
    _validate_required_gate_contract,
)


def _jdt_error_receipt() -> dict:
    return {
        "status": "PASS",
        "diagnostics": {
            "file:///src/main/java/demo/Feature.java": [
                {
                    "severity": 1,
                    "source": "jdtls",
                    "code": "compiler.err.cant.resolve.location",
                    "message": "The method missing() is undefined for the type Feature",
                }
            ]
        },
    }


def test_debug_fixture_prepare_binds_generated_token_into_host_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        CompleteProductionOrchestrator,
        "_ensure_debug_fixture_resources",
        staticmethod(lambda _approved, _root: None),
    )
    root = tmp_path / "project"
    package = root / "src/main/java/dev/mmm/debugfixture"
    metadata = root / ".minecraft_ai"
    package.mkdir(parents=True)
    metadata.mkdir(parents=True)

    main_source = package / "MmmDebugFixtureMod.java"
    main_source.write_text(
        """
package dev.mmm.debugfixture;

public final class MmmDebugFixtureMod {
    public void onInitialize() {
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    gametest_source = package / "MmmDebugFixtureModGameTests.java"
    gametest_source.write_text(
        """
package dev.mmm.debugfixture;

public final class MmmDebugFixtureModGameTests {
    public void generatedRegistriesAreLive(Object context) {
        context.succeed();
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (metadata / "fabric-template-receipt.json").write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "main_source": (
                        "src/main/java/dev/mmm/debugfixture/MmmDebugFixtureMod.java"
                    )
                },
                "gametest_contract": {
                    "source": (
                        "src/main/java/dev/mmm/debugfixture/"
                        "MmmDebugFixtureModGameTests.java"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    module = SimpleNamespace(
        module_id="debug_token",
        config={
            "observable_source_contract": {
                "schema_version": "mmm/debug-source-contract-v1",
                "binding_field": "DEBUG_TOKEN",
            }
        },
    )
    proposal = SimpleNamespace(
        schema_version="mmm/complete-proposal-v1",
        game_design={
            "mode": "debug_fixture",
            "fixture": {"module_id": "debug_token"},
        },
        modules=(module,),
        base_proposal=SimpleNamespace(
            spec=SimpleNamespace(
                package_name="dev.mmm.debugfixture",
                mod_id="mmm_debug_fixture",
            )
        ),
    )

    bound = CompleteProductionOrchestrator._bind_debug_fixture_runtime(
        proposal,
        root,
    )

    assert bound == root.resolve()
    main_text = main_source.read_text(encoding="utf-8")
    gametest_text = gametest_source.read_text(encoding="utf-8")
    assert "MMM_DEBUG_FIXTURE_RUNTIME_BINDING" in main_text
    assert 'Class.forName("dev.mmm.debugfixture.DebugToken", true,' in main_text
    assert 'getField("DEBUG_TOKEN").get(null) == null' in main_text
    assert "MMM_DEBUG_FIXTURE_REGISTRY_ASSERTION" in gametest_text
    assert 'Class.forName("dev.mmm.debugfixture.DebugToken", true,' in gametest_text
    assert 'getField("DEBUG_TOKEN").get(null) == null' in gametest_text

    CompleteProductionOrchestrator._bind_debug_fixture_runtime(proposal, root)
    assert (
        main_source.read_text(encoding="utf-8").count(
            "MMM_DEBUG_FIXTURE_RUNTIME_BINDING"
        )
        == 1
    )
    assert (
        gametest_source.read_text(encoding="utf-8").count(
            "MMM_DEBUG_FIXTURE_REGISTRY_ASSERTION"
        )
        == 1
    )


def test_final_source_validation_failure_cannot_be_marked_pass() -> None:
    assert _final_validation_failure(
        source_report={"status": "FAIL"},
        jdt_receipt=None,
        run_jdt=False,
    ) == "Final project failed deterministic source validation after build/repair."


def test_final_jdt_source_error_blocks_success_even_after_gradle_build() -> None:
    receipt = _jdt_error_receipt()

    assert _blocking_jdt_errors(receipt)
    assert _final_validation_failure(
        source_report={"status": "PASS"},
        jdt_receipt=receipt,
        run_jdt=True,
    ) == "Final JDT validation still reports source errors after build/repair."


def test_jdt_infrastructure_unavailable_is_not_misclassified_as_source_error() -> None:
    receipt = {
        "status": "UNAVAILABLE",
        "error": "RuntimeError: JDT workspace unavailable",
        "diagnostics": {},
    }

    assert _blocking_jdt_errors(receipt) == []
    assert _final_validation_failure(
        source_report={"status": "PASS"},
        jdt_receipt=receipt,
        run_jdt=True,
    ) is None


def test_jdt_receipt_is_ignored_when_jdt_was_not_requested() -> None:
    assert _final_validation_failure(
        source_report={"status": "PASS"},
        jdt_receipt=_jdt_error_receipt(),
        run_jdt=False,
    ) is None


def test_requested_gametest_requires_structured_matching_evidence(tmp_path) -> None:
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="DemoModGameTests.generatedRegistriesAreLive"/>'
        '</testsuite></testsuites>',
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "commands": [
            {"name": "build", "exit_code": 0, "timed_out": False},
            {"name": "gametest", "exit_code": 0, "timed_out": False},
        ],
        "gametest_report": str(report),
    }

    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "PASS"


def test_incremental_full_build_receipt_is_release_grade_evidence(
    tmp_path,
) -> None:
    jar = tmp_path / "demo.jar"
    jar.write_bytes(b"jar")
    build = {
        "status": "PASS",
        "jar_path": str(jar),
        "commands": [
            {
                "name": "incremental_build",
                "command": [
                    "/opt/gradle/bin/gradle",
                    "--daemon",
                    "--parallel",
                    "--max-workers=4",
                    "build",
                    "--build-cache",
                    "--stacktrace",
                ],
                "exit_code": 0,
                "timed_out": False,
            }
        ],
        "gametest_report": None,
    }

    assert CompleteProductionOrchestrator._full_gradle_build_receipt_passed(build)
    assert CompleteProductionOrchestrator._cached_build_exists(build)


def test_incremental_receipt_without_gradle_build_task_is_not_release_grade() -> None:
    build = {
        "status": "PASS",
        "commands": [
            {
                "name": "incremental_build",
                "command": ["gradle", "compileJava", "--stacktrace"],
                "exit_code": 0,
                "timed_out": False,
            }
        ],
    }

    assert not CompleteProductionOrchestrator._full_gradle_build_receipt_passed(build)


def test_integrated_gametest_accepts_incremental_build_receipt(tmp_path) -> None:
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="DemoModGameTests.generatedRegistriesAreLive"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "gametest_mode": "integrated_build",
        "gametest_task": "runGameTest",
        "commands": [
            {
                "name": "incremental_build",
                "command": ["gradle", "build", "--build-cache", "--stacktrace"],
                "exit_code": 0,
                "timed_out": False,
            }
        ],
        "gametest_report": str(report),
    }

    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "PASS"


def test_incremental_integrated_build_receipt_matches_runtime_log_shape(
    tmp_path,
) -> None:
    jar = tmp_path / "demo.jar"
    jar.write_bytes(b"jar")
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="DemoModGameTests.generatedRegistriesAreLive"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "jar_path": str(jar),
        "gametest_mode": "integrated_build",
        "gametest_task": "runGameTest",
        "gametest_report": str(report),
        "commands": [
            {
                "name": "incremental_build",
                "command": [
                    "/root/.cache/mmm/gradle/bin/gradle",
                    "--daemon",
                    "--parallel",
                    "--max-workers=4",
                    "build",
                    "--build-cache",
                    "--stacktrace",
                ],
                "exit_code": 0,
                "timed_out": False,
            }
        ],
    }

    assert CompleteProductionOrchestrator._full_gradle_build_receipt_passed(build)
    assert CompleteProductionOrchestrator._cached_build_exists(
        build,
        require_gametest=True,
        spec=SimpleNamespace(mod_id="demo"),
    )
    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "PASS"


def test_requested_gametest_without_report_is_not_attested() -> None:
    build = {
        "status": "PASS",
        "commands": [{"name": "gametest", "exit_code": 0, "timed_out": False}],
        "gametest_report": None,
    }

    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "NO_EVIDENCE"
    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=False,
    ) == "NOT_REQUIRED"


def test_execute_wires_required_gate_failures_into_both_release_paths() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert source.count("self._required_gate_failures(") == 2
    assert "build_report=None" in source
    assert "build_report=build" in source
    assert source.count("unresolved.extend(") >= 2


def test_optional_runtime_checks_do_not_create_unresolved_release_gates() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert (
        "if approved.external_runtime_required:\n"
        "                    unresolved.append('runtime:not-requested')"
    ) in source
    assert (
        "if approved.external_runtime_required:\n"
        "                    unresolved.append('mineflayer:not-requested')"
    ) in source
    assert (
        "if approved.external_runtime_required:\n"
        "                    unresolved.append('visual-review:not-requested')"
    ) in source


def test_cached_build_requires_real_build_command_and_requested_gametest(tmp_path) -> None:
    jar = tmp_path / "demo.jar"
    jar.write_bytes(b"jar")
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="DemoModGameTests.generatedRegistriesAreLive"/>'
        '</testsuite></testsuites>',
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "jar_path": str(jar),
        "commands": [
            {"name": "build", "exit_code": 0, "timed_out": False},
            {"name": "gametest", "exit_code": 0, "timed_out": False},
        ],
        "gametest_report": str(report),
    }

    assert CompleteProductionOrchestrator._cached_build_exists(build)
    assert CompleteProductionOrchestrator._cached_build_exists(
        build,
        require_gametest=True,
        spec=SimpleNamespace(mod_id="demo"),
    )

    no_build_receipt = {**build, "commands": build["commands"][1:]}
    assert not CompleteProductionOrchestrator._cached_build_exists(no_build_receipt)

    missing_gametest = {**build, "gametest_report": None}
    assert not CompleteProductionOrchestrator._cached_build_exists(
        missing_gametest,
        require_gametest=True,
        spec=SimpleNamespace(mod_id="demo"),
    )


def test_required_runtime_needs_live_client_playtest_and_visual_evidence(tmp_path) -> None:
    artifact_sha = "sha256:" + "a" * 64
    evidence = tmp_path / "runtime-proof.png"
    evidence.write_bytes(b"runtime-proof")
    runtime = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "server": {"server_running": True},
        "client": {"client_running": True},
    }
    playtest = {
        "status": "PASS",
        "interaction_count": 1,
        "assertion_count": 1,
        "artifact_sha256": artifact_sha,
    }
    visual = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "runtime_screenshots": [
            {
                "sha256": CompleteProductionOrchestrator._file_hash(evidence),
                "evidence_path": str(evidence),
                "server_running": True,
                "client_running": True,
            }
        ],
    }

    assert _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt=visual,
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt={**runtime, "client": None},
        playtest_receipt=playtest,
        visual_receipt=visual,
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt={**playtest, "assertion_count": 0},
        visual_receipt=visual,
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt={**visual, "status": "FAIL"},
    )


def test_optional_runtime_is_not_required_for_release_readiness() -> None:
    assert _runtime_verification_passed(
        required=False,
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )


def test_required_runtime_missing_receipt_is_not_recorded_as_not_required() -> None:
    missing = _persisted_runtime_evidence(
        None,
        required=True,
        artifact_sha256="sha256:" + "a" * 64,
    )
    optional = _persisted_runtime_evidence(
        None,
        required=False,
        artifact_sha256="sha256:" + "b" * 64,
    )

    assert missing["status"] == "REQUIRED_NOT_RUN"
    assert optional["status"] == "NOT_REQUIRED"


def test_package_cache_requires_matching_file_digest(tmp_path) -> None:
    package = tmp_path / "bundle.zip"
    package.write_bytes(b"first")
    receipt = {
        "status": "PACKAGED",
        "path": str(package),
        "sha256": CompleteProductionOrchestrator._file_hash(package),
    }

    assert CompleteProductionOrchestrator._cached_package_exists(
        receipt,
        path_key="path",
    )

    package.write_bytes(b"tampered")
    assert not CompleteProductionOrchestrator._cached_package_exists(
        receipt,
        path_key="path",
    )


def test_release_package_fingerprint_changes_with_evidence() -> None:
    base = {
        "jar_sha256": "sha256:" + "a" * 64,
        "coverage_sha256": "sha256:" + "b" * 64,
    }
    changed = {
        **base,
        "coverage_sha256": "sha256:" + "c" * 64,
    }

    assert _stable_payload_sha256(base) != _stable_payload_sha256(changed)


def test_execute_checkpoints_distribution_packaging_for_resume() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert "'package-distribution'" in source
    assert "distribution-bundle-" in source
    assert "release_package_input" in source
    assert "runtime_receipt_sha256" in source
    assert "coverage_sha256" in source


def test_cached_download_bundle_validates_every_member_digest(tmp_path) -> None:
    root = tmp_path / "download"
    root.mkdir()
    jar = root / "demo.jar"
    jar.write_bytes(b"jar")
    members = [
        {
            "path": "demo.jar",
            "sha256": CompleteProductionOrchestrator._file_hash(jar),
        }
    ]
    receipt = {
        "status": "PASS",
        "path": str(root),
        "artifact_sha256": "sha256:" + "a" * 64,
        "members": members,
        "manifest_sha256": "sha256:" + "b" * 64,
    }
    (root / "bundle-receipt.json").write_text(
        __import__("json").dumps(
            {
                "status": "PASS",
                "artifact_sha256": receipt["artifact_sha256"],
                "members": members,
                "manifest_sha256": receipt["manifest_sha256"],
            }
        ),
        encoding="utf-8",
    )

    assert CompleteProductionOrchestrator._cached_download_bundle_exists(receipt)

    jar.write_bytes(b"tampered")
    assert not CompleteProductionOrchestrator._cached_download_bundle_exists(receipt)


def test_execute_checkpoints_publish_and_download_side_effects() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert "'publish-' + provider" in source
    assert "'package-downloadable'" in source
    assert "final-mod-download-" in source
    assert "validate_cached=self._cached_download_bundle_exists" in source


def test_external_runtime_preflight_fails_before_generation_on_missing_inputs() -> None:
    proposal = SimpleNamespace(external_runtime_required=True)
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        eula_accepted=False,
        server_launcher=None,
        playtest_actions=(),
        screenshot_paths=(),
    )

    with pytest.raises(Exception, match="external runtime verification"):
        _validate_external_execution_preflight(proposal, options)


def test_external_runtime_preflight_allows_optional_checks_to_be_disabled() -> None:
    proposal = SimpleNamespace(external_runtime_required=False)
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        eula_accepted=False,
        server_launcher=None,
        playtest_actions=(),
        screenshot_paths=(),
    )

    _validate_external_execution_preflight(proposal, options)


def test_source_only_skips_external_runtime_preflight() -> None:
    _validate_external_execution_preflight(
        SimpleNamespace(external_runtime_required=True),
        SimpleNamespace(source_only=True),
    )


def test_source_only_package_cache_rejects_tampered_zip(tmp_path) -> None:
    archive = tmp_path / "complete-source.zip"
    archive.write_bytes(b"source-zip")
    receipt = CompleteProductionOrchestrator._source_package_receipt(str(archive))

    assert CompleteProductionOrchestrator._cached_package_exists(
        receipt,
        path_key="release_zip",
    )

    archive.write_bytes(b"tampered")
    assert not CompleteProductionOrchestrator._cached_package_exists(
        receipt,
        path_key="release_zip",
    )


def test_runtime_receipt_is_refreshed_with_terminal_liveness() -> None:
    initial = {
        "status": "PASS",
        "server": {"schema_version": "mmm/runtime-status-v1", "server_running": True},
        "client": {"schema_version": "mmm/runtime-status-v1", "client_running": True},
    }

    dead_client = _refresh_runtime_receipt_status(
        initial,
        {
            "server_running": True,
            "client_running": False,
            "server_log_lines": 12,
            "client_log_lines": 3,
        },
        require_client=True,
    )
    assert dead_client["status"] == "FAIL"
    assert dead_client["server"]["server_running"] is True
    assert dead_client["client"]["client_running"] is False

    server_only = _refresh_runtime_receipt_status(
        {**initial, "client": None},
        {"server_running": True, "client_running": False, "server_log_lines": 12},
        require_client=False,
    )
    assert server_only["status"] == "PASS"


def test_checkpoint_miss_replaces_stale_package_targets(tmp_path) -> None:
    stale_file = tmp_path / "package.zip"
    stale_file.write_bytes(b"partial")
    assert _replace_stale_file_target(
        stale_file,
        lambda: (stale_file.write_bytes(b"rebuilt"), "file-ok")[1],
    ) == "file-ok"
    assert stale_file.read_bytes() == b"rebuilt"

    stale_dir = tmp_path / "download"
    stale_dir.mkdir()
    (stale_dir / "partial").write_text("partial", encoding="utf-8")

    def rebuild_dir():
        stale_dir.mkdir()
        (stale_dir / "complete").write_text("ok", encoding="utf-8")
        return "dir-ok"

    assert _replace_stale_directory_target(stale_dir, rebuild_dir) == "dir-ok"
    assert not (stale_dir / "partial").exists()
    assert (stale_dir / "complete").read_text(encoding="utf-8") == "ok"


def test_generation_executor_quiesces_mutating_workers_before_return() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator._execute_generation_work)

    assert "run_with_model_execution_deadline" in source
    assert "time.monotonic() + lease_seconds" in source
    assert "shutdown(wait=False" not in source
    assert source.count("shutdown(wait=True, cancel_futures=True)") == 5


def test_asset_shard_cache_validates_asset_and_document_digests(tmp_path) -> None:
    texture = tmp_path / "texture.png"
    document = tmp_path / "model.json"
    texture.write_bytes(b"png")
    document.write_text("{}", encoding="utf-8")
    receipt = {
        "status": "TEXTURE_PRODUCTION_PASS",
        "assets": [
            {
                "target": str(texture),
                "sha256": CompleteProductionOrchestrator._file_hash(texture),
            }
        ],
        "documents": [
            {
                "resolved_path": str(document),
                "sha256": CompleteProductionOrchestrator._file_hash(document),
            }
        ],
        "resource_graph_validation": {"status": "PASS"},
        "resource_contract_validation": {"status": "PASS"},
    }

    assert CompleteProductionOrchestrator._cached_asset_shard(receipt)

    document.write_text('{"changed": true}', encoding="utf-8")
    assert not CompleteProductionOrchestrator._cached_asset_shard(receipt)


def test_unsupported_required_gate_fails_before_generation() -> None:
    proposal = SimpleNamespace(
        modules=(
            SimpleNamespace(
                module_id="broken_gate",
                required_gates=("Imaginary verifier that does not exist",),
            ),
        ),
    )

    with pytest.raises(Exception, match="unsupported required gates"):
        _validate_required_gate_contract(proposal)


def test_supported_required_gate_contract_is_accepted() -> None:
    proposal = SimpleNamespace(
        modules=(
            SimpleNamespace(
                module_id="entity",
                required_gates=(
                    "JDT diagnostics",
                    "Gradle clean build",
                    "Blockbench UV and bone hierarchy review",
                    "runtime animation review",
                ),
            ),
        ),
    )

    _validate_required_gate_contract(proposal)



def test_resource_pack_bundle_is_finalized_deterministically(tmp_path) -> None:
    run_root = tmp_path / "run"
    pack_root = run_root / "resource-pack"
    (pack_root / "assets/demo/textures").mkdir(parents=True)
    (pack_root / "pack.mcmeta").write_text('{"pack":{"pack_format":1}}\n', encoding="utf-8")
    (pack_root / "assets/demo/textures/item.txt").write_text("payload", encoding="utf-8")
    shard = {
        "container_validation": {
            "standalone_resource_pack": {
                "status": "PASS",
                "root": str(pack_root),
            }
        }
    }

    first = CompleteProductionOrchestrator._finalize_resource_pack_bundle(
        [shard],
        run_root=run_root,
    )
    second = CompleteProductionOrchestrator._finalize_resource_pack_bundle(
        [shard],
        run_root=run_root,
    )

    assert first is not None and second is not None
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == 2
    with zipfile.ZipFile(first["path"]) as archive:
        assert archive.namelist() == [
            "assets/demo/textures/item.txt",
            "pack.mcmeta",
        ]


def test_downloadable_bundle_keeps_verified_additional_resource_pack(tmp_path) -> None:
    jar = tmp_path / "demo.jar"
    jar.write_bytes(b"jar")
    pack = tmp_path / "resource-pack.zip"
    pack.write_bytes(b"pack")
    artifact_sha = CompleteProductionOrchestrator._file_hash(jar)
    pack_sha = CompleteProductionOrchestrator._file_hash(pack)

    bundle = write_downloadable_bundle(
        tmp_path / "download",
        artifact_receipt={
            "status": "PASS",
            "artifact_path": str(jar),
            "sha256": artifact_sha,
        },
        requirement_coverage={
            "status": "PASS",
            "artifact_sha256": artifact_sha,
        },
        reuse_manifest={},
        build_receipt={
            "status": "PASS",
            "artifact_sha256": artifact_sha,
        },
        runtime_receipt={
            "status": "NOT_REQUIRED",
            "artifact_sha256": artifact_sha,
        },
        additional_artifacts={
            "generated-resource-pack.zip": {
                "path": str(pack),
                "sha256": pack_sha,
            }
        },
    )

    assert bundle["additional_artifacts"] == {
        "generated-resource-pack.zip": pack_sha
    }
    assert (tmp_path / "download" / "generated-resource-pack.zip").read_bytes() == b"pack"
    assert any(
        item["path"] == "generated-resource-pack.zip"
        and item["sha256"] == pack_sha
        for item in bundle["members"]
    )


def test_primary_release_zip_keeps_mcp_authority_narrow_and_attaches_privately(tmp_path) -> None:
    assert "additional_artifacts" not in inspect.signature(
        MMMToolService.package_release
    ).parameters

    run_root = tmp_path / "run"
    run_root.mkdir()
    release_zip = run_root / "release.zip"
    with zipfile.ZipFile(release_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source/demo.txt", "source")
        archive.writestr(
            "release-manifest.json",
            '{"schema_version":"mmm/release-manifest-v2"}',
        )
    pack = run_root / "generated-resource-pack.zip"
    pack.write_bytes(b"pack")
    release = {
        "status": "PACKAGED",
        "release_zip": str(release_zip),
        "sha256": CompleteProductionOrchestrator._file_hash(release_zip),
    }
    pack_sha = CompleteProductionOrchestrator._file_hash(pack)

    updated = _attach_verified_release_artifact(
        release,
        {"path": str(pack), "sha256": pack_sha},
        archive_name="generated-resource-pack.zip",
        allowed_root=run_root,
    )

    assert updated["sha256"] == CompleteProductionOrchestrator._file_hash(release_zip)
    assert updated["additional_artifacts"] == {
        "generated-resource-pack.zip": pack_sha
    }
    with zipfile.ZipFile(release_zip) as archive:
        assert archive.read("additional/generated-resource-pack.zip") == b"pack"
        manifest = __import__("json").loads(
            archive.read("release-manifest.json").decode("utf-8")
        )
    assert manifest["additional_artifacts"] == {
        "generated-resource-pack.zip": pack_sha
    }


def test_release_manifest_rebinds_complete_provenance_without_attachment(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    release_zip = run_root / "release.zip"
    with zipfile.ZipFile(release_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source/demo.txt", "source")
        archive.writestr(
            "release-manifest.json",
            json.dumps(
                {
                    "schema_version": "mmm/release-manifest-v2",
                    "proposal_hash": "sha256:base",
                }
            ),
        )
    release = {
        "status": "PACKAGED",
        "release_zip": str(release_zip),
        "sha256": CompleteProductionOrchestrator._file_hash(release_zip),
    }

    updated = _attach_verified_release_artifact(
        release,
        None,
        archive_name="generated-resource-pack.zip",
        allowed_root=run_root,
        manifest_provenance={
            "proposal_hash": "sha256:complete",
            "base_proposal_hash": "sha256:base",
            "proposal_scope": "complete",
        },
    )

    assert updated["sha256"] == CompleteProductionOrchestrator._file_hash(release_zip)
    assert updated["manifest_provenance"]["proposal_hash"] == "sha256:complete"
    with zipfile.ZipFile(release_zip) as archive:
        manifest = json.loads(archive.read("release-manifest.json"))
        assert "additional/generated-resource-pack.zip" not in archive.namelist()
    assert manifest["proposal_hash"] == "sha256:complete"
    assert manifest["base_proposal_hash"] == "sha256:base"
    assert manifest["proposal_scope"] == "complete"


def test_blockbench_checkpoint_dependency_tracks_geometry_digest(tmp_path) -> None:
    geo = tmp_path / "entity.geo.json"
    geo.write_text('{"minecraft:geometry":[]}', encoding="utf-8")
    receipt = {"files": [str(geo)]}

    first = CompleteProductionOrchestrator._blockbench_geometry_sha256(receipt)
    geo.write_text('{"minecraft:geometry":[{"description":{}}]}', encoding="utf-8")
    second = CompleteProductionOrchestrator._blockbench_geometry_sha256(receipt)

    assert first != second


def test_parallel_generation_receipts_sort_deterministically() -> None:
    receipts = [
        {"schema_version": "z", "module_id": "b", "value": 1},
        {"schema_version": "a", "module_id": "a", "value": 2},
        {"schema_version": "b", "module_id": "a", "value": 3},
    ]

    forward = sorted(receipts, key=_generation_receipt_sort_key)
    reverse = sorted(reversed(receipts), key=_generation_receipt_sort_key)

    assert forward == reverse
    assert [item["module_id"] for item in forward] == ["a", "a", "b"]


def test_runtime_visual_evidence_rejects_unbound_or_stale_receipts(tmp_path) -> None:
    artifact_sha = "sha256:" + "c" * 64
    evidence = tmp_path / "visual-evidence.png"
    evidence.write_bytes(b"visual-evidence")
    runtime = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "server": {"server_running": True},
        "client": {"client_running": True},
    }
    playtest = {
        "status": "PASS",
        "interaction_count": 1,
        "assertion_count": 1,
        "artifact_sha256": artifact_sha,
    }
    screenshot = {
        "sha256": CompleteProductionOrchestrator._file_hash(evidence),
        "evidence_path": str(evidence),
        "server_running": True,
        "client_running": True,
    }
    base_visual = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "runtime_screenshots": [screenshot],
    }

    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt={**base_visual, "artifact_sha256": "sha256:" + "e" * 64},
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt={**base_visual, "runtime_screenshots": []},
    )
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt={
            **base_visual,
            "runtime_screenshots": [{**screenshot, "client_running": False}],
        },
    )

    evidence.write_bytes(b"tampered-after-review")
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt=base_visual,
    )


def test_full_entity_preflight_requires_blockbench_even_if_plan_omits_gate() -> None:
    proposal = SimpleNamespace(
        external_runtime_required=False,
        acceptance_tests=(),
        modules=(SimpleNamespace(kind="entity"),),
    )
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        run_blockbench=False,
        eula_accepted=False,
        server_launcher=None,
        playtest_actions=(),
        screenshot_paths=(),
    )

    with pytest.raises(Exception, match="requires Blockbench"):
        _validate_external_execution_preflight(proposal, options)


def test_optional_client_cannot_run_without_runtime() -> None:
    proposal = SimpleNamespace(
        external_runtime_required=False,
        acceptance_tests=(),
        modules=(),
    )
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=True,
        run_mineflayer=False,
        run_visual_review=False,
        run_blockbench=True,
        eula_accepted=False,
        server_launcher=None,
        playtest_actions=(),
        screenshot_paths=(),
    )

    with pytest.raises(Exception, match="Client verification requires runtime"):
        _validate_external_execution_preflight(proposal, options)


def test_host_required_blockbench_review_is_independent_of_plan_gate(tmp_path) -> None:
    preview = tmp_path / "entity-preview.png"
    preview.write_bytes(b"preview")
    proposal = SimpleNamespace(
        modules=(
            SimpleNamespace(module_id="dragon", kind="boss", required_gates=()),
        ),
    )
    receipt = {
        "entity": "dragon",
        "uv": {"status": "PASS"},
        "preview": str(preview),
        "preview_sha256": CompleteProductionOrchestrator._file_hash(preview),
    }

    assert CompleteProductionOrchestrator._mandatory_blockbench_failures(
        proposal,
        [receipt],
    ) == []
    assert CompleteProductionOrchestrator._mandatory_blockbench_failures(
        proposal,
        [],
    ) == ["blockbench:dragon:missing-host-required-review"]


def test_runtime_verification_rejects_playtest_from_other_artifact(tmp_path) -> None:
    artifact_sha = "sha256:" + "1" * 64
    evidence = tmp_path / "proof.png"
    evidence.write_bytes(b"proof")
    runtime = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "prepared": {"instance_root": "/runtime/current"},
        "server": {"server_running": True},
        "client": {"client_running": True},
    }
    visual = {
        "status": "PASS",
        "artifact_sha256": artifact_sha,
        "runtime_screenshots": [
            {
                "sha256": CompleteProductionOrchestrator._file_hash(evidence),
                "evidence_path": str(evidence),
                "server_running": True,
                "client_running": True,
            }
        ],
    }
    playtest = {
        "status": "PASS",
        "interaction_count": 1,
        "assertion_count": 1,
        "artifact_sha256": "sha256:" + "2" * 64,
        "runtime_instance_root": "/runtime/current",
    }

    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt=visual,
    )

    playtest["artifact_sha256"] = artifact_sha
    playtest["runtime_instance_root"] = "/runtime/other"
    assert not _runtime_verification_passed(
        required=True,
        runtime_receipt=runtime,
        playtest_receipt=playtest,
        visual_receipt=visual,
    )


def test_runtime_visual_download_artifacts_preserve_verified_screenshots(tmp_path) -> None:
    evidence = tmp_path / "proof.png"
    evidence.write_bytes(b"proof")
    digest = CompleteProductionOrchestrator._file_hash(evidence)
    artifacts = _runtime_visual_download_artifacts(
        {
            "status": "PASS",
            "runtime_screenshots": [
                {
                    "evidence_path": str(evidence),
                    "sha256": digest,
                }
            ],
        }
    )

    assert artifacts == {
        "runtime-screenshot-001.png": {
            "path": str(evidence),
            "sha256": digest,
        }
    }


def test_validate_jar_checkpoint_uses_versioned_pass_only_cache_policy() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    assert "validation_checkpoint_input(\n                'validate-jar'" in source
    assert "cached_validation_is_reusable('validate-jar', cached)" in source
    assert "validate_cached=lambda _cached: jar_path.is_file()" not in source


def test_release_readiness_is_decided_before_packaging() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator.execute)

    readiness = source.index("release_ready = (")
    package = source.index("tool_service.package_release(")
    distribution = source.index("package_distribution_bundle(")

    assert readiness < package < distribution
    assert "release_zip: str | None = None" in source
    assert "distribution_receipt: dict[str, Any] | None = None" in source


def test_generic_native_gametest_name_is_not_release_attestation(tmp_path) -> None:
    report = tmp_path / "gametest-report.xml"
    report.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase name="minecraft:generatedregistriesarelive"/>'
        '<testcase name="minecraft:other_required_test"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    build = {
        "status": "PASS",
        "gametest_mode": "integrated_build",
        "gametest_task": "runGameTest",
        "commands": [
            {
                "name": "incremental_build",
                "command": ["gradle", "build", "--build-cache", "--stacktrace"],
                "exit_code": 0,
                "timed_out": False,
            }
        ],
        "gametest_report": str(report),
    }

    assert _gametest_attestation_status(
        build,
        SimpleNamespace(mod_id="demo"),
        requested=True,
    ) == "NO_EVIDENCE"


def test_jdt_verification_timeout_defaults_to_colab_safe_window(monkeypatch) -> None:
    monkeypatch.delenv("MMM_JDT_VERIFICATION_TIMEOUT_SECONDS", raising=False)
    assert _jdt_verification_timeout_seconds() == 180


def test_jdt_verification_timeout_is_bounded_and_configurable(monkeypatch) -> None:
    monkeypatch.setenv("MMM_JDT_VERIFICATION_TIMEOUT_SECONDS", "240")
    assert _jdt_verification_timeout_seconds() == 240
    monkeypatch.setenv("MMM_JDT_VERIFICATION_TIMEOUT_SECONDS", "5")
    assert _jdt_verification_timeout_seconds() == 30
    monkeypatch.setenv("MMM_JDT_VERIFICATION_TIMEOUT_SECONDS", "9999")
    assert _jdt_verification_timeout_seconds() == 600
    monkeypatch.setenv("MMM_JDT_VERIFICATION_TIMEOUT_SECONDS", "bad")
    assert _jdt_verification_timeout_seconds() == 180


def test_build_artifact_bundle_preserves_unresolved_release_state(tmp_path) -> None:
    jar = tmp_path / "demo.jar"
    jar.write_bytes(b"jar")
    digest = sha256_file(jar)
    bundle = write_build_artifact_bundle(
        tmp_path / "build-artifact.zip",
        artifact_receipt={
            "status": "PASS",
            "artifact": jar.name,
            "artifact_path": str(jar),
            "sha256": digest,
        },
        build_receipt={"status": "PASS", "artifact_sha256": digest},
        unresolved_gates=("execution-gate:jdt:missing-jdt",),
        release_ready=False,
        proposal_hash="sha256:" + "1" * 64,
        receipts={
            "jdt-receipt.json": {"status": "UNAVAILABLE", "diagnostics": {}}
        },
    )
    assert Path(bundle["build_bundle_zip"]).is_file()
    assert bundle["release_ready"] is False
    assert bundle["unresolved_gates"] == ["execution-gate:jdt:missing-jdt"]
    with zipfile.ZipFile(bundle["build_bundle_zip"], "r") as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("build-manifest.json"))
    assert "artifact/demo.jar" in names
    assert "receipts/build-receipt.json" in names
    assert "receipts/jdt-receipt.json" in names
    assert manifest["release_certified"] is False


def test_jdt_verification_attempts_default_and_bounds(monkeypatch) -> None:
    monkeypatch.delenv("MMM_JDT_VERIFICATION_ATTEMPTS", raising=False)
    assert _jdt_verification_attempts() == 2
    monkeypatch.setenv("MMM_JDT_VERIFICATION_ATTEMPTS", "99")
    assert _jdt_verification_attempts() == 3


def test_only_service_ready_bootstrap_miss_is_retryable() -> None:
    assert _retryable_jdt_bootstrap_failure({
        "status": "UNAVAILABLE",
        "error": "JDTWorkspaceBootstrapError: ServiceReady was not observed before validation",
        "diagnostics": {},
    })
    assert not _retryable_jdt_bootstrap_failure({
        "status": "UNAVAILABLE",
        "error": "JDTWorkspaceBootstrapError: no project JDK matching Java 25",
        "diagnostics": {},
    })
