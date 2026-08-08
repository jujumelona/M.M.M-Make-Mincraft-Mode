from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from minecraft_mod_ai.production_contract import (
    ProductionContractError,
    compile_production_contract,
    evaluate_quality_contract,
)
from minecraft_mod_ai.quality_evidence import compile_quality_evidence


PROPOSAL_HASH = "sha256:" + "a" * 64


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _research_design() -> dict:
    return {
        "title": "Evidence fixture",
        "_technology_radar": {
            "schema_version": "mmm/technology-radar-page-v1",
            "aggregate_schema_version": "mmm/technology-radar-aggregate-v1",
            "radar_sha256": _digest("b"),
            "requirements": [{"requirement_id": "fabric-target"}],
            "pagination": {
                "offset": 0,
                "page_size": 50,
                "returned": 1,
                "total_requirements": 1,
                "next_cursor": "",
                "pages_collected": 1,
                "complete": True,
            },
            "collection_receipt": {
                "schema_version": "mmm/technology-page-collection-receipt-v1",
                "page_count": 1,
                "pages_sha256": _digest("c"),
            },
        },
        "_ecosystem_discovery": {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "aggregate_schema_version": "mmm/ecosystem-seed-aggregate-v1",
            "status": "available",
            "route_sha256": _digest("d"),
            "route_count": 2,
            "processed_route_count": 2,
            "remaining_route_count": 0,
            "next_route_cursor": "",
            "routes_complete": True,
            "errors": [],
            "collection_receipt": {
                "schema_version": "mmm/ecosystem-route-collection-receipt-v1",
                "route_page_count": 1,
                "route_pages_sha256": _digest("e"),
            },
        },
        "_technical_evidence": {
            "schema_version": "mmm/central-evidence-graph-v1",
            "evidence_sha256": _digest("f"),
            "domains": [{"domain_id": "fabric"}],
            "unresolved_official_domains": [],
        },
    }


def _module(module_id: str, kind: str = "item") -> dict:
    return {
        "module_id": module_id,
        "kind": kind,
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }


def _compile(
    design: dict,
    *,
    prompt: str = "Add one independently tested item.",
    modules: list[dict] | None = None,
    assets: list[dict] | None = None,
    audio: list[dict] | None = None,
):
    return compile_production_contract(
        requested_prompt=prompt,
        game_design=design,
        research_brief=None,
        modules=modules or [_module("tested_item")],
        assets=assets or [],
        audio=audio or [],
        acceptance_tests=[],
    )


def _baseline_inputs(tmp_path: Path) -> dict:
    report = tmp_path / "gametest.xml"
    report.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase name="registry"/><testcase name="behavior"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    jar = tmp_path / "mod.jar"
    jar.write_bytes(b"independently validated jar")
    return {
        "source_validation": {
            "status": "PASS",
            "checks_run": 18,
            "findings": [],
            "observed_at": "2026-08-02T01:00:00+09:00",
        },
        "build_report": {
            "status": "PASS",
            "gradle_version": "8.5",
            "commands": [
                {"name": "clean_build", "exit_code": 0, "timed_out": False},
                {"name": "gametest", "exit_code": 0, "timed_out": False},
            ],
            "jar_path": str(jar),
            "gametest_report": str(report),
            "observed_at": "2026-08-02T01:01:00+09:00",
        },
        "jar_validation": {
            "status": "PASS",
            "checks_run": 12,
            "findings": [],
        },
        "runtime_receipt": {
            "prepared": {
                "schema_version": "mmm/runtime-instance-v1",
                "minecraft_version": "1.20.1",
                "disposable": True,
            },
            "server": {
                "schema_version": "mmm/runtime-status-v1",
                "server_running": True,
                "server_log_lines": 34,
            },
        },
        "playtest_receipt": {
            "schema_version": "mmm/playtest-result-v3",
            "status": "PASS",
            "interaction_count": 1,
            "assertion_count": 1,
            "results": [
                {"action": "connect"},
                {"action": "use", "timestamp": "2026-08-02T00:00:01Z"},
                {"action": "wait_for", "matched": True},
            ],
        },
    }


def _compile_call(compiled, design: dict, inputs: dict, **overrides):
    values = {
        "game_design": design,
        "source_validation": inputs.get("source_validation"),
        "build_report": inputs.get("build_report"),
        "jar_validation": inputs.get("jar_validation"),
        "module_receipts": inputs.get("module_receipts", ()),
        "asset_receipt": inputs.get("asset_receipt"),
        "audio_receipt": inputs.get("audio_receipt"),
        "blockbench_receipts": inputs.get("blockbench_receipts", ()),
        "runtime_receipt": inputs.get("runtime_receipt"),
        "playtest_receipt": inputs.get("playtest_receipt"),
        "visual_receipt": inputs.get("visual_receipt"),
    }
    values.update(overrides)
    return compile_quality_evidence(
        compiled.contract,
        PROPOSAL_HASH,
        **values,
    )


def test_baseline_receipts_are_objective_stable_and_evaluator_compatible(
    tmp_path: Path,
) -> None:
    design = _research_design()
    compiled = _compile(design)
    inputs = _baseline_inputs(tmp_path)

    first = _compile_call(compiled, design, inputs)
    assert set(first) == {"correctness", "build", "research", "runtime"}
    assert all(value["status"] == "PASS" for value in first.values())
    assert all(value["producer"] != value["verified_by"] for value in first.values())
    assert all("+00:00" in value["observed_at"] for value in first.values())

    changed_time = copy.deepcopy(inputs)
    changed_time["source_validation"]["observed_at"] = (
        "2026-08-03T22:30:00-04:00"
    )
    changed_time["build_report"]["observed_at"] = (
        "2026-08-04T03:00:00+00:00"
    )
    changed_time["playtest_receipt"]["results"][1]["timestamp"] = (
        "2030-01-01T00:00:00Z"
    )
    second = _compile_call(compiled, design, changed_time)
    assert {
        key: value["receipt_id"] for key, value in first.items()
    } == {key: value["receipt_id"] for key, value in second.items()}
    assert {
        key: value["evidence_refs"] for key, value in first.items()
    } == {key: value["evidence_refs"] for key, value in second.items()}

    report = evaluate_quality_contract(
        compiled.contract, first, PROPOSAL_HASH
    )
    assert report["overall_status"] == "PASS"


@pytest.mark.parametrize(
    ("mutator"),
    [
        lambda design: design["_technology_radar"]["pagination"].update(
            {"complete": False, "next_cursor": "next"}
        ),
        lambda design: design["_ecosystem_discovery"]["errors"].append(
            {"provider": "modrinth", "error_type": "timeout"}
        ),
        lambda design: design["_technical_evidence"][
            "unresolved_official_domains"
        ].append("networking"),
    ],
)
def test_research_pass_requires_exhaustion_and_no_unresolved_evidence(
    tmp_path: Path, mutator
) -> None:
    design = _research_design()
    mutator(design)
    compiled = _compile(design)
    evidence = _compile_call(compiled, design, _baseline_inputs(tmp_path))

    assert "research" not in evidence


def test_generic_pass_and_gametest_do_not_close_conditional_dimensions(
    tmp_path: Path,
) -> None:
    design = _research_design()
    prompt = (
        "Build a game-scale multiplayer mod with a 3D model, audio, "
        "persistent saves, accessibility, and an AI voice companion."
    )
    compiled = _compile(
        design,
        prompt=prompt,
        modules=[
            _module("native_system", "custom_java"),
            _module("network", "networking"),
            _module("ai_voice", "integration"),
        ],
        assets=[
            {
                "asset_id": "companion",
                "kind": "entity",
                "prompt": "companion model",
                "target_path": "assets/companion.png",
                "width": 32,
                "height": 32,
            }
        ],
        audio=[{"sound_id": "companion_voice"}],
    )
    inputs = _baseline_inputs(tmp_path)
    inputs.update(
        {
            "module_receipts": ({"status": "PASS", "complete": True},),
            "asset_receipt": {"status": "PASS"},
            "audio_receipt": {"status": "PASS"},
            "visual_receipt": {"status": "PASS"},
        }
    )

    evidence = _compile_call(compiled, design, inputs)
    conditional = {
        "visual_3d",
        "audio",
        "state_save_migration",
        "multiplayer",
        "performance",
        "accessibility",
        "ai_voice",
    }
    assert conditional.isdisjoint(evidence)


def test_typed_conditional_metrics_close_each_requested_dimension(
    tmp_path: Path,
) -> None:
    design = _research_design()
    prompt = (
        "Build a game-scale multiplayer mod with a 3D model, audio, "
        "persistent saves, accessibility, and an AI voice companion."
    )
    compiled = _compile(
        design,
        prompt=prompt,
        modules=[
            _module("native_system", "custom_java"),
            _module("network", "networking"),
            _module("ai_voice", "integration"),
        ],
        assets=[
            {
                "asset_id": "companion",
                "kind": "entity",
                "prompt": "companion model",
                "target_path": "assets/companion.png",
                "width": 32,
                "height": 32,
            }
        ],
        audio=[{"sound_id": "companion_voice"}],
    )
    inputs = _baseline_inputs(tmp_path)

    asset = tmp_path / "companion.png"
    asset.write_bytes(b"png fixture")
    asset_sha = "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest()
    sound = tmp_path / "companion.ogg"
    sound.write_bytes(b"OggS" + b"audio" * 40)
    screenshot = tmp_path / "runtime.png"
    screenshot.write_bytes(b"runtime screenshot")
    preview = tmp_path / "blockbench.png"
    preview.write_bytes(b"blockbench preview")

    validations = {
        "audio_validation": {
            "schema_version": "mmm/audio-validation-v1",
            "status": "PASS",
            "registered_event_count": 1,
            "played_event_count": 1,
            "subtitle_check_count": 1,
            "missing_event_count": 0,
            "clipping_count": 0,
        },
        "state_validation": {
            "schema_version": "mmm/state-validation-v1",
            "status": "PASS",
            "round_trip_case_count": 2,
            "restart_case_count": 1,
            "corruption_case_count": 1,
            "migration_not_applicable": True,
            "data_loss_count": 0,
        },
        "multiplayer_validation": {
            "schema_version": "mmm/multiplayer-validation-v1",
            "status": "PASS",
            "client_count": 3,
            "authority_check_count": 2,
            "message_validation_check_count": 4,
            "reconnect_check_count": 1,
            "rejected_invalid_message_count": 2,
            "desync_count": 0,
        },
        "performance_validation": {
            "schema_version": "mmm/performance-validation-v1",
            "status": "PASS",
            "sample_count": 500,
            "skipped_work_count": 0,
            "budgets": {
                "tick_ms": {
                    "observed": 41.5,
                    "limit": 50.0,
                    "comparison": "lte",
                },
                "throughput": {
                    "observed": 120,
                    "limit": 100,
                    "comparison": "gte",
                },
            },
        },
        "accessibility_validation": {
            "schema_version": "mmm/accessibility-validation-v1",
            "status": "PASS",
            "declared_paths": ["keyboard", "subtitles"],
            "checks": [
                {
                    "path_ref": "keyboard",
                    "passed": True,
                    "evidence": "all actions are remappable",
                },
                {
                    "path_ref": "subtitles",
                    "passed": True,
                    "evidence": "voice event has a subtitle",
                },
            ],
        },
        "ai_voice_validation": {
            "schema_version": "mmm/ai-voice-validation-v1",
            "status": "PASS",
            "authority_check_count": 3,
            "privacy_check_count": 2,
            "failure_check_count": 2,
            "fallback_check_count": 2,
            "sidecar_unavailable_fallback_passed": True,
            "latency_sample_count": 50,
            "latency_p95_ms": 85.0,
            "latency_budget_ms": 100.0,
            "voice_adaptation": True,
            "consent_check_count": 1,
        },
    }
    inputs.update(
        {
            "module_receipts": ({"validations": validations},),
            "asset_receipt": {
                "schema_version": "mmm/complete-assets-sharded-v1",
                "status": "GENERATED",
                "shards": [
                    {
                        "assets": [
                            {
                                "asset_id": "companion",
                                "target": str(asset),
                                "width": 32,
                                "height": 32,
                                "sha256": asset_sha,
                            }
                        ]
                    }
                ],
            },
            "audio_receipt": {
                "schema_version": "mmm/complete-audio-sharded-v1",
                "status": "GENERATED",
                "shards": [
                    {
                        "sounds": [
                            {
                                "sound_id": "companion_voice",
                                "path": str(sound),
                                "size_bytes": sound.stat().st_size,
                            }
                        ]
                    }
                ],
            },
            "blockbench_receipts": (
                {
                    "entity": "companion",
                    "uv": {"status": "PASS"},
                    "preview": str(preview),
                },
            ),
        }
    )
    visual_test = next(
        item["statement"]
        for item in compiled.contract["acceptance_catalog"]
        if item["acceptance_ref"] == "acceptance:quality:visual_3d"
    )
    inputs["visual_receipt"] = {
        "schema_version": "mmm/visual-review-v2",
        "status": "PASS",
        "findings": [],
        "acceptance_test_results": [
            {
                "test": visual_test,
                "status": "PASS",
                "evidence": "runtime screenshot shows loaded model and texture",
            }
        ],
        "screenshots": [str(screenshot)],
    }

    evidence = _compile_call(compiled, design, inputs)
    assert {
        "visual_3d",
        "audio",
        "state_save_migration",
        "multiplayer",
        "performance",
        "accessibility",
        "ai_voice",
    } <= set(evidence)


def test_completion_claim_is_not_evidence_and_large_receipt_sets_are_exhausted(
) -> None:
    design = {"title": "Scale test"}
    compiled = _compile(
        design,
        prompt="Run a game-scale performance workload.",
    )
    performance = {
        "schema_version": "mmm/performance-validation-v1",
        "status": "PASS",
        "sample_count": 10_000,
        "skipped_work_count": 0,
        "budgets": {
            "tick_ms": {
                "observed": 42,
                "limit": 50,
                "comparison": "lte",
            }
        },
    }
    receipts = tuple(
        {"module_id": f"module_{index:05d}", "records": []}
        for index in range(10_000)
    ) + ({"nested": {"performance_validation": performance}},)
    empty_inputs = {
        "source_validation": None,
        "build_report": None,
        "jar_validation": None,
        "runtime_receipt": None,
        "playtest_receipt": None,
        "module_receipts": receipts,
    }

    evidence = _compile_call(compiled, design, empty_inputs)
    assert set(evidence) == {"performance"}
    assert evidence["performance"]["producer"] != evidence["performance"][
        "verified_by"
    ]

    claimed = copy.deepcopy(performance)
    claimed["complete"] = True
    rejected = _compile_call(
        compiled,
        design,
        {**empty_inputs, "module_receipts": ({"performance_validation": claimed},)},
    )
    assert "performance" not in rejected


def test_game_design_must_match_the_contract_binding(tmp_path: Path) -> None:
    design = _research_design()
    compiled = _compile(design)
    changed = copy.deepcopy(design)
    changed["title"] = "Different design"

    with pytest.raises(ProductionContractError, match="game_design"):
        _compile_call(compiled, changed, _baseline_inputs(tmp_path))
