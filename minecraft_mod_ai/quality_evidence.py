from __future__ import annotations

"""Compile existing production receipts into quality-contract evidence.

The adapter is deliberately fail-closed.  It emits only ``PASS`` receipts and
only for dimensions for which the production pipeline supplied objective,
dimension-specific evidence.  Missing or malformed evidence is omitted so
``evaluate_quality_contract`` reports ``MISSING``; this module never invents a
failure and never treats a worker's claim of completion as verification.

Conditional validators use small, typed metric contracts. They are intended
for runtime/test adapters to populate, not for the planner or generator:

* ``audio_validation`` (``mmm/audio-validation-v1``): registered and played
  events, subtitles, and zero missing events/clipping.
* ``state_validation`` (``mmm/state-validation-v1``): round-trip, restart,
  corruption, migration/not-applicable, and zero data loss.
* ``multiplayer_validation`` (``mmm/multiplayer-validation-v1``): at least two
  clients, authority/message/reconnect checks, invalid-message rejection, and
  zero desynchronization.
* ``performance_validation`` (``mmm/performance-validation-v1``): sampled
  workloads and explicit numeric budgets, all met with no skipped work.
* ``accessibility_validation`` (``mmm/accessibility-validation-v1``): every
  declared path has a passing, evidence-bearing check.
* ``ai_voice_validation`` (``mmm/ai-voice-validation-v1``): authority,
  privacy, failure and offline-fallback checks plus measured latency budget.

Visual evidence uses the existing visual-review receipt and, when relevant,
asset and Blockbench file receipts.  A generic GameTest can support baseline
correctness, but it can never satisfy one of the conditional dimensions.
"""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .production_contract import (
    ProductionContractError,
    bound_game_design,
    validate_production_contract,
)


_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_FORBIDDEN_COMPLETION_FIELDS = {
    "all_passed",
    "complete",
    "completion",
    "overall_status",
}
_CONDITIONAL_KEYS = {
    "audio": "audio_validation",
    "state_save_migration": "state_validation",
    "multiplayer": "multiplayer_validation",
    "performance": "performance_validation",
    "accessibility": "accessibility_validation",
    "ai_voice": "ai_voice_validation",
}


EvidenceResult = tuple[list[str], list[Mapping[str, Any]]]


def compile_quality_evidence(
    contract: Mapping[str, Any],
    proposal_hash: str,
    *,
    game_design: Mapping[str, Any],
    source_validation: Mapping[str, Any] | None,
    build_report: Mapping[str, Any] | None,
    jar_validation: Mapping[str, Any] | None,
    module_receipts: Iterable[Mapping[str, Any]] = (),
    asset_receipt: Mapping[str, Any] | None = None,
    audio_receipt: Mapping[str, Any] | None = None,
    blockbench_receipts: Iterable[Mapping[str, Any]] = (),
    runtime_receipt: Mapping[str, Any] | None = None,
    playtest_receipt: Mapping[str, Any] | None = None,
    visual_receipt: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return independently checked ``PASS`` receipts keyed by dimension.

    Receipt IDs are hashes of the proposal binding, evidence route, and stable
    evidence references.  ``observed_at`` records when this adapter observed the
    evidence, but wall time is intentionally excluded from the ID.
    """

    module_ids = [
        str(item["implementation_id"])
        for item in contract.get("implementation_catalog", [])
        if isinstance(item, Mapping) and item.get("source_kind") == "module"
    ]
    acceptance_tests = [
        str(item["statement"])
        for item in contract.get("acceptance_catalog", [])
        if isinstance(item, Mapping) and isinstance(item.get("statement"), str)
    ]
    validate_production_contract(contract, module_ids, acceptance_tests)
    if not isinstance(proposal_hash, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", proposal_hash
    ):
        raise ProductionContractError(
            "proposal_hash must be a canonical SHA-256"
        )
    if not isinstance(game_design, Mapping):
        raise ProductionContractError("game_design must be an object")
    design_snapshot = bound_game_design(game_design)
    if _canonical_sha256(design_snapshot) != contract["source_bindings"][
        "game_design_sha256"
    ]:
        raise ProductionContractError(
            "game_design does not match the production contract binding"
        )

    modules = tuple(_mapping_items(module_receipts))
    blockbench = tuple(_mapping_items(blockbench_receipts))
    routes = {
        str(item["dimension_id"]): str(item["evidence_route_ref"])
        for item in contract["quality_dimension_catalog"]
    }
    roots: tuple[Any, ...] = (
        *modules,
        asset_receipt,
        audio_receipt,
        *blockbench,
        runtime_receipt,
        playtest_receipt,
        visual_receipt,
    )

    source = _source_validation_evidence(source_validation)
    clean_build = _clean_build_evidence(build_report)
    gametest = _gametest_evidence(build_report)
    jar = _jar_evidence(build_report, jar_validation)
    research = _research_evidence(game_design)
    runtime = _runtime_evidence(runtime_receipt, playtest_receipt)

    candidates: dict[str, EvidenceResult | None] = {
        "correctness": _combine(source, clean_build, gametest),
        "build": _combine(clean_build, jar),
        "research": research,
        "runtime": runtime,
    }

    if "visual_3d" in routes:
        candidates["visual_3d"] = _visual_evidence(
            contract,
            asset_receipt,
            blockbench,
            visual_receipt,
        )
    if "audio" in routes:
        candidates["audio"] = _audio_evidence(
            contract, audio_receipt, roots
        )
    for dimension_id, key in _CONDITIONAL_KEYS.items():
        if dimension_id not in routes or dimension_id == "audio":
            continue
        validators: dict[str, Callable[[Mapping[str, Any]], bool]] = {
            "state_save_migration": _valid_state_validation,
            "multiplayer": _valid_multiplayer_validation,
            "performance": _valid_performance_validation,
            "accessibility": _valid_accessibility_validation,
            "ai_voice": _valid_ai_voice_validation,
        }
        candidates[dimension_id] = _explicit_evidence(
            key,
            roots,
            validators[dimension_id],
        )

    output: dict[str, dict[str, Any]] = {}
    for dimension_id in routes:
        candidate = candidates.get(dimension_id)
        if candidate is None:
            continue
        refs, observed_sources = candidate
        output[dimension_id] = _quality_receipt(
            dimension_id=dimension_id,
            route_ref=routes[dimension_id],
            proposal_hash=proposal_hash,
            evidence_refs=refs,
            observed_sources=observed_sources,
        )
    return output


def _source_validation_evidence(
    value: Mapping[str, Any] | None,
) -> EvidenceResult | None:
    if not _objective_pass(value):
        return None
    checks = _nonnegative_int(value.get("checks_run"))
    findings = value.get("findings")
    if checks is None or checks <= 0 or not _is_sequence(findings):
        return None
    if any(
        isinstance(item, Mapping)
        and str(item.get("severity", "")).casefold() in {"error", "fatal"}
        for item in findings
    ):
        return None
    facts = {
        "status": "PASS",
        "checks_run": checks,
        "finding_count": len(findings),
    }
    return [_evidence_ref("source-validation", facts)], [value]


def _clean_build_evidence(
    value: Mapping[str, Any] | None,
) -> EvidenceResult | None:
    if not _objective_pass(value):
        return None
    commands = _commands(value)
    clean = [item for item in commands if item.get("name") == "clean_build"]
    if not clean or not all(_command_passed(item) for item in clean):
        return None
    facts = {
        "status": "PASS",
        "gradle_version": str(value.get("gradle_version", "")),
        "clean_build_count": len(clean),
        "commands": [
            {
                "name": str(item.get("name", "")),
                "exit_code": item.get("exit_code"),
                "timed_out": item.get("timed_out", False),
            }
            for item in commands
        ],
    }
    return [_evidence_ref("gradle-clean-build", facts)], [value]


def _gametest_evidence(
    value: Mapping[str, Any] | None,
) -> EvidenceResult | None:
    if not _objective_pass(value):
        return None
    commands = _commands(value)
    runs = [item for item in commands if item.get("name") == "gametest"]
    if not runs or not all(_command_passed(item) for item in runs):
        return None
    raw_path = value.get("gametest_report")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    report = _passing_gametest_xml(Path(raw_path))
    if report is None:
        return None
    facts = {"command_count": len(runs), **report}
    return [_evidence_ref("fabric-gametest", facts)], [value]


def _jar_evidence(
    build: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
) -> EvidenceResult | None:
    if not _objective_pass(build) or not _objective_pass(validation):
        return None
    checks = _nonnegative_int(validation.get("checks_run"))
    findings = validation.get("findings")
    if checks is None or checks <= 0 or not _is_sequence(findings):
        return None
    if any(
        isinstance(item, Mapping)
        and str(item.get("severity", "")).casefold() in {"error", "fatal"}
        for item in findings
    ):
        return None
    raw_path = build.get("jar_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    digest = _regular_file_sha256(Path(raw_path))
    if digest is None:
        return None
    facts = {
        "jar_sha256": digest,
        "checks_run": checks,
        "finding_count": len(findings),
    }
    return [_evidence_ref("jar-validation", facts)], [build, validation]


def _research_evidence(game_design: Mapping[str, Any]) -> EvidenceResult | None:
    technology = game_design.get("_technology_radar")
    ecosystem = game_design.get("_ecosystem_discovery")
    technical = game_design.get("_technical_evidence")
    if not all(isinstance(item, Mapping) for item in (technology, ecosystem, technical)):
        return None
    assert isinstance(technology, Mapping)
    assert isinstance(ecosystem, Mapping)
    assert isinstance(technical, Mapping)

    pagination = technology.get("pagination")
    requirements = technology.get("requirements")
    technology_collection = technology.get("collection_receipt")
    if (
        technology.get("aggregate_schema_version")
        != "mmm/technology-radar-aggregate-v1"
        or not isinstance(pagination, Mapping)
        or pagination.get("complete") is not True
        or pagination.get("next_cursor") != ""
        or not _is_sequence(requirements)
        or _nonnegative_int(pagination.get("returned")) != len(requirements)
        or _nonnegative_int(pagination.get("total_requirements"))
        != len(requirements)
        or _positive_int(pagination.get("pages_collected")) is None
        or not isinstance(technology_collection, Mapping)
        or technology_collection.get("schema_version")
        != "mmm/technology-page-collection-receipt-v1"
        or not _valid_digest(technology.get("radar_sha256"))
    ):
        return None

    route_count = _nonnegative_int(ecosystem.get("route_count"))
    processed = _nonnegative_int(ecosystem.get("processed_route_count"))
    remaining = _nonnegative_int(ecosystem.get("remaining_route_count"))
    errors = ecosystem.get("errors")
    ecosystem_collection = ecosystem.get("collection_receipt")
    status = str(ecosystem.get("status", ""))
    if (
        ecosystem.get("aggregate_schema_version")
        != "mmm/ecosystem-seed-aggregate-v1"
        or ecosystem.get("routes_complete") is not True
        or ecosystem.get("next_route_cursor") != ""
        or route_count is None
        or processed != route_count
        or remaining != 0
        or not _is_sequence(errors)
        or len(errors) != 0
        or not isinstance(ecosystem_collection, Mapping)
        or ecosystem_collection.get("schema_version")
        != "mmm/ecosystem-route-collection-receipt-v1"
        or not _valid_digest(ecosystem.get("route_sha256"))
        or (route_count > 0 and status not in {"available", "empty"})
    ):
        return None

    unresolved = technical.get("unresolved_official_domains")
    domains = technical.get("domains")
    if (
        technical.get("schema_version") != "mmm/central-evidence-graph-v1"
        or not _is_sequence(unresolved)
        or len(unresolved) != 0
        or not _is_sequence(domains)
        or not _valid_digest(technical.get("evidence_sha256"))
    ):
        return None

    refs = [
        _evidence_ref(
            "technology-pagination",
            {
                "radar_sha256": _digest(technology["radar_sha256"]),
                "requirements": len(requirements),
                "pages": pagination["pages_collected"],
                "collection": technology_collection,
            },
        ),
        _evidence_ref(
            "ecosystem-route-catalog",
            {
                "route_sha256": _digest(ecosystem["route_sha256"]),
                "routes": route_count,
                "processed": processed,
                "status": status,
                "collection": ecosystem_collection,
            },
        ),
        _evidence_ref(
            "official-rag",
            {
                "evidence_sha256": _digest(technical["evidence_sha256"]),
                "domain_count": len(domains),
                "unresolved": [],
            },
        ),
    ]
    return refs, [technology, ecosystem, technical]


def _runtime_evidence(
    runtime: Mapping[str, Any] | None,
    playtest: Mapping[str, Any] | None,
) -> EvidenceResult | None:
    if not isinstance(runtime, Mapping) or not _objective_pass(playtest):
        return None
    prepared = runtime.get("prepared")
    server = runtime.get("server")
    results = playtest.get("results")
    interactions = _positive_int(playtest.get("interaction_count"))
    assertions = _positive_int(playtest.get("assertion_count"))
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("schema_version") != "mmm/runtime-instance-v1"
        or prepared.get("disposable") is not True
        or not isinstance(server, Mapping)
        or server.get("schema_version") != "mmm/runtime-status-v1"
        or server.get("server_running") is not True
        or _positive_int(server.get("server_log_lines")) is None
        or interactions is None
        or assertions is None
        or not _is_sequence(results)
        or len(results) < interactions + assertions
    ):
        return None
    refs = [
        _evidence_ref(
            "minecraft-runtime",
            {
                "minecraft_version": prepared.get("minecraft_version"),
                "disposable": True,
                "server_log_lines": server.get("server_log_lines"),
            },
        ),
        _evidence_ref(
            "runtime-playtest",
            {
                "interaction_count": interactions,
                "assertion_count": assertions,
                "results_sha256": _canonical_sha256(_stable_value(results)),
            },
        ),
    ]
    return refs, [runtime, playtest]


def _visual_evidence(
    contract: Mapping[str, Any],
    assets: Mapping[str, Any] | None,
    blockbench: Sequence[Mapping[str, Any]],
    visual: Mapping[str, Any] | None,
) -> EvidenceResult | None:
    if not _objective_pass(visual):
        return None
    if visual.get("schema_version") != "mmm/visual-review-v2":
        return None
    screenshots = visual.get("screenshots")
    results = visual.get("acceptance_test_results")
    if not _is_sequence(screenshots) or not screenshots or not _is_sequence(results):
        return None
    screenshot_hashes: list[str] = []
    for raw in screenshots:
        if not isinstance(raw, str):
            return None
        digest = _regular_file_sha256(Path(raw))
        if digest is None:
            return None
        screenshot_hashes.append(digest)
    expected_test = next(
        (
            str(item["statement"])
            for item in contract["acceptance_catalog"]
            if isinstance(item, Mapping)
            and item.get("acceptance_ref") == "acceptance:quality:visual_3d"
        ),
        "",
    )
    if not expected_test or not any(
        isinstance(item, Mapping)
        and item.get("test") == expected_test
        and item.get("status") == "PASS"
        and isinstance(item.get("evidence"), str)
        and bool(item["evidence"].strip())
        for item in results
    ):
        return None
    if any(
        not isinstance(item, Mapping) or item.get("status") != "PASS"
        for item in results
    ):
        return None

    refs = [
        _evidence_ref(
            "visual-review",
            {
                "screenshots": screenshot_hashes,
                "acceptance_results": [
                    {
                        "test": item.get("test"),
                        "status": item.get("status"),
                        "evidence": item.get("evidence"),
                    }
                    for item in results
                ],
            },
        )
    ]
    sources: list[Mapping[str, Any]] = [visual]

    expected_assets = _implementation_ids(contract, "asset")
    if expected_assets:
        asset_result = _asset_integrity_evidence(assets, expected_assets)
        if asset_result is None:
            return None
        refs.extend(asset_result[0])
        sources.extend(asset_result[1])

    if _requires_blockbench(contract):
        if not blockbench:
            return None
        for receipt in blockbench:
            uv = receipt.get("uv")
            preview = receipt.get("preview")
            if (
                not isinstance(uv, Mapping)
                or uv.get("status") not in {"PASS", "OK"}
                or not isinstance(preview, str)
            ):
                return None
            preview_hash = _regular_file_sha256(Path(preview))
            if preview_hash is None:
                return None
            refs.append(
                _evidence_ref(
                    "blockbench-review",
                    {
                        "entity": receipt.get("entity"),
                        "uv_status": uv.get("status"),
                        "preview_sha256": preview_hash,
                    },
                )
            )
            sources.append(receipt)
    return refs, sources


def _asset_integrity_evidence(
    receipt: Mapping[str, Any] | None,
    expected_ids: set[str],
) -> EvidenceResult | None:
    if not isinstance(receipt, Mapping) or receipt.get("status") != "GENERATED":
        return None
    entries = _named_list_mappings(receipt, "assets")
    by_id = {
        str(item.get("asset_id")): item
        for item in entries
        if isinstance(item.get("asset_id"), str)
    }
    if not expected_ids <= set(by_id):
        return None
    facts: list[dict[str, Any]] = []
    for asset_id in sorted(expected_ids):
        item = by_id[asset_id]
        raw_path = item.get("target")
        width = _positive_int(item.get("width"))
        height = _positive_int(item.get("height"))
        if not isinstance(raw_path, str) or width is None or height is None:
            return None
        actual = _regular_file_sha256(Path(raw_path))
        if actual is None or _digest(item.get("sha256")) != actual:
            return None
        facts.append(
            {
                "asset_id": asset_id,
                "width": width,
                "height": height,
                "sha256": actual,
            }
        )
    return [_evidence_ref("asset-integrity", facts)], [receipt]


def _audio_evidence(
    contract: Mapping[str, Any],
    audio: Mapping[str, Any] | None,
    roots: Sequence[Any],
) -> EvidenceResult | None:
    expected_ids = _implementation_ids(contract, "audio")
    if not expected_ids or not isinstance(audio, Mapping):
        return None
    if audio.get("status") != "GENERATED":
        return None
    entries = _named_list_mappings(audio, "sounds")
    by_id = {
        str(item.get("sound_id")): item
        for item in entries
        if isinstance(item.get("sound_id"), str)
    }
    if not expected_ids <= set(by_id):
        return None
    sound_facts: list[dict[str, Any]] = []
    for sound_id in sorted(expected_ids):
        item = by_id[sound_id]
        raw_path = item.get("path")
        size = _positive_int(item.get("size_bytes"))
        if not isinstance(raw_path, str) or size is None:
            return None
        digest = _regular_file_sha256(Path(raw_path))
        if digest is None or Path(raw_path).stat().st_size != size:
            return None
        sound_facts.append(
            {"sound_id": sound_id, "size_bytes": size, "sha256": digest}
        )
    explicit = _explicit_evidence(
        "audio_validation",
        roots,
        lambda value: _valid_audio_validation(value, len(expected_ids)),
    )
    if explicit is None:
        return None
    refs, sources = explicit
    refs.append(_evidence_ref("audio-assets", sound_facts))
    return refs, [audio, *sources]


def _explicit_evidence(
    key: str,
    roots: Sequence[Any],
    validator: Callable[[Mapping[str, Any]], bool],
) -> EvidenceResult | None:
    records = _named_mappings(roots, key)
    if not records or any(not validator(record) for record in records):
        return None
    refs = [
        _evidence_ref(key.replace("_", "-"), _stable_record(record))
        for record in records
    ]
    return refs, list(records)


def _valid_audio_validation(
    value: Mapping[str, Any], expected_count: int
) -> bool:
    return (
        _typed_pass(value, "mmm/audio-validation-v1")
        and (_nonnegative_int(value.get("registered_event_count")) or 0)
        >= expected_count
        and (_nonnegative_int(value.get("played_event_count")) or 0)
        >= expected_count
        and (_nonnegative_int(value.get("subtitle_check_count")) or 0)
        >= expected_count
        and _zero(value.get("missing_event_count"))
        and _zero(value.get("clipping_count"))
    )


def _valid_state_validation(value: Mapping[str, Any]) -> bool:
    migration = _positive_int(value.get("migration_case_count")) is not None
    not_applicable = value.get("migration_not_applicable") is True or (
        isinstance(value.get("migration_not_applicable_reason"), str)
        and bool(value["migration_not_applicable_reason"].strip())
    )
    return (
        _typed_pass(value, "mmm/state-validation-v1")
        and _positive_int(value.get("round_trip_case_count")) is not None
        and _positive_int(value.get("restart_case_count")) is not None
        and _positive_int(value.get("corruption_case_count")) is not None
        and (migration or not_applicable)
        and _zero(value.get("data_loss_count"))
    )


def _valid_multiplayer_validation(value: Mapping[str, Any]) -> bool:
    clients = _nonnegative_int(value.get("client_count"))
    return (
        _typed_pass(value, "mmm/multiplayer-validation-v1")
        and clients is not None
        and clients >= 2
        and _positive_int(value.get("authority_check_count")) is not None
        and _positive_int(value.get("message_validation_check_count"))
        is not None
        and _positive_int(value.get("reconnect_check_count")) is not None
        and _positive_int(value.get("rejected_invalid_message_count"))
        is not None
        and _zero(value.get("desync_count"))
    )


def _valid_performance_validation(value: Mapping[str, Any]) -> bool:
    if (
        not _typed_pass(value, "mmm/performance-validation-v1")
        or _positive_int(value.get("sample_count")) is None
        or not _zero(value.get("skipped_work_count"))
    ):
        return False
    budgets = value.get("budgets")
    if not isinstance(budgets, Mapping) or not budgets:
        return False
    return all(
        isinstance(name, str)
        and bool(name.strip())
        and isinstance(metric, Mapping)
        and _budget_met(metric)
        for name, metric in budgets.items()
    )


def _valid_accessibility_validation(value: Mapping[str, Any]) -> bool:
    if not _typed_pass(value, "mmm/accessibility-validation-v1"):
        return False
    paths = value.get("declared_paths")
    checks = value.get("checks")
    if (
        not _is_sequence(paths)
        or not paths
        or any(not isinstance(path, str) or not path.strip() for path in paths)
        or len(set(paths)) != len(paths)
        or not _is_sequence(checks)
    ):
        return False
    covered: set[str] = set()
    for check in checks:
        if (
            not isinstance(check, Mapping)
            or check.get("passed") is not True
            or not isinstance(check.get("path_ref"), str)
            or check.get("path_ref") not in paths
            or not isinstance(check.get("evidence"), str)
            or not check["evidence"].strip()
        ):
            return False
        covered.add(str(check["path_ref"]))
    return covered == set(paths)


def _valid_ai_voice_validation(value: Mapping[str, Any]) -> bool:
    samples = _positive_int(value.get("latency_sample_count"))
    p95 = _finite_number(value.get("latency_p95_ms"))
    budget = _finite_number(value.get("latency_budget_ms"))
    if not (
        _typed_pass(value, "mmm/ai-voice-validation-v1")
        and _positive_int(value.get("authority_check_count")) is not None
        and _positive_int(value.get("privacy_check_count")) is not None
        and _positive_int(value.get("failure_check_count")) is not None
        and _positive_int(value.get("fallback_check_count")) is not None
        and value.get("sidecar_unavailable_fallback_passed") is True
        and samples is not None
        and p95 is not None
        and budget is not None
        and 0 <= p95 <= budget
        and budget > 0
    ):
        return False
    if value.get("voice_adaptation") is True:
        return _positive_int(value.get("consent_check_count")) is not None
    return True


def _typed_pass(value: Mapping[str, Any], schema_version: str) -> bool:
    return value.get("schema_version") == schema_version and _objective_pass(value)


def _objective_pass(value: Mapping[str, Any] | None) -> bool:
    if not isinstance(value, Mapping) or value.get("status") != "PASS":
        return False
    if _FORBIDDEN_COMPLETION_FIELDS & set(value):
        return False
    producer = value.get("producer")
    verifier = value.get("verified_by")
    return not (
        isinstance(producer, str)
        and isinstance(verifier, str)
        and producer.strip().casefold() == verifier.strip().casefold()
    )


def _combine(*values: EvidenceResult | None) -> EvidenceResult | None:
    if any(value is None for value in values):
        return None
    refs: list[str] = []
    sources: list[Mapping[str, Any]] = []
    for value in values:
        assert value is not None
        refs.extend(value[0])
        sources.extend(value[1])
    return refs, sources


def _quality_receipt(
    *,
    dimension_id: str,
    route_ref: str,
    proposal_hash: str,
    evidence_refs: Sequence[str],
    observed_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    refs = sorted(set(evidence_refs))
    receipt_id = "quality:" + dimension_id + ":" + _canonical_sha256(
        {
            "dimension_id": dimension_id,
            "route_ref": route_ref,
            "proposal_hash": proposal_hash,
            "evidence_refs": refs,
        }
    ).removeprefix("sha256:")
    return {
        "dimension_id": dimension_id,
        "route_ref": route_ref,
        "status": "PASS",
        "proposal_hash": proposal_hash,
        "receipt_id": receipt_id,
        "observed_at": _observed_at(observed_sources),
        "producer": "mmm.production-orchestrator",
        "verified_by": "mmm.quality-evidence-adapter/v1",
        "evidence_refs": refs,
    }


def _observed_at(values: Sequence[Mapping[str, Any]]) -> str:
    candidates: list[datetime] = []
    for value in values:
        for field in ("observed_at", "finished_at", "created_at", "timestamp"):
            raw = value.get(field)
            if not isinstance(raw, str):
                continue
            candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                candidates.append(parsed)
    observed = max(candidates) if candidates else datetime.now(timezone.utc)
    return observed.astimezone(timezone.utc).isoformat()


def _commands(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    commands = value.get("commands")
    if not _is_sequence(commands):
        return []
    return [item for item in commands if isinstance(item, Mapping)]


def _command_passed(value: Mapping[str, Any]) -> bool:
    return (
        _nonnegative_int(value.get("exit_code")) == 0
        and value.get("timed_out", False) is False
    )


def _passing_gametest_xml(path: Path) -> dict[str, Any] | None:
    digest = _regular_file_sha256(path)
    if digest is None:
        return None
    test_count = 0
    suite_count = 0
    failed = False
    try:
        for _event, element in ET.iterparse(path, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag in {"testsuite", "testsuites"}:
                suite_count += tag == "testsuite"
                for field in ("failures", "errors", "skipped"):
                    raw = element.attrib.get(field, "0")
                    try:
                        failed = failed or int(raw) != 0
                    except ValueError:
                        return None
            elif tag == "testcase":
                test_count += 1
                if any(
                    child.tag.rsplit("}", 1)[-1]
                    in {"failure", "error", "skipped"}
                    for child in element
                ):
                    failed = True
            element.clear()
    except (ET.ParseError, OSError):
        return None
    if failed or test_count <= 0 or suite_count <= 0:
        return None
    return {
        "report_sha256": digest,
        "test_count": test_count,
        "suite_count": suite_count,
    }


def _implementation_ids(contract: Mapping[str, Any], kind: str) -> set[str]:
    return {
        str(item["implementation_id"])
        for item in contract["implementation_catalog"]
        if isinstance(item, Mapping) and item.get("source_kind") == kind
    }


def _requires_blockbench(contract: Mapping[str, Any]) -> bool:
    module_kinds = {
        str(item.get("kind", "")).casefold()
        for item in contract["implementation_catalog"]
        if isinstance(item, Mapping) and item.get("source_kind") == "module"
    }
    asset_kinds = {
        str(item.get("kind", "")).casefold()
        for item in contract["implementation_catalog"]
        if isinstance(item, Mapping) and item.get("source_kind") == "asset"
    }
    text = " " + str(contract.get("requested_prompt", "")).casefold() + " "
    return bool(module_kinds & {"entity", "boss", "npc"}) or bool(
        asset_kinds & {"entity", "model", "animation"}
    ) or any(
        term in text
        for term in (" 3d ", " model ", " animation ", " 3차원 ", " 모델 ", " 애니메이션 ")
    )


def _named_mappings(roots: Sequence[Any], key: str) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    result_ids: set[int] = set()
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            selected = value.get(key)
            if isinstance(selected, Mapping):
                if id(selected) not in result_ids:
                    result.append(selected)
                    result_ids.add(id(selected))
            elif _is_sequence(selected):
                for item in selected:
                    if isinstance(item, Mapping) and id(item) not in result_ids:
                        result.append(item)
                        result_ids.add(id(item))
            stack.extend(reversed(list(value.values())))
        elif _is_sequence(value):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(reversed(value))
    return result


def _named_list_mappings(root: Any, key: str) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            selected = value.get(key)
            if _is_sequence(selected):
                result.extend(item for item in selected if isinstance(item, Mapping))
            stack.extend(reversed(list(value.values())))
        elif _is_sequence(value):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(reversed(value))
    return result


def _mapping_items(values: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for value in values:
        if not isinstance(value, Mapping):
            raise ProductionContractError("evidence receipt must be an object")
        yield value


def _stable_record(value: Mapping[str, Any]) -> dict[str, Any]:
    stable = _stable_value(value)
    assert isinstance(stable, dict)
    return stable


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in value.items()
            if str(key)
            not in {"observed_at", "finished_at", "created_at", "timestamp"}
        }
    if _is_sequence(value):
        return [_stable_value(item) for item in value]
    return value


def _budget_met(value: Mapping[str, Any]) -> bool:
    observed = _finite_number(value.get("observed"))
    limit = _finite_number(value.get("limit"))
    comparison = value.get("comparison")
    if observed is None or limit is None or comparison not in {"lte", "gte"}:
        return False
    return observed <= limit if comparison == "lte" else observed >= limit


def _evidence_ref(label: str, value: Any) -> str:
    return f"{label}:{_canonical_sha256(_stable_value(value))}"


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionContractError(
            "quality evidence must be finite JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _regular_file_sha256(path: Path) -> str | None:
    try:
        path = path.expanduser()
        if path.is_symlink():
            return None
        path = path.resolve(strict=True)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except (OSError, RuntimeError):
        return None


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value.casefold()))


def _digest(value: Any) -> str:
    if not _valid_digest(value):
        return ""
    return "sha256:" + str(value).casefold().removeprefix("sha256:")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _zero(value: Any) -> bool:
    return type(value) is int and value == 0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = ["compile_quality_evidence"]
