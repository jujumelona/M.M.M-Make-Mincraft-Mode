from __future__ import annotations

"""Validate durable work-node completion evidence before receipt persistence.

Generation is a phase completion, never semantic verification. Validation/build/package
nodes may become durable SUCCEEDED only when their receipt contains independently
checkable evidence tied to the ledger's exact input hash. This module is deliberately
pure: the canonical work-graph receipt owner calls the decorator instead of installing
another runtime wrapper around ``DurableWorkLedger.succeed``.
"""

from collections.abc import Mapping, Sequence
import hashlib
import json
import marshal
from typing import Any


class VerifierReceiptTruthError(RuntimeError):
    pass


_EVIDENCE_SCHEMA = "mmm/work-completion-evidence-v2"
_VERIFIER_REUSE_CONTRACT = "mmm/verifier-reuse-contract-v1"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_hash(value: Any) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise VerifierReceiptTruthError(
            "VERIFIER_CONFIG_INVALID: verifier configuration must be canonical JSON."
        ) from exc
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _commands_passed(build: Mapping[str, Any]) -> bool:
    commands = build.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes, bytearray)):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("name") or "") in {"build", "clean_build"}
        and item.get("exit_code") == 0
        and item.get("timed_out") is not True
        for item in commands
    )


def _validation_evidence(
    stage: str,
    node_id: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(receipt.get("status") or "").strip().upper()
    if status not in {"PASS", "NOT_REQUIRED"}:
        raise VerifierReceiptTruthError(
            f"VERIFIER_RECEIPT_STATUS_INVALID: {node_id} cannot succeed with status {status!r}."
        )

    if stage == "validate:source":
        checks = receipt.get("checks_run")
        manifest = str(receipt.get("project_manifest") or "")
        if type(checks) is not int or checks <= 0 or not manifest:
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: source validation requires positive "
                "checks_run and project_manifest."
            )
        return {
            "verifier": "source_validator",
            "checks_run": checks,
            "project_manifest": manifest,
        }

    if stage == "validate:jar":
        checks = receipt.get("checks_run")
        jar_sha = str(receipt.get("jar_sha256") or "")
        if type(checks) is not int or checks <= 0 or not jar_sha.startswith("sha256:"):
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: JAR validation requires checks_run and jar_sha256."
            )
        return {
            "verifier": "jar_validator",
            "checks_run": checks,
            "artifact_sha256": jar_sha,
        }

    if stage == "validate:quality":
        receipt_id = str(receipt.get("receipt_id") or "")
        receipt_sha = str(receipt.get("receipt_sha256") or "")
        dimension_id = str(receipt.get("dimension_id") or "")
        if not receipt_id or not dimension_id or not receipt_sha.startswith("sha256:"):
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: quality validation requires "
                "dimension/receipt identity and hash."
            )
        return {
            "verifier": "quality_contract",
            "dimension_id": dimension_id,
            "receipt_id": receipt_id,
            "receipt_sha256": receipt_sha,
        }

    if stage == "validate:runtime":
        if status == "NOT_REQUIRED":
            return {"verifier": "runtime_policy", "status": "NOT_REQUIRED"}
        runtime = _mapping(receipt.get("runtime"))
        playtest = _mapping(receipt.get("playtest"))
        visual = _mapping(receipt.get("visual"))
        server = _mapping(runtime.get("server"))
        client = _mapping(runtime.get("client"))
        if server.get("server_running") is not True:
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: runtime PASS requires a running verified server."
            )
        if client and client.get("client_running") is not True:
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: runtime client receipt is present but not running."
            )
        if (
            playtest.get("status") != "PASS"
            or int(playtest.get("interaction_count") or 0) <= 0
            or int(playtest.get("assertion_count") or 0) <= 0
        ):
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: runtime PASS requires an executed "
                "assertion-bearing playtest."
            )
        if visual.get("status") != "PASS":
            raise VerifierReceiptTruthError(
                "VERIFIER_RECEIPT_MISSING: runtime PASS requires visual verification "
                "when declared complete."
            )
        return {
            "verifier": "runtime_playtest_visual",
            "server_running": True,
            "client_verified": bool(client),
            "interaction_count": int(playtest.get("interaction_count") or 0),
            "assertion_count": int(playtest.get("assertion_count") or 0),
        }

    raise VerifierReceiptTruthError(
        f"VERIFIER_RECEIPT_UNSUPPORTED_STAGE: no verifier truth contract for {stage!r}."
    )


def _build_evidence(node_id: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    if node_id != "build-project":
        raise VerifierReceiptTruthError(
            f"VERIFIER_RECEIPT_UNSUPPORTED_BUILD_NODE: {node_id!r}."
        )
    build = _mapping(receipt.get("build"))
    final = _mapping(receipt.get("final_build_receipt"))
    artifact = _mapping(build.get("artifact_receipt"))
    if build.get("status") != "PASS" or not _commands_passed(build):
        raise VerifierReceiptTruthError(
            "VERIFIER_RECEIPT_MISSING: build success requires a passing full Gradle "
            "command receipt."
        )
    if final.get("status") != "PASS" or final.get("production_jar") != "PASS":
        raise VerifierReceiptTruthError(
            "VERIFIER_RECEIPT_MISSING: build success requires final production-JAR attestation."
        )
    artifact_sha = str(artifact.get("sha256") or final.get("artifact_sha256") or "")
    if not artifact_sha.startswith("sha256:"):
        raise VerifierReceiptTruthError(
            "VERIFIER_RECEIPT_MISSING: build success requires a SHA-256 artifact identity."
        )
    return {
        "verifier": "gradle_and_final_artifact",
        "artifact_sha256": artifact_sha,
        "toolchain_attested": final.get("toolchain_attested") is True,
    }


def _package_evidence(node_id: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    if node_id != "package-release":
        return {
            "verifier": "package_phase",
            "status": str(receipt.get("status") or ""),
        }
    release_zip = str(receipt.get("release_zip") or "")
    if str(receipt.get("status") or "").upper() != "PASS" or not release_zip:
        raise VerifierReceiptTruthError(
            "VERIFIER_RECEIPT_MISSING: release package success requires an explicit "
            "release artifact path."
        )
    return {"verifier": "release_package", "release_zip": release_zip}


def _verifier_implementation_hash(verifier: str) -> str:
    if verifier in {
        "source_validator",
        "jar_validator",
        "quality_contract",
        "runtime_policy",
        "runtime_playtest_visual",
    }:
        function = _validation_evidence
    elif verifier == "gradle_and_final_artifact":
        function = _build_evidence
    elif verifier in {"package_phase", "release_package"}:
        function = _package_evidence
    else:
        raise VerifierReceiptTruthError(
            f"VERIFIER_ID_UNKNOWN: no implementation identity for verifier {verifier!r}."
        )
    code_hash = "sha256:" + hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    return _canonical_hash(
        {
            "contract": _VERIFIER_REUSE_CONTRACT,
            "verifier": verifier,
            "implementation_code_hash": code_hash,
        }
    )


def _stage_evidence(
    stage: str,
    node_id: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if stage.startswith("generate:"):
        return {"completion_scope": "phase_only", "verifier": None}
    if stage.startswith("validate:"):
        return {
            "completion_scope": "verified_stage",
            **_validation_evidence(stage, node_id, receipt),
        }
    if stage == "build":
        return {
            "completion_scope": "verified_stage",
            **_build_evidence(node_id, receipt),
        }
    if stage.startswith("package"):
        return {
            "completion_scope": "packaging_stage",
            **_package_evidence(node_id, receipt),
        }
    return {"completion_scope": "phase_only", "verifier": None}


def _completion_evidence(
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    node_id = str(row.get("node_id") or "")
    stage = str(row.get("stage") or "")
    input_hash = str(row.get("input_hash") or "")
    evidence = _stage_evidence(stage, node_id, receipt)
    result = {
        "schema_version": _EVIDENCE_SCHEMA,
        "node_id": node_id,
        "stage": stage,
        "input_hash": input_hash,
        **evidence,
    }
    verifier = evidence.get("verifier")
    if verifier:
        payload = row.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise VerifierReceiptTruthError(
                f"VERIFIER_CONFIG_INVALID: work node {node_id!r} payload must be an object."
            )
        result.update(
            {
                "verifier_input_hash": input_hash,
                "verifier_version_hash": _verifier_implementation_hash(str(verifier)),
                "verifier_config_hash": _canonical_hash(
                    {"stage": stage, "payload": dict(payload)}
                ),
            }
        )
    return result


def _assert_reusable(
    node_id: str,
    expected: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> None:
    if (
        existing.get("input_hash") != expected.get("input_hash")
        or existing.get("stage") != expected.get("stage")
    ):
        raise VerifierReceiptTruthError(
            f"VERIFIER_RECEIPT_STALE: work node {node_id!r} carries mismatched "
            "completion evidence."
        )

    if expected.get("verifier") is None:
        return

    required = (
        "schema_version",
        "completion_scope",
        "verifier",
        "verifier_input_hash",
        "verifier_version_hash",
        "verifier_config_hash",
    )
    if any(existing.get(key) != expected.get(key) for key in required):
        raise VerifierReceiptTruthError(
            f"VERIFIER_RECEIPT_STALE: work node {node_id!r} verifier identity, "
            "version, configuration, or inputs changed."
        )


def _decorate_receipt(
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise VerifierReceiptTruthError(
            "VERIFIER_RECEIPT_INVALID: work row must be an object."
        )
    node_id = str(row.get("node_id") or "")
    if not isinstance(receipt, Mapping):
        raise VerifierReceiptTruthError(
            f"VERIFIER_RECEIPT_INVALID: work node {node_id!r} receipt must be an object."
        )

    input_hash = str(row.get("input_hash") or "")
    if not input_hash:
        raise VerifierReceiptTruthError(
            f"VERIFIER_INPUT_HASH_MISSING: work node {node_id!r} has no exact input identity."
        )

    result = dict(receipt)
    expected = _completion_evidence(row, result)
    existing = result.get("_mmm_completion_evidence")
    if existing is not None:
        _assert_reusable(node_id, expected, _mapping(existing))
        return result

    result["_mmm_completion_evidence"] = expected
    return result


def install(work_graph_module: Any) -> None:
    """Fail closed unless verifier truth is owned by the canonical receipt wrapper."""

    succeed = work_graph_module.DurableWorkLedger.succeed
    if not getattr(succeed, "_mmm_verifier_receipt_truth_integrated", False):
        raise RuntimeError(
            "verifier receipt truth is not integrated into the canonical durable "
            "receipt owner"
        )


__all__ = ["VerifierReceiptTruthError", "_decorate_receipt", "install"]
