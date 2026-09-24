from __future__ import annotations

"""Runtime evidence collection and release-gate predicates."""

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .complete_orchestrator_support import CompleteProductionError, file_sha256
from .runtime_manager import MinecraftRuntimeManager


def runtime_visual_download_artifacts(
    visual_receipt: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(visual_receipt, dict):
        return {}
    screenshots = visual_receipt.get("runtime_screenshots")
    if not isinstance(screenshots, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(screenshots, start=1):
        if not isinstance(item, dict):
            continue
        raw = item.get("evidence_path")
        digest = item.get("sha256")
        if not isinstance(raw, str) or not isinstance(digest, str):
            continue
        path = Path(raw).expanduser().resolve()
        suffix = path.suffix.lower()
        result[f"runtime-screenshot-{index:03d}{suffix}"] = {
            "path": str(path),
            "sha256": digest,
        }
    return result


def collect_runtime_screenshot_receipts(
    runtime_manager: MinecraftRuntimeManager,
    explicit_paths: Iterable[str],
    *,
    evidence_root: Path | None = None,
) -> list[dict[str, Any]]:
    status = runtime_manager.status()
    instance_raw = status.get("instance_root")
    if (
        not isinstance(instance_raw, str)
        or status.get("server_running") is not True
        or status.get("client_running") is not True
    ):
        raise CompleteProductionError(
            "Runtime visual evidence requires a live server and client."
        )

    client_root = (Path(instance_raw).expanduser().resolve() / "client").resolve()
    explicit = tuple(str(value) for value in explicit_paths if str(value).strip())
    if explicit:
        candidates = [Path(value).expanduser().resolve() for value in explicit]
    else:
        screenshots_root = client_root / "screenshots"
        candidates = (
            [
                path.resolve()
                for path in sorted(screenshots_root.rglob("*"))
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if screenshots_root.is_dir()
            else []
        )
    if not candidates:
        raise CompleteProductionError(
            "No screenshots were produced by the current disposable runtime client."
        )

    receipts: list[dict[str, Any]] = []
    for path in candidates:
        try:
            path.relative_to(client_root)
        except ValueError as exc:
            raise CompleteProductionError(
                "Visual evidence must come from the current disposable client directory."
            ) from exc

        receipt = runtime_manager.register_screenshot(path)
        if (
            receipt.get("server_running") is not True
            or receipt.get("client_running") is not True
            or not isinstance(receipt.get("sha256"), str)
        ):
            raise CompleteProductionError(
                "Runtime screenshot receipt is not bound to a live client session."
            )

        expected_sha = str(receipt["sha256"])
        digest = expected_sha.removeprefix("sha256:")
        preserved_root = (
            evidence_root.expanduser().resolve()
            if evidence_root is not None
            else (
                Path(runtime_manager.workspace_root).expanduser().resolve()
                / "integration-evidence"
                / "runtime-screenshots"
            )
        )
        preserved_root.mkdir(parents=True, exist_ok=True)
        evidence_path = preserved_root / (digest + path.suffix.lower())

        if evidence_path.exists():
            if (
                evidence_path.is_symlink()
                or not evidence_path.is_file()
                or file_sha256(evidence_path) != expected_sha
            ):
                raise CompleteProductionError(
                    "Existing runtime screenshot evidence does not match its digest."
                )
        else:
            shutil.copy2(path, evidence_path)
            if file_sha256(evidence_path) != expected_sha:
                evidence_path.unlink(missing_ok=True)
                raise CompleteProductionError(
                    "Runtime screenshot changed while preserving visual evidence."
                )

        receipts.append(
            {
                **receipt,
                "runtime_source_path": str(path),
                "path": str(evidence_path),
                "evidence_path": str(evidence_path),
            }
        )
    return receipts


def visual_runtime_evidence_passed(
    visual_receipt: dict[str, Any] | None,
    runtime_receipt: dict[str, Any] | None,
) -> bool:
    if (
        not isinstance(visual_receipt, dict)
        or visual_receipt.get("status") != "PASS"
        or not isinstance(runtime_receipt, dict)
    ):
        return False
    artifact_sha = runtime_receipt.get("artifact_sha256")
    if (
        not isinstance(artifact_sha, str)
        or visual_receipt.get("artifact_sha256") != artifact_sha
    ):
        return False
    screenshots = visual_receipt.get("runtime_screenshots")
    return (
        isinstance(screenshots, list)
        and bool(screenshots)
        and all(
            isinstance(item, dict)
            and item.get("server_running") is True
            and item.get("client_running") is True
            and isinstance(item.get("sha256"), str)
            and bool(item.get("sha256"))
            and isinstance(item.get("evidence_path"), str)
            and Path(str(item["evidence_path"])).is_file()
            and not Path(str(item["evidence_path"])).is_symlink()
            and file_sha256(Path(str(item["evidence_path"]))) == item.get("sha256")
            for item in screenshots
        )
    )


def persisted_runtime_evidence(
    runtime_receipt: dict[str, Any] | None,
    *,
    required: bool,
    artifact_sha256: str,
) -> dict[str, Any]:
    if isinstance(runtime_receipt, dict):
        return dict(runtime_receipt)
    return {
        "schema_version": "mmm/final-runtime-receipt-v1",
        "status": "REQUIRED_NOT_RUN" if required else "NOT_REQUIRED",
        "artifact_sha256": artifact_sha256,
    }


def refresh_runtime_receipt_status(
    receipt: dict[str, Any],
    live_status: dict[str, Any],
    *,
    require_client: bool,
) -> dict[str, Any]:
    server = dict(receipt.get("server") or {})
    server["server_running"] = live_status.get("server_running") is True
    if "server_log_lines" in live_status:
        server["server_log_lines"] = live_status.get("server_log_lines")

    client_value = receipt.get("client")
    client = dict(client_value) if isinstance(client_value, dict) else None
    if client is not None:
        client["client_running"] = live_status.get("client_running") is True
        if "client_log_lines" in live_status:
            client["client_log_lines"] = live_status.get("client_log_lines")

    terminal_ok = server.get("server_running") is True and (
        not require_client
        or (client is not None and client.get("client_running") is True)
    )
    return {
        **receipt,
        "status": "PASS" if terminal_ok else "FAIL",
        "server": server,
        "client": client,
        "final_status": dict(live_status),
    }


def playtest_evidence_passed(
    playtest_receipt: dict[str, Any] | None,
    expected_acceptance_tests: Iterable[str] = (),
) -> bool:
    if (
        not isinstance(playtest_receipt, dict)
        or playtest_receipt.get("status") != "PASS"
        or int(playtest_receipt.get("interaction_count", 0)) <= 0
        or int(playtest_receipt.get("assertion_count", 0)) <= 0
    ):
        return False

    expected = tuple(str(value) for value in expected_acceptance_tests)
    if not expected:
        return True
    return (
        playtest_receipt.get("acceptance_tests") == list(expected)
        and playtest_receipt.get("covered_acceptance_tests") == list(expected)
        and isinstance(playtest_receipt.get("acceptance_test_results"), list)
        and {
            str(item.get("test"))
            for item in playtest_receipt["acceptance_test_results"]
            if isinstance(item, dict) and item.get("status") == "PASS"
        }
        >= set(expected)
    )


def runtime_verification_passed(
    *,
    required: bool,
    runtime_receipt: dict[str, Any] | None,
    playtest_receipt: dict[str, Any] | None,
    visual_receipt: dict[str, Any] | None,
    expected_acceptance_tests: Iterable[str] = (),
) -> bool:
    if not required:
        return True
    if (
        not isinstance(runtime_receipt, dict)
        or runtime_receipt.get("status") != "PASS"
    ):
        return False

    server = runtime_receipt.get("server")
    client = runtime_receipt.get("client")
    if not (
        isinstance(server, dict)
        and server.get("server_running") is True
        and isinstance(client, dict)
        and client.get("client_running") is True
    ):
        return False

    if (
        not isinstance(playtest_receipt, dict)
        or playtest_receipt.get("artifact_sha256")
        != runtime_receipt.get("artifact_sha256")
    ):
        return False

    prepared = runtime_receipt.get("prepared")
    if (
        isinstance(prepared, dict)
        and isinstance(prepared.get("instance_root"), str)
        and playtest_receipt.get("runtime_instance_root")
        != prepared.get("instance_root")
    ):
        return False

    if not playtest_evidence_passed(
        playtest_receipt,
        expected_acceptance_tests,
    ):
        return False
    return visual_runtime_evidence_passed(visual_receipt, runtime_receipt)


__all__ = [
    "collect_runtime_screenshot_receipts",
    "persisted_runtime_evidence",
    "playtest_evidence_passed",
    "refresh_runtime_receipt_status",
    "runtime_verification_passed",
    "runtime_visual_download_artifacts",
    "visual_runtime_evidence_passed",
]
