from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def flatten_diagnostics(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize JDT-LS URI->diagnostics and legacy list receipts to one list."""

    if not isinstance(receipt, Mapping):
        return []
    raw = receipt.get("diagnostics", {})
    if isinstance(raw, Mapping):
        return [
            dict(item)
            for group in raw.values()
            if isinstance(group, list)
            for item in group
            if isinstance(item, Mapping)
        ]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def diagnostic_errors(receipt: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Only LSP severity=1 is a blocking compile error."""

    return [
        item for item in flatten_diagnostics(receipt)
        if int(item.get("severity", 1)) == 1
    ]


def install(repair_module: Any, validation_module: Any) -> None:
    # One canonical diagnostic reader is shared by progressive validation and repair.
    validation_module._diagnostic_errors = diagnostic_errors
    diagnostic_errors._mmm_flattened_jdt_mapping = True

    cls = repair_module.RepairEngine

    def signature(evidence: dict[str, Any]) -> str:
        diagnostics = [
            {
                "path": item.get("path") or item.get("uri"),
                "message": item.get("message"),
                "code": item.get("code"),
                "severity": item.get("severity"),
            }
            for item in flatten_diagnostics(evidence.get("diagnostics"))
        ]
        build = evidence.get("build", {})
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def context(self: Any, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        diagnostic_paths: list[str] = []
        query_parts: list[str] = []
        for item in flatten_diagnostics(evidence.get("diagnostics")):
            path = item.get("path") or item.get("uri")
            if isinstance(path, str):
                diagnostic_paths.append(path)
            message = item.get("message")
            if isinstance(message, str):
                query_parts.append(message)
        build = evidence.get("build", {})
        if isinstance(build.get("error"), str):
            query_parts.append(build["error"])
        for command in build.get("commands", []):
            if isinstance(command, dict) and isinstance(command.get("log_path"), str):
                log = Path(command["log_path"])
                if log.is_file() and not log.is_symlink():
                    query_parts.append(
                        log.read_text(encoding="utf-8", errors="replace")[-32_000:]
                    )
        # RepairEngine owns one ContextVar-isolated ProjectIndex per repair call.
        # Reuse it here instead of rescanning the complete source tree on every
        # diagnostic attempt; successfully committed patch paths update it in place.
        index = repair_module.active_repair_project_index(root, self.policy)
        return {
            "manifest": index.manifest_receipt(),
            "relevant": index.select(
                query="\n".join(query_parts),
                diagnostic_paths=diagnostic_paths,
            ),
        }

    signature._mmm_flattened_jdt = True
    context._mmm_flattened_jdt = True
    context._mmm_reuses_repair_project_index = True
    cls._signature = staticmethod(signature)
    cls._context = context
