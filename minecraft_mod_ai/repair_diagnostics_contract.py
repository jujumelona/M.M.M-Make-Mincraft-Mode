from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .validation_diagnostic_contract import diagnostic_items


def install(repair_module: Any) -> None:
    """Install repair context/index optimizations without redefining JDT semantics."""

    cls = repair_module.RepairEngine

    def signature(evidence: dict[str, Any]) -> str:
        diagnostics = [
            {
                "path": item.get("path") or item.get("uri"),
                "message": item.get("message"),
                "code": item.get("code"),
                "severity": item.get("severity"),
            }
            for item in diagnostic_items(evidence.get("diagnostics"))
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
        for item in diagnostic_items(evidence.get("diagnostics")):
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
