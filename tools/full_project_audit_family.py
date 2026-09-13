from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools import full_project_audit as audit

FAMILY_NAMES = (
    "syntax",
    "versions",
    "subsystems",
    "model_registry",
    "runtime_contracts",
    "environment",
    "filesystem",
    "structured_output",
    "timeout_retry",
    "concurrency_queue",
    "pipeline",
    "smtp",
)


def _families(files: list[Path]):
    return {
        "syntax": lambda: audit.audit_syntax_and_data(files),
        "versions": audit.audit_versions,
        "subsystems": audit.audit_real_subsystems,
        "model_registry": audit.audit_model_registry,
        "runtime_contracts": audit.audit_runtime_contracts,
        "environment": audit.audit_environment_config,
        "filesystem": audit.audit_filesystem_permissions,
        "structured_output": audit.audit_structured_output,
        "timeout_retry": audit.audit_timeout_retry_primitives,
        "concurrency_queue": audit.audit_concurrency_queue,
        "pipeline": audit.audit_pipeline_wiring,
        "smtp": audit.audit_smtp,
    }


def run_family(name: str, *, output: Path) -> int:
    if name not in FAMILY_NAMES:
        raise ValueError(f"unknown audit family: {name}")
    audit.CHECKS.clear()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit.LOG_PATH.write_text("", encoding="utf-8")
    files = audit.tracked_files()
    started = time.monotonic()
    try:
        _families(files)[name]()
    except Exception as exc:
        audit._record_internal_exception(
            f"{name}_audit_internal",
            exc,
            time.monotonic() - started,
        )
    if not audit.CHECKS:
        audit.record(
            f"{name}_audit_internal",
            audit.WARN,
            "audit family produced no checks",
            time.monotonic() - started,
            category="audit-runner",
        )
    checks = [check.payload() for check in audit.CHECKS]
    failures = [check["name"] for check in checks if check["status"] == audit.FAIL]
    output.write_text(
        json.dumps(
            {
                "family": name,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "checks": checks,
                "failed_checks": failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=FAMILY_NAMES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_family(args.family, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
