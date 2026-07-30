from __future__ import annotations

from pathlib import Path

from .scale_policy import ScalePolicy
from .validator import Finding, ProjectValidator, ValidationReport


class ScalableProjectValidator:
    """Compatibility wrapper that removes old arbitrary scale findings.

    Semantic, path, JSON, resource, registry and safety findings remain unchanged.
    The legacy 4 MiB/file and 64-line arena budgets are replaced with the explicit
    host resource policy and function sharding performed by the scalable compiler.
    """

    def __init__(self, *, policy: ScalePolicy | None = None) -> None:
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.legacy = ProjectValidator()

    def validate(self, root: Path, spec) -> ValidationReport:
        root = root.resolve()
        report = self.legacy.validate(root, spec)
        findings: list[Finding] = []
        for finding in report.findings:
            if finding.code == "ARENA_COMMAND_BUDGET":
                continue
            if finding.code == "FILE_TOO_LARGE":
                path = root / finding.path
                if path.is_file() and path.stat().st_size <= self.policy.max_single_file_bytes:
                    continue
            findings.append(finding)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_size > self.policy.max_single_file_bytes:
                relative = path.relative_to(root).as_posix()
                if not any(
                    item.code == "FILE_TOO_LARGE" and item.path == relative for item in findings
                ):
                    findings.append(
                        Finding(
                            "FILE_TOO_LARGE",
                            "error",
                            relative,
                            "File exceeds MMM_MAX_SINGLE_FILE_BYTES host resource policy.",
                        )
                    )
        status = "PASS" if not any(item.severity == "error" for item in findings) else "FAIL"
        return ValidationReport(
            status=status,
            checks_run=report.checks_run,
            findings=tuple(findings),
        )
