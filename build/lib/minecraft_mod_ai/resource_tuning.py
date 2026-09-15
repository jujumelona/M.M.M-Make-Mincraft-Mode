from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .production_hardener import harden_generated_project
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher, sha256_file


def _positive_int(value: str | None) -> int | None:
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _gradle_worker_count() -> int:
    """Use the host instead of a fixed Gradle worker cap, reserving one CPU when possible."""

    explicit = _positive_int(os.environ.get("MMM_GRADLE_MAX_WORKERS"))
    if explicit is not None:
        return explicit
    logical = max(1, int(os.cpu_count() or 1))
    return logical if logical <= 2 else logical - 1


def _upsert_property(text: str, key: str, value: str) -> str:
    rendered = f"{key}={value}"
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, rendered, text, count=1, flags=re.MULTILINE)
    prefix = "" if not text or text.startswith("\n") else "\n"
    return text.rstrip("\n") + prefix + rendered + "\n"


def tune_gradle_resources(
    project_root: str | Path,
    *,
    module_count: int,
    source_file_count: int,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    root = Path(project_root).expanduser().resolve()
    hardening = harden_generated_project(root, policy=policy)
    properties = root / "gradle.properties"
    if not properties.is_file() or properties.is_symlink():
        raise FileNotFoundError(properties)
    heap = policy.gradle_heap_mb(
        module_count=module_count,
        source_file_count=source_file_count,
    )
    workers = _gradle_worker_count()
    text = properties.read_text(encoding="utf-8")
    changed = text
    for key, value in (
        ("org.gradle.jvmargs", f"-Xmx{heap}M -Dfile.encoding=UTF-8"),
        ("org.gradle.daemon", "true"),
        ("org.gradle.parallel", "true"),
        ("org.gradle.caching", "true"),
        ("org.gradle.workers.max", str(workers)),
    ):
        changed = _upsert_property(changed, key, value)

    if changed == text:
        return {
            "status": (
                "HARDENED" if hardening.get("status") == "HARDENED" else "UNCHANGED"
            ),
            "heap_mb": heap,
            "gradle_workers": workers,
            "path": str(properties),
            "hardening": hardening,
        }
    receipt = TransactionalSourcePatcher(root).apply(
        [
            {
                "operation": "replace",
                "path": "gradle.properties",
                "expected_sha256": sha256_file(properties),
                "content": changed,
            }
        ]
    )
    return {
        "status": "TUNED",
        "heap_mb": heap,
        "gradle_workers": workers,
        "receipt": receipt,
        "hardening": hardening,
    }
