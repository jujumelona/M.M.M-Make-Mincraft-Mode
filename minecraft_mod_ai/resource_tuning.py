from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher, sha256_file


def tune_gradle_resources(
    project_root: str | Path,
    *,
    module_count: int,
    source_file_count: int,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    root = Path(project_root).expanduser().resolve()
    properties = root / "gradle.properties"
    if not properties.is_file() or properties.is_symlink():
        raise FileNotFoundError(properties)
    heap = policy.gradle_heap_mb(
        module_count=module_count,
        source_file_count=source_file_count,
    )
    text = properties.read_text(encoding="utf-8")
    rendered = f"org.gradle.jvmargs=-Xmx{heap}M -Dfile.encoding=UTF-8"
    if re.search(r"^org\.gradle\.jvmargs=.*$", text, flags=re.MULTILINE):
        changed = re.sub(
            r"^org\.gradle\.jvmargs=.*$",
            rendered,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        changed = rendered + "\n" + text
    if changed == text:
        return {"status": "UNCHANGED", "heap_mb": heap, "path": str(properties)}
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
    return {"status": "TUNED", "heap_mb": heap, "receipt": receipt}
