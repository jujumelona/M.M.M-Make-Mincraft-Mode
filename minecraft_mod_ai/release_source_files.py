"""Select portable project sources and explicit build evidence for distribution."""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

_WORKSPACE_DIRS = frozenset({
    ".git", ".gradle", ".cache", "gradle-user-home", ".idea", ".settings",
    "bin", "out", "run", "runs", "releases", "__pycache__",
})
_LOCAL_METADATA_DIRS = frozenset({
    "trajectory-memory", "clean-room", ".mmm-custom-checkpoints",
})
_BUILD_EVIDENCE = frozenset({"gametest-report.xml", "mmm-gametest-attestation.xml"})


def release_source_files(root: Path, *, include_build_evidence: bool = True) -> Iterator[Path]:
    """Prune machine outputs, preserving source resources even when named build/bin."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        parts = parent.relative_to(root).parts
        in_source = "src" in parts
        dirs[:] = sorted(
            name for name in dirs
            if not (parent / name).is_symlink()
            and name != ".git"
            and (in_source or name not in _WORKSPACE_DIRS)
            and not (".minecraft_ai" in parts and name in _LOCAL_METADATA_DIRS)
            and not (not in_source and "build" in parts)
        )
        for name in sorted(files):
            path = parent / name
            if path.is_symlink() or not path.is_file():
                continue
            if not in_source:
                if name in {".classpath", ".project"} or name.endswith(".iml"):
                    continue
                if "build" in parts and not (
                    include_build_evidence and parts[-1] == "build" and name in _BUILD_EVIDENCE
                ):
                    continue
            yield path
