from __future__ import annotations

"""Install content-digest reuse on custom-coder checkpoint tree scans.

Checkpoint integrity still walks every relevant path, but unchanged regular files reuse the
canonical stat-validated SHA-256 cache instead of rereading their contents after every tool
edit. Changed files miss the cache because device/inode/size/mtime/ctime are part of the
cache identity.
"""

from pathlib import Path
from typing import Any

from .research_validation_fingerprint_performance import content_digest

_MARKER = "_mmm_checkpoint_stat_digest_v1"


def _digest(path: Path) -> str:
    return "sha256:" + content_digest(path).hex()


def install(custom_module_generator: Any) -> None:
    current_tree = custom_module_generator._checkpoint_tree_state_sha256
    current_snapshot = custom_module_generator._project_snapshot
    if getattr(current_tree, _MARKER, False) and getattr(current_snapshot, _MARKER, False):
        return

    def checkpoint_tree_state_sha256(root: Path) -> str:
        rows: list[tuple[str, str, str]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(
                part in custom_module_generator._STAGE_IGNORED_DIRS
                for part in relative.parts
            ):
                continue
            normalized = relative.as_posix()
            if path.is_symlink():
                rows.append((normalized, "symlink", str(path.readlink())))
            elif path.is_file():
                rows.append((normalized, "file", _digest(path)))
            elif path.is_dir():
                rows.append((normalized, "directory", ""))
        return custom_module_generator._sha256_json(rows)

    def project_snapshot(root: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(
                part in custom_module_generator._STAGE_IGNORED_DIRS
                for part in relative.parts
            ):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            snapshot[relative.as_posix()] = _digest(path)
        return snapshot

    setattr(checkpoint_tree_state_sha256, _MARKER, True)
    setattr(project_snapshot, _MARKER, True)
    checkpoint_tree_state_sha256.__wrapped__ = current_tree  # type: ignore[attr-defined]
    project_snapshot.__wrapped__ = current_snapshot  # type: ignore[attr-defined]
    custom_module_generator._checkpoint_tree_state_sha256 = checkpoint_tree_state_sha256
    custom_module_generator._project_snapshot = project_snapshot


__all__ = ["install"]
