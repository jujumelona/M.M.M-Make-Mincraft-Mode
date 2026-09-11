from __future__ import annotations

"""Install bounded-I/O checkpoint primitives for custom coder workspaces.

Checkpoint integrity still walks every relevant path. Unchanged regular files reuse the
canonical stat-validated SHA-256 cache instead of rereading their contents after every tool
edit, and checkpoint base/stage trees use the same reflink-or-copy primitive as parallel
staging so CoW-capable hosts do not duplicate project bytes during initialization/rebase.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .research_validation_fingerprint_performance import content_digest

_DIGEST_MARKER = "_mmm_checkpoint_stat_digest_v1"
_COPY_MARKER = "_mmm_checkpoint_reflink_copy_v1"


def _digest(path: Path) -> str:
    return "sha256:" + content_digest(path).hex()


def install(custom_module_generator: Any) -> None:
    from .performance_final_contract import _reflink_or_copy

    current_tree = custom_module_generator._checkpoint_tree_state_sha256
    current_snapshot = custom_module_generator._project_snapshot
    current_initialize = custom_module_generator._initialize_generation_checkpoint
    current_rebase = custom_module_generator._rebase_generation_checkpoint

    if not (
        getattr(current_tree, _DIGEST_MARKER, False)
        and getattr(current_snapshot, _DIGEST_MARKER, False)
    ):

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

        setattr(checkpoint_tree_state_sha256, _DIGEST_MARKER, True)
        setattr(project_snapshot, _DIGEST_MARKER, True)
        checkpoint_tree_state_sha256.__wrapped__ = current_tree  # type: ignore[attr-defined]
        project_snapshot.__wrapped__ = current_snapshot  # type: ignore[attr-defined]
        custom_module_generator._checkpoint_tree_state_sha256 = checkpoint_tree_state_sha256
        custom_module_generator._project_snapshot = project_snapshot

    if not getattr(current_initialize, _COPY_MARKER, False):

        def initialize_generation_checkpoint(
            root: Path,
            checkpoint_root: Path,
            *,
            identity_sha256: str,
        ) -> Path:
            checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_root.mkdir(parents=False, exist_ok=False)
            base_root = custom_module_generator._checkpoint_base(checkpoint_root)
            staged_root = checkpoint_root / "project"
            try:
                shutil.copytree(
                    root,
                    base_root,
                    symlinks=True,
                    ignore=custom_module_generator._stage_ignore,
                    copy_function=_reflink_or_copy,
                )
                shutil.copytree(
                    base_root,
                    staged_root,
                    symlinks=True,
                    copy_function=_reflink_or_copy,
                )
                custom_module_generator._persist_generation_checkpoint(
                    checkpoint_root,
                    staged_root,
                    identity_sha256=identity_sha256,
                )
            except BaseException:
                if checkpoint_root.exists() and not checkpoint_root.is_symlink():
                    shutil.rmtree(checkpoint_root, ignore_errors=True)
                raise
            return staged_root

        setattr(initialize_generation_checkpoint, _COPY_MARKER, True)
        initialize_generation_checkpoint.__wrapped__ = current_initialize  # type: ignore[attr-defined]
        custom_module_generator._initialize_generation_checkpoint = initialize_generation_checkpoint

    if not getattr(current_rebase, _COPY_MARKER, False):

        def rebase_generation_checkpoint(
            root: Path,
            checkpoint_root: Path,
            *,
            identity_sha256: str,
        ) -> Path:
            base_root = custom_module_generator._checkpoint_base(checkpoint_root)
            staged_root = checkpoint_root / "project"
            operations = custom_module_generator._checkpoint_patch_operations(
                base_root,
                staged_root,
            )
            next_base = Path(
                tempfile.mkdtemp(prefix=".base-rebase-", dir=checkpoint_root)
            )
            next_stage = Path(
                tempfile.mkdtemp(prefix=".project-rebase-", dir=checkpoint_root)
            )
            next_base.rmdir()
            next_stage.rmdir()
            try:
                shutil.copytree(
                    root,
                    next_base,
                    symlinks=True,
                    ignore=custom_module_generator._stage_ignore,
                    copy_function=_reflink_or_copy,
                )
                shutil.copytree(
                    next_base,
                    next_stage,
                    symlinks=True,
                    copy_function=_reflink_or_copy,
                )
                if operations:
                    custom_module_generator.TransactionalSourcePatcher(next_stage).apply(
                        operations
                    )
                shutil.rmtree(base_root)
                shutil.rmtree(staged_root)
                os.replace(next_base, base_root)
                os.replace(next_stage, staged_root)
                custom_module_generator._persist_generation_checkpoint(
                    checkpoint_root,
                    staged_root,
                    identity_sha256=identity_sha256,
                )
                return staged_root
            finally:
                for temporary in (next_base, next_stage):
                    if temporary.exists() and not temporary.is_symlink():
                        shutil.rmtree(temporary, ignore_errors=True)

        setattr(rebase_generation_checkpoint, _COPY_MARKER, True)
        rebase_generation_checkpoint.__wrapped__ = current_rebase  # type: ignore[attr-defined]
        custom_module_generator._rebase_generation_checkpoint = rebase_generation_checkpoint


__all__ = ["install"]
