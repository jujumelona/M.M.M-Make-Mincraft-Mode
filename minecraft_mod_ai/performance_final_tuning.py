from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


class _JsonParseFallback(Exception):
    pass


def install(performance_module: Any) -> None:
    """Tighten staging cost and concurrent-merge correctness."""

    if getattr(performance_module, "_mmm_final_tuning_installed", False):
        return

    def clone_source_snapshot(live_root: Path) -> Path:
        parent = live_root.parent / ".mmm-parallel-staging"
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix="custom-", dir=parent)
        ).resolve()
        stage.rmdir()

        skip_dirs = set(performance_module._SKIP_STAGE_DIRS)
        skip_suffixes = set(performance_module._SKIP_STAGE_SUFFIXES)

        # Walk only the tree that will actually be copied. In particular, do not
        # descend through Gradle/build/runtime output just to discover that copytree
        # will ignore it. This keeps snapshot setup proportional to source size.
        for directory, dirnames, filenames in os.walk(
            live_root,
            topdown=True,
            followlinks=False,
        ):
            base = Path(directory)
            retained: list[str] = []
            for name in dirnames:
                candidate = base / name
                if candidate.is_symlink():
                    raise performance_module.StagedCommitConflict(
                        f"Staging refused project symlink: {candidate}"
                    )
                if name not in skip_dirs:
                    retained.append(name)
            dirnames[:] = retained
            for name in filenames:
                candidate = base / name
                if candidate.is_symlink():
                    raise performance_module.StagedCommitConflict(
                        f"Staging refused project symlink: {candidate}"
                    )

        def ignore(directory: str, names: list[str]) -> set[str]:
            ignored: set[str] = set()
            base = Path(directory)
            for name in names:
                path = base / name
                if name in skip_dirs and path.is_dir():
                    ignored.add(name)
                elif path.is_file() and path.suffix.lower() in skip_suffixes:
                    ignored.add(name)
            return ignored

        shutil.copytree(
            live_root,
            stage,
            copy_function=performance_module._reflink_or_copy,
            ignore=ignore,
        )
        return stage

    def three_way_merge(
        relative: str,
        *,
        base_text: str,
        staged_text: str,
        live_text: str,
    ) -> str:
        if staged_text == base_text:
            return live_text
        if live_text == base_text:
            return staged_text
        if staged_text == live_text:
            return staged_text

        if relative.lower().endswith(".json"):
            try:
                base = json.loads(base_text)
                staged = json.loads(staged_text)
                live = json.loads(live_text)
            except json.JSONDecodeError:
                # A file carrying .json but not valid JSON can still be merged as
                # plain text. Once all three documents parse, semantic conflicts
                # must remain fail-closed instead of falling through to line merge.
                return performance_module._merge_text_lines(
                    relative,
                    base_text=base_text,
                    staged_text=staged_text,
                    live_text=live_text,
                )
            merged = performance_module._merge_json_value(
                base,
                staged,
                live,
                path=relative,
            )
            return json.dumps(
                merged,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n"

        return performance_module._merge_text_lines(
            relative,
            base_text=base_text,
            staged_text=staged_text,
            live_text=live_text,
        )

    performance_module._clone_source_snapshot = clone_source_snapshot
    performance_module._three_way_merge = three_way_merge
    performance_module._mmm_final_tuning_installed = True
