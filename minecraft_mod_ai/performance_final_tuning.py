from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


_STAGE_TEXT_SUFFIXES = frozenset(
    {
        ".java",
        ".json",
        ".mcmeta",
        ".gradle",
        ".properties",
        ".accesswidener",
        ".mixins",
        ".toml",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
    }
)
_STAGE_TEXT_NAMES = frozenset(
    {
        "build.gradle",
        "settings.gradle",
        "gradle.properties",
        "fabric.mod.json",
    }
)
_STAGE_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".gradle",
        ".cache",
        ".minecraft_ai",
        "build",
        "logs",
        "node_modules",
        "run",
    }
)


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

        # Custom generation reads ProjectIndex source context, not compiled output,
        # textures, audio, caches or audit metadata. Mirror the indexable text tree
        # only. This turns staging setup from project-size copying into source-size
        # copying and avoids recursively scanning Gradle/runtime output.
        for directory, dirnames, filenames in os.walk(
            live_root,
            topdown=True,
            followlinks=False,
        ):
            base = Path(directory)
            relative_dir = base.relative_to(live_root)
            retained: list[str] = []
            for name in dirnames:
                candidate = base / name
                if candidate.is_symlink():
                    raise performance_module.StagedCommitConflict(
                        f"Staging refused project symlink: {candidate}"
                    )
                if name not in _STAGE_IGNORED_DIRS:
                    retained.append(name)
            dirnames[:] = retained

            for name in filenames:
                source = base / name
                if source.is_symlink():
                    raise performance_module.StagedCommitConflict(
                        f"Staging refused project symlink: {source}"
                    )
                if (
                    source.suffix.lower() not in _STAGE_TEXT_SUFFIXES
                    and name not in _STAGE_TEXT_NAMES
                ):
                    continue
                target = stage / relative_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                performance_module._reflink_or_copy(
                    str(source),
                    str(target),
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
