"""Immutable, target-bound feature work orders for small-model implementation.

Work orders express WHAT needs to run, never which Java method a model should
guess. They are hash-bound to the selected Minecraft version and loader.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from collections.abc import Mapping, Sequence


def _nonempty(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("FEATURE_WORK_ORDER_" + label + "_REQUIRED")
    return text


def _path(value: object) -> str:
    path = _nonempty(value, "PATH")
    if ("\\" in path or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts):
        raise ValueError("FEATURE_WORK_ORDER_PATH_INVALID")
    return path


@dataclass(frozen=True)
class FeatureWorkOrder:
    feature_id: str
    requirement_id: str
    minecraft_version: str
    loader: str
    java_version: int
    target_path: str
    behavior_ids: tuple[str, ...]
    intents: tuple[str, ...]
    effects: tuple[str, ...]
    state_reads: tuple[str, ...]
    state_writes: tuple[str, ...]
    schema_version: str = "mmm/feature-work-order-v1"

    def __post_init__(self) -> None:
        for field in ("feature_id", "requirement_id", "minecraft_version", "loader"):
            _nonempty(getattr(self, field), field.upper())
        if type(self.java_version) is not int or self.java_version < 1:
            raise ValueError("FEATURE_WORK_ORDER_JAVA_VERSION_INVALID")
        _path(self.target_path)
        if (not self.behavior_ids or len(self.behavior_ids) != len(set(self.behavior_ids))
                or len(self.behavior_ids) != len(self.intents)
                or len(self.intents) != len(self.effects)
                or any(not value.strip() for value in (*self.intents, *self.effects))):
            raise ValueError("FEATURE_WORK_ORDER_BEHAVIOR_COVERAGE_INVALID")
        if self.schema_version != "mmm/feature-work-order-v1":
            raise ValueError("FEATURE_WORK_ORDER_SCHEMA_INVALID")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "feature_id": self.feature_id,
            "requirement_id": self.requirement_id,
            "target": {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "java_version": self.java_version,
            },
            "target_path": self.target_path,
            "behavior_ids": list(self.behavior_ids),
            "intents": list(self.intents),
            "effects": list(self.effects),
            "state_reads": list(self.state_reads),
            "state_writes": list(self.state_writes),
            "acceptance": {
                "compile": "real_gradle_compileJava",
                "integration": "linked_feature_entrypoint",
                "runtime": "required_behavior_acceptance",
                "max_compiler_repair_attempts": 2,
            },
        }

    def fingerprint(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()

    @classmethod
    def for_feature(cls, *, requirement_id: str, target: Mapping,
                    target_path: str, behaviors: Sequence[Mapping],
                    ) -> "FeatureWorkOrder":
        if not isinstance(target, Mapping) or not isinstance(behaviors, (list, tuple)):
            raise ValueError("FEATURE_WORK_ORDER_INPUT_INVALID")
        version = _nonempty(target.get("minecraft_version"), "MINECRAFT_VERSION")
        loader = _nonempty(target.get("loader"), "LOADER").casefold()
        java = target.get("java_version")
        if type(java) is not int:
            raise ValueError("FEATURE_WORK_ORDER_JAVA_VERSION_INVALID")
        selected = tuple(dict(item) for item in behaviors)
        if any(str(row.get("requirement_id") or "") != requirement_id for row in selected):
            raise ValueError("FEATURE_WORK_ORDER_SCOPE_MISMATCH")
        ids = tuple(_nonempty(row.get("behavior_id"), "BEHAVIOR_ID") for row in selected)
        if len(ids) != len(set(ids)):
            raise ValueError("FEATURE_WORK_ORDER_BEHAVIOR_DUPLICATE")
        path = _path(target_path)
        key = hashlib.sha256(json.dumps(
            [requirement_id, path, ids], ensure_ascii=False, separators=(",", ":")
        ).encode()).hexdigest()[:20]
        return cls(
            feature_id="feature_" + key, requirement_id=requirement_id,
            minecraft_version=version, loader=loader, java_version=java,
            target_path=path, behavior_ids=ids,
            intents=tuple(_nonempty(row.get("intent"), "INTENT") for row in selected),
            effects=tuple(_nonempty(row.get("effects"), "EFFECTS") for row in selected),
            state_reads=tuple(sorted({str(key).strip() for row in selected
                                      for key in row.get("state_reads", ()) if str(key).strip()})),
            state_writes=tuple(sorted({str(key).strip() for row in selected
                                       for key in row.get("state_writes", ()) if str(key).strip()})),
        )
