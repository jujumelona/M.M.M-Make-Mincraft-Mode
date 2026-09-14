"""Immutable resolved build facts and a single build-input revision authority."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project_mutation import is_model_input


def _digest(value: bytes) -> str:
    return 'sha256:' + hashlib.sha256(value).hexdigest()


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Resolved build model requires {key}')
    return value


def _paths(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f'Resolved build model requires {key} array')
    if any(not Path(value).is_absolute() for value in values):
        raise ValueError(f'Resolved build model {key} paths must be absolute')
    return tuple(values)


@dataclass(frozen=True)
class SourceSetModel:
    id: str
    project_path: str
    name: str
    source_roots: tuple[str, ...]
    classpath: tuple[str, ...]
    output_dirs: tuple[str, ...]
    java_home: str
    source_compatibility: str
    target_compatibility: str
    release: int | None
    compiler_args: tuple[str, ...]
    annotation_processor_path: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SourceSetModel:
        if not isinstance(raw, dict):
            raise TypeError('Source set must be an object')
        release = raw.get('release')
        if isinstance(release, float) and release.is_integer():
            release = int(release)
        if release is not None and (type(release) is not int or release < 6):
            raise ValueError('Invalid Java release')
        args = raw.get('compiler_args')
        if not isinstance(args, list) or any(not isinstance(v, str) for v in args):
            raise ValueError('Invalid compiler_args')
        java_home = _text(raw, 'java_home')
        if not Path(java_home).is_absolute():
            raise ValueError('java_home must be absolute')
        return cls(_text(raw, 'id'), _text(raw, 'project_path'), _text(raw, 'name'),
                   _paths(raw, 'source_roots'), _paths(raw, 'classpath'),
                   _paths(raw, 'output_dirs'), java_home,
                   _text(raw, 'source_compatibility'), _text(raw, 'target_compatibility'),
                   release, tuple(args), _paths(raw, 'annotation_processor_path'))


@dataclass(frozen=True)
class ResolvedBuildModel:
    project_root: str
    gradle_version: str
    source_sets: tuple[SourceSetModel, ...]
    model_id: str
    _canonical_json: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResolvedBuildModel:
        root = _text(raw, 'project_root')
        if not Path(root).is_absolute():
            raise ValueError('project_root must be absolute')
        rows = raw.get('source_sets')
        if not isinstance(rows, list) or not rows:
            raise ValueError('Resolved build model requires source_sets')
        source_sets = tuple(SourceSetModel.from_dict(row) for row in rows)
        if len({row.id for row in source_sets}) != len(source_sets):
            raise ValueError('Resolved build model has duplicate source set IDs')
        canonical = json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return cls(root, _text(raw, 'gradle_version'), source_sets,
                   _digest(canonical.encode('utf-8')), canonical)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


@dataclass(frozen=True)
class ProjectModelRevision:
    digest: str


class ProjectModelInputs:
    """Detect model changes once per owner; source consumers never decide refresh.

Only build inputs are hashed. Cached bytes are reusable only with unchanged OS
identity, size, modification and change timestamps. A baseline/restart rehashes all
model inputs. JDT owns discovery and incremental refresh of Java source files.
"""
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._files: dict[str, tuple[tuple[int, ...], str]] = {}
        self._external: set[Path] = set()

    def track_resolved_inputs(self, model: ResolvedBuildModel) -> None:
        self._external = {Path(source.java_home) / 'release' for source in model.source_sets}
        self._external.update(Path(path) for source in model.source_sets
                              for path in (*source.classpath, *source.annotation_processor_path)
                              if Path(path).is_file())
        mappings = model.to_dict().get('mappings', {})
        self._external.update(Path(mapping['path']) for mapping in mappings.values() if mapping.get('path'))

    def revision(self) -> ProjectModelRevision:
        files = {}
        excluded = {'.git', '.gradle', '.mmm', '.minecraft_ai', 'build', 'node_modules',
                    '.venv', '__pycache__', 'run', 'logs'}
        for directory, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in excluded]
            for name in names:
                path = Path(directory) / name
                relative = path.relative_to(self.root).as_posix()
                if not is_model_input(relative):
                    continue
                if path.is_symlink():
                    raise ValueError(f'Build input may not be a symlink: {relative}')
                before = path.stat()
                stamp = (before.st_dev, before.st_ino, before.st_size,
                         before.st_mtime_ns, before.st_ctime_ns)
                old = self._files.get(relative)
                if old is not None and old[0] == stamp:
                    digest = old[1]
                else:
                    digest = _digest(path.read_bytes())
                    after = path.stat()
                    if (after.st_dev, after.st_ino, after.st_size,
                        after.st_mtime_ns, after.st_ctime_ns) != stamp:
                        raise ValueError(f'Build input changed while observing it: {relative}')
                files[relative] = stamp, digest
        self._files = files
        material = json.dumps({path: row[1] for path, row in files.items()}, sort_keys=True)
        external = []
        for path in sorted(self._external):
            stat = path.stat()
            external.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        material += json.dumps(external)
        return ProjectModelRevision(_digest(material.encode('utf-8')))
