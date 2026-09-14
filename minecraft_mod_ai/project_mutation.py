"""Committed project changes and revision identity owned by the mutation boundary.

Hashes come from the transaction's staged bytes, never a post-write filesystem scan.
The incarnation prevents facts from a previous owner process being reused after restart.
"""
from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .project_lock_state import project_lock_state


def is_model_input(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        path.name in {'build.gradle', 'build.gradle.kts', 'settings.gradle',
                      'settings.gradle.kts', 'gradle.properties', 'pom.xml',
                      '.classpath', '.project', 'gradle.lockfile'}
        or path.suffix in {'.gradle', '.kts'}
        or bool(set(path.parts) & {'buildSrc', 'build-logic', '.settings'})
        or 'gradle' in path.parts
    )


@dataclass(frozen=True)
class FileChange:
    path: str
    operation: str
    before_sha256: str | None
    after_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class ProjectRevision:
    owner_id: str
    revision: int
    model_revision: int

    def to_dict(self) -> dict[str, Any]:
        return {'owner_id': self.owner_id, 'revision': self.revision,
                'model_revision': self.model_revision}


@dataclass(frozen=True)
class ChangeSet:
    project_root: str
    owner_id: str
    base_revision: int
    revision: int
    model_revision: int
    changes: tuple[FileChange, ...]

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': 'mmm/change-set-v1', 'project_root': self.project_root,
                'owner_id': self.owner_id, 'base_revision': self.base_revision,
                'revision': self.revision, 'model_revision': self.model_revision,
                'changes': [change.to_dict() for change in self.changes]}


class ProjectMutationOwner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock_state = project_lock_state(root)
        self._id = uuid.uuid4().hex
        self._revision = 0
        self._model_revision = 0
        self._lock = threading.RLock()
        self._changes: dict[str, tuple[int, FileChange]] = {}

    @contextmanager
    def transaction(self, paths: Iterable[str]) -> Iterator[None]:
        from .project_write_lock import project_path_write_locks

        with project_path_write_locks(self.root, paths):
            yield

    @property
    def revision(self) -> ProjectRevision:
        with self._lock:
            return ProjectRevision(self._id, self._revision, self._model_revision)

    def committed(self, changes: Iterable[FileChange]) -> ChangeSet:
        actual = tuple(change for change in changes
                       if change.before_sha256 != change.after_sha256)
        with self._lock:
            base = self._revision
            if actual:
                self._revision += 1
                if any(is_model_input(change.path) for change in actual):
                    self._model_revision += 1
                for change in actual:
                    self._changes[change.path] = self._revision, change
            return ChangeSet(str(self.root), self._id, base, self._revision,
                             self._model_revision, actual)

    def changes_since(self, revision: ProjectRevision) -> ChangeSet:
        with self._lock:
            if revision.owner_id != self._id or revision.revision > self._revision:
                raise ValueError('Project revision belongs to another owner or future')
            changes = tuple(change for _, change in sorted(
                self._changes.values(), key=lambda row: (row[0], row[1].path))
                if _ > revision.revision)
            return ChangeSet(str(self.root), self._id, revision.revision,
                             self._revision, self._model_revision, changes)

    def invalidate(self) -> None:
        """Discard identity after unprovable rollback; no prior fact remains usable."""
        with self._lock:
            self._id = uuid.uuid4().hex
            self._revision = 0
            self._model_revision = 0
            self._changes.clear()


_OWNERS: dict[Path, ProjectMutationOwner] = {}
_OWNERS_LOCK = threading.Lock()


def mutation_owner(project_root: str | Path) -> ProjectMutationOwner:
    root = Path(project_root).expanduser().resolve()
    with _OWNERS_LOCK:
        owner = _OWNERS.get(root)
        if owner is None:
            owner = ProjectMutationOwner(root)
            _OWNERS[root] = owner
        return owner