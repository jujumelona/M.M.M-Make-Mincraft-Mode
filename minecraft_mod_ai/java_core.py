"""Persistent Java semantic owner backed by actual JDT workspace builds."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any

from .owner_rpc import OwnerRPC, OwnerRPCError
from .project_model import ProjectModelInputs, ResolvedBuildModel
from .project_mutation import mutation_owner
from .project_write_lock import project_write_lock


class JavaCoreService:
    def __init__(self) -> None:
        self._rpc: OwnerRPC | None = None
        self._workspace: tempfile.TemporaryDirectory[str] | None = None
        self._root: Path | None = None
        self._inputs: ProjectModelInputs | None = None
        self._model_revision = None
        self._model: ResolvedBuildModel | None = None
        self._revision = None
        self._lock = threading.RLock()

    def diagnostics(self, project_root: str | Path, *, relative_files=None,
                    timeout_seconds: int = 600, full_scan: bool = False) -> dict[str, Any]:
        root = Path(project_root).resolve()
        # The owner snapshot must never overlap a multi-file transaction halfway through commit.
        with self._lock, project_write_lock(root):
            try:
                return self._diagnostics(root, timeout_seconds, full_scan)
            except (OSError, ValueError, TypeError, OwnerRPCError):
                self.close()
                raise

    def _diagnostics(self, root: Path, timeout: int, full_scan: bool) -> dict[str, Any]:
        from .jvm_owner_bootstrap import owner_command

        if self._root != root:
            self.close()
            self._root = root
            self._inputs = ProjectModelInputs(root)
        if self._rpc is None:
            self._workspace = tempfile.TemporaryDirectory(prefix='mmm-jdt-core-')
            self._rpc = OwnerRPC(owner_command(Path(self._workspace.name)))
        assert self._inputs is not None
        owner = mutation_owner(root)
        revision = owner.revision
        inputs = self._inputs.revision()
        refresh = self._model is None or inputs != self._model_revision
        if refresh:
            raw = self._rpc.request('resolve', {'project_root': str(root)}, timeout=timeout)
            model = ResolvedBuildModel.from_dict(raw)
            if Path(model.project_root).resolve() != root:
                raise OwnerRPCError('Resolved model belongs to another project')
            response = self._rpc.request('open', {'model': model.to_dict()}, timeout=timeout)
            self._model = model
        else:
            changes = []
            if self._revision is not None and self._revision.owner_id == revision.owner_id:
                changes = [change.to_dict() for change in owner.changes_since(self._revision).changes]
            response = self._rpc.request('build', {'changes': changes, 'full': full_scan}, timeout=timeout)
        if response.get('complete') is not True or not isinstance(response.get('diagnostics'), list):
            raise OwnerRPCError('Incomplete JDT Core build response')
        if not response.get('session_id') or not isinstance(response.get('generation'), (int, float)):
            raise OwnerRPCError('JDT Core response lacks build identity')
        if inputs != self._inputs.revision() or owner.revision != revision:
            raise OwnerRPCError('Project changed during verification')
        if refresh:
            self._inputs.track_resolved_inputs(self._model)
        self._model_revision = self._inputs.revision()
        self._revision = revision
        diagnostics: dict[str, list[dict[str, Any]]] = {}
        for row in response['diagnostics']:
            if not isinstance(row, dict) or row.get('severity') not in {'error', 'warning', 'info'}:
                raise OwnerRPCError('Malformed JDT Core diagnostic')
            severity = {'error': 1, 'warning': 2, 'info': 3}[row['severity']]
            line = max(0, int(row.get('line', 1)) - 1)
            diagnostics.setdefault(str(row.get('uri') or row.get('path') or root.as_uri()), []).append({
                **row, 'severity': severity,
                'range': {'start': {'line': line, 'character': 0}, 'end': {'line': line, 'character': 0}},
            })
        assert self._model is not None
        return {
            'schema_version': 'mmm/java-diagnostics-v3', 'project_root': str(root),
            'verification_backend': 'jdt_core', 'verification_scope': 'full' if refresh or full_scan else 'incremental',
            'error_count': sum(row['severity'] == 1 for rows in diagnostics.values() for row in rows),
            'warning_count': sum(row['severity'] == 2 for rows in diagnostics.values() for row in rows),
            'diagnostics': diagnostics, 'complete': True, 'skipped': False,
            'project_revision': revision.to_dict(), 'model_id': self._model.model_id,
            'model_revision': self._model_revision.digest, 'session_id': response['session_id'],
            'generation': response['generation'],
        }

    def close(self) -> None:
        with self._lock:
            if self._rpc is not None:
                self._rpc.close()
                self._rpc = None
            if self._workspace is not None:
                self._workspace.cleanup()
                self._workspace = None
            self._model = None
            self._model_revision = None
            self._revision = None
