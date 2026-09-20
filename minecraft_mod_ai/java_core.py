"""Persistent Java semantic owner backed by actual JDT workspace builds."""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .owner_rpc import OwnerRPC, OwnerRPCError
from .project_model import ProjectModelInputs, ResolvedBuildModel
from .project_mutation import mutation_owner
from .project_write_lock import project_write_lock


def _remaining_verifier_seconds(deadline: float, *, operation: str) -> float:
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError(f"JDT Core {operation} deadline exceeded")
    return remaining


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
        requested_files = self._normalize_relative_files(root, relative_files)
        if isinstance(timeout_seconds, bool):
            raise ValueError("JDT Core timeout must be a positive number")
        try:
            timeout_value = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("JDT Core timeout must be a positive number") from exc
        if timeout_value <= 0.0 or timeout_value != timeout_value:
            raise ValueError("JDT Core timeout must be a positive finite number")
        deadline = time.monotonic() + timeout_value

        # The owner snapshot must never overlap a multi-file transaction halfway through
        # commit. Lock acquisition itself consumes the same verifier deadline instead of
        # silently waiting outside the caller's budget.
        acquired = self._lock.acquire(
            timeout=_remaining_verifier_seconds(deadline, operation="service lock")
        )
        if not acquired:
            raise TimeoutError("JDT Core service lock deadline exceeded")
        try:
            with project_write_lock(
                root,
                timeout_seconds=_remaining_verifier_seconds(
                    deadline,
                    operation="project lock",
                ),
            ):
                try:
                    return self._diagnostics(
                        root,
                        deadline,
                        full_scan,
                        requested_files=requested_files,
                    )
                except (OSError, ValueError, TypeError, OwnerRPCError, TimeoutError):
                    self.close()
                    raise
        finally:
            self._lock.release()

    @staticmethod
    def _normalize_relative_files(
        root: Path,
        relative_files: Any,
    ) -> tuple[str, ...] | None:
        if relative_files is None:
            return None
        if not isinstance(relative_files, (list, tuple)) or not relative_files:
            raise ValueError("relative_files must be a non-empty sequence when supplied")
        normalized: list[str] = []
        for raw in relative_files:
            relative = str(raw or "").replace("\\", "/").strip()
            while relative.startswith("./"):
                relative = relative[2:]
            candidate = Path(relative)
            if not relative or candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"invalid relative diagnostic path: {relative!r}")
            resolved = (root / candidate).resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"diagnostic path escapes project root: {relative!r}"
                ) from exc
            normalized.append(resolved.relative_to(root).as_posix())
        return tuple(dict.fromkeys(normalized))

    def _prepare_project(self, root: Path, timeout_seconds: int | float) -> None:
        from .jvm_owner_bootstrap import owner_command

        if self._root != root:
            self.close()
            self._root = root
            self._inputs = ProjectModelInputs(root)
        if self._rpc is None:
            self._workspace = tempfile.TemporaryDirectory(prefix='mmm-jdt-core-')
            self._rpc = OwnerRPC(
                owner_command(
                    Path(self._workspace.name),
                    timeout_seconds=timeout_seconds,
                )
            )

    def _resolve_and_open(self, root: Path, timeout: float) -> dict[str, Any]:
        assert self._rpc is not None
        deadline = time.monotonic() + float(timeout)
        params = self._owner_resolve_parameters(
            root,
            timeout_seconds=_remaining_verifier_seconds(
                deadline,
                operation="Gradle model materialization",
            ),
        )
        raw = self._rpc.request(
            'resolve',
            params,
            timeout=_remaining_verifier_seconds(
                deadline,
                operation="Gradle model resolution",
            ),
        )
        model = ResolvedBuildModel.from_dict(raw)
        if Path(model.project_root).resolve() != root:
            raise OwnerRPCError('Resolved model belongs to another project')
        response = self._rpc.request(
            'open',
            {'model': model.to_dict()},
            timeout=_remaining_verifier_seconds(
                deadline,
                operation="JDT owner open",
            ),
        )
        self._model = model
        return response

    @staticmethod
    def _resolve_parameters(root: Path) -> dict[str, str]:
        from .java_lsp import _requested_project_java_major, _resolve_project_java_home
        from .platform_catalog import _project_platform_lock, adapter_from_project

        params = {'project_root': str(root)}
        # Gradle/Loom model resolution must run on the JVM that owns the Tooling API,
        # not on the project's source/compile target JDK. The daemon-side model reads
        # JavaCompile.javaCompiler and returns the exact target toolchain separately.
        # For platform-locked projects, therefore, never force execution.setJavaHome()
        # to adapter.java_version; doing so can run a modern Loom plugin on an older
        # source-target JDK (for example Loom 1.17.x on Java 17).
        platform_lock = _project_platform_lock(root)
        if platform_lock is not None:
            adapter = adapter_from_project(root)
            params['gradle_version'] = adapter.gradle
            params['gradle_sha256'] = adapter.gradle_sha256
            return params
        if os.environ.get('MMM_JAVA_VERSION', '').strip():
            major = _requested_project_java_major()
            params['java_home'] = str(_resolve_project_java_home(int(major)))
        return params

    def _owner_resolve_parameters(
        self,
        root: Path,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, str]:
        """Materialize execution dependencies after pure target resolution."""

        params = self._resolve_parameters(root)
        gradle_version = params.pop('gradle_version', None)
        gradle_sha256 = params.pop('gradle_sha256', None)
        if gradle_version is None and gradle_sha256 is None:
            return params
        if not gradle_version or not gradle_sha256:
            raise OwnerRPCError('Pinned Gradle target coordinates are incomplete')

        from .runner import GradleRunner

        cache = Path.home() / '.cache' / 'mmm' / 'project-model-gradle'
        if timeout_seconds is None:
            executable = GradleRunner(cache).ensure_gradle(
                gradle_version,
                gradle_sha256,
            )
        else:
            bounded_timeout = max(1, int(timeout_seconds))
            executable = GradleRunner(
                cache,
                download_timeout_seconds=min(300, bounded_timeout),
            ).ensure_gradle(
                gradle_version,
                gradle_sha256,
                lock_timeout_seconds=bounded_timeout,
            )
        params['gradle_home'] = str(executable.parent.parent.resolve())
        params['gradle_user_home'] = str((cache / 'gradle-user-home').resolve())
        return params

    def _incremental_build(self, owner, revision, timeout: float, full_scan: bool) -> dict[str, Any]:
        assert self._rpc is not None
        changes = []
        if self._revision is not None and self._revision.owner_id == revision.owner_id:
            changes = [change.to_dict() for change in owner.changes_since(self._revision).changes]
        return self._rpc.request(
            'build',
            {'changes': changes, 'full': full_scan},
            timeout=timeout,
        )

    @staticmethod
    def _validate_response(response: dict[str, Any]) -> None:
        if response.get('complete') is not True or not isinstance(response.get('diagnostics'), list):
            raise OwnerRPCError('Incomplete JDT Core build response')
        if not response.get('session_id') or not isinstance(response.get('generation'), (int, float)):
            raise OwnerRPCError('JDT Core response lacks build identity')

    @staticmethod
    def _normalize_diagnostic(row: Any, root: Path) -> tuple[str, dict[str, Any]]:
        if not isinstance(row, dict) or row.get('severity') not in {'error', 'warning', 'info'}:
            raise OwnerRPCError('Malformed JDT Core diagnostic')
        severity = {'error': 1, 'warning': 2, 'info': 3}[row['severity']]
        line = max(0, int(row.get('line', 1)) - 1)
        raw_uri = str(row.get('uri') or '').strip()
        raw_path = str(row.get('path') or '').strip()
        if raw_uri:
            uri = raw_uri
        elif raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            uri = path.resolve(strict=False).as_uri()
        else:
            uri = root.as_uri()
        normalized = {
            **row,
            'severity': severity,
            'range': {
                'start': {'line': line, 'character': 0},
                'end': {'line': line, 'character': 0},
            },
        }
        return uri, normalized

    def _normalize_diagnostics(self, rows: list[Any], root: Path) -> dict[str, list[dict[str, Any]]]:
        diagnostics: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            uri, normalized = self._normalize_diagnostic(row, root)
            diagnostics.setdefault(uri, []).append(normalized)
        return diagnostics

    def _result_payload(
        self,
        root: Path,
        revision,
        response: dict[str, Any],
        diagnostics: dict[str, list[dict[str, Any]]],
        *,
        full_scope: bool,
    ) -> dict[str, Any]:
        assert self._model is not None
        assert self._model_revision is not None
        return {
            'schema_version': 'mmm/java-diagnostics-v3',
            'project_root': str(root),
            'verification_backend': 'jdt_core',
            'verification_scope': 'full' if full_scope else 'incremental',
            'error_count': sum(row['severity'] == 1 for rows in diagnostics.values() for row in rows),
            'warning_count': sum(row['severity'] == 2 for rows in diagnostics.values() for row in rows),
            'diagnostics': diagnostics,
            'complete': True,
            'skipped': False,
            'project_revision': revision.to_dict(),
            'model_id': self._model.model_id,
            'model_revision': self._model_revision.digest,
            'session_id': response['session_id'],
            'generation': response['generation'],
        }

    def _diagnostics(
        self,
        root: Path,
        deadline: float,
        full_scan: bool,
        *,
        requested_files: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        self._prepare_project(
            root,
            _remaining_verifier_seconds(
                deadline,
                operation="JVM owner bootstrap",
            ),
        )
        assert self._inputs is not None
        owner = mutation_owner(root)
        revision = owner.revision
        inputs = self._inputs.revision()
        refresh = self._model is None or inputs != self._model_revision
        response = (
            self._resolve_and_open(
                root,
                _remaining_verifier_seconds(
                    deadline,
                    operation="cold project resolution",
                ),
            )
            if refresh
            else self._incremental_build(
                owner,
                revision,
                _remaining_verifier_seconds(
                    deadline,
                    operation="incremental build",
                ),
                full_scan,
            )
        )
        self._validate_response(response)
        if inputs != self._inputs.revision() or owner.revision != revision:
            raise OwnerRPCError('Project changed during verification')
        if refresh:
            self._inputs.track_resolved_inputs(self._model)
        self._model_revision = self._inputs.revision()
        self._revision = revision
        diagnostics = self._normalize_diagnostics(response['diagnostics'], root)
        if requested_files is not None:
            requested_uris = {
                (root / relative).resolve(strict=False).as_uri()
                for relative in requested_files
            }
            diagnostics = {
                uri: rows
                for uri, rows in diagnostics.items()
                if uri in requested_uris
            }
        result = self._result_payload(
            root,
            revision,
            response,
            diagnostics,
            full_scope=refresh or full_scan,
        )
        if requested_files is not None:
            result["verification_scope"] = "targeted"
            result["relative_files"] = list(requested_files)
        return result

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
