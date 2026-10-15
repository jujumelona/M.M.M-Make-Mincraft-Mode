"""A failed scheduler must fence its workers; healthy decode may refresh its lease."""
from __future__ import annotations

import threading
import time

import pytest

import minecraft_mod_ai.artifact_graph_executor as executor
from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.artifact_materializer import materialize_whole_file
from minecraft_mod_ai.artifact_ports import PortRegistry
from minecraft_mod_ai.model_concurrency import (
    refresh_model_execution_deadline,
    remaining_model_execution_seconds,
)


@pytest.fixture(autouse=True)
def _isolate_version_gate(monkeypatch):
    monkeypatch.setattr('minecraft_mod_ai.resolved_version_context.execution_context',
                        lambda *_: None)
    monkeypatch.setenv('MMM_ARTIFACT_MAX_WORKERS', '2')
    monkeypatch.setattr(executor, 'planning_work_unit_timeout_seconds', lambda: 0.08)


def _jobs():
    return [ArtifactJob(job_id=name, template_id='test', owner_module='test')
            for name in ('slow', 'peer')]


@pytest.mark.parametrize('late_action', ['model_capacity', 'port', 'file'])
def test_timeout_cancels_running_worker_before_late_side_effect(monkeypatch, tmp_path, late_action):
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    registry = PortRegistry()
    proceeded = []
    errors = []
    target = tmp_path / 'late.java'

    def execute(job, **_):
        if job.job_id == 'peer':
            assert started.wait(2)
            return {'status': 'PASS'}
        started.set()
        try:
            assert release.wait(2)
            if late_action == 'model_capacity':
                remaining_model_execution_seconds()
            elif late_action == 'port':
                registry.register('late', 'invalid')
            else:
                materialize_whole_file(target, 'late result')
            proceeded.append('late work')
            return {'status': 'PASS'}
        except Exception as exc:
            errors.append(exc)
            raise
        finally:
            stopped.set()

    monkeypatch.setattr(executor, '_execute_one', execute)
    try:
        with pytest.raises(executor.ArtifactGraphError, match='ARTIFACT_GRAPH_WORK_TIMEOUT'):
            executor.execute_artifact_graph(_jobs(), port_registry=registry)
    finally:
        release.set()
        assert stopped.wait(2)
    assert not proceeded
    assert errors and 'cancel' in str(errors[0]).lower()
    assert not registry.has('late')
    assert not target.exists()


def test_graph_observes_worker_progress_refreshed_deadline(monkeypatch):
    def execute(job, **_):
        if job.job_id == 'slow':
            until = time.monotonic() + 0.20
            while time.monotonic() < until:
                refresh_model_execution_deadline(0.12)
                time.sleep(0.01)
        return {'status': 'PASS'}

    monkeypatch.setattr(executor, '_execute_one', execute)
    result = executor.execute_artifact_graph(_jobs())
    assert result['status'] == 'PASS'
    assert result['completed_jobs'] == ['slow', 'peer']


def test_peer_failure_cancels_running_sibling(monkeypatch):
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    proceeded = []

    def execute(job, **_):
        if job.job_id == 'peer':
            assert started.wait(2)
            raise RuntimeError('peer failed')
        started.set()
        try:
            assert release.wait(2)
            remaining_model_execution_seconds()
            proceeded.append('late work')
            return {'status': 'PASS'}
        finally:
            stopped.set()

    monkeypatch.setattr(executor, '_execute_one', execute)
    try:
        with pytest.raises(RuntimeError, match='peer failed'):
            executor.execute_artifact_graph(_jobs())
    finally:
        release.set()
        assert stopped.wait(2)
    assert not proceeded


@pytest.mark.parametrize('kind', ['artifact', 'checkpoint'])
def test_cancellation_between_staging_and_publication_is_fenced(monkeypatch, tmp_path, kind):
    from pathlib import Path

    from minecraft_mod_ai.artifact_job_checkpoint import _record_path, _write_record

    staging, release, stopped = threading.Event(), threading.Event(), threading.Event()
    target = (tmp_path / 'late.java' if kind == 'artifact'
              else _record_path(tmp_path, 'jobs', 'late'))
    original = Path.write_bytes if kind == 'artifact' else Path.write_text

    def delayed_write(path, *args, **kwargs):
        staging.set()
        assert release.wait(2)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_bytes' if kind == 'artifact' else 'write_text', delayed_write)

    def execute(job, **_):
        if job.job_id == 'peer':
            assert staging.wait(2)
            return {'status': 'PASS'}
        try:
            if kind == 'artifact':
                materialize_whole_file(target, 'late result')
            else:
                _write_record(tmp_path, 'jobs', 'late', {'status': 'SUCCESS'})
            return {'status': 'PASS'}
        finally:
            stopped.set()

    monkeypatch.setattr(executor, '_execute_one', execute)
    try:
        with pytest.raises(executor.ArtifactGraphError, match='ARTIFACT_GRAPH_WORK_TIMEOUT'):
            executor.execute_artifact_graph(_jobs())
        assert not target.exists()
    finally:
        release.set()
        assert stopped.wait(2)
    assert not target.exists()
    assert not list(tmp_path.rglob('*.tmp'))
