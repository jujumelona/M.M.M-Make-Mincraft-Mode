from __future__ import annotations

from pathlib import Path
import inspect
import time

import pytest

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.deadline_executor import ParallelExecutionTimeout
from minecraft_mod_ai.model_router import _execute_tool_waves


class _Call:
    def __init__(self, name: str) -> None:
        self.name = name
        self.arguments = {}


def test_resume_cache_tracks_touched_paths(tmp_path: Path) -> None:
    target = tmp_path / "src" / "Example.java"
    target.parent.mkdir(parents=True)
    target.write_text("class Example {}", encoding="utf-8")
    receipt = {"status": "SUCCEEDED", "touched_paths": ["src/Example.java"]}
    assert CompleteProductionOrchestrator._receipt_outputs_exist(receipt, project_root=tmp_path)
    target.unlink()
    assert not CompleteProductionOrchestrator._receipt_outputs_exist(receipt, project_root=tmp_path)


def test_parallel_read_wave_has_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_PLANNING_WORK_UNIT_TIMEOUT_SECONDS", "0.05")
    calls = (_Call("search_code_rag"), _Call("search_project_rag"))

    def execute(call: _Call):
        if call.name == "search_project_rag":
            time.sleep(0.2)
        return call, {"ok": True}

    started = time.monotonic()
    with pytest.raises(ParallelExecutionTimeout):
        _execute_tool_waves(calls, execute)
    assert time.monotonic() - started < 0.18


def test_orchestrator_generation_shutdown_is_nonblocking() -> None:
    source = inspect.getsource(CompleteProductionOrchestrator._execute_generation_work)
    assert "node_deadlines" in source
    assert "shutdown(wait=True" not in source
    assert "shutdown(wait=False" in source
    assert "lease deadline exceeded" in source
