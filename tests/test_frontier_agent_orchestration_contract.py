from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from minecraft_mod_ai import model_router


@dataclass(frozen=True)
class _Call:
    name: str
    id: str
    arguments: dict[str, Any] = field(default_factory=dict)


def test_mixed_tool_batch_runs_parallel_read_waves_around_serial_barrier(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_PARALLEL_READS", "4")
    calls = (
        _Call("search_code_rag", "a"),
        _Call("search_project_rag", "b"),
        _Call("external_mcp_call", "barrier", {"max_access": "write"}),
        _Call("inspect_existing_mod", "c"),
        _Call("quality_status", "d"),
    )

    first_wave = threading.Barrier(2)
    lock = threading.Lock()
    active_reads = 0
    peak_reads = 0
    barrier_active_reads: list[int] = []

    def execute(call: _Call):
        nonlocal active_reads, peak_reads
        is_read = call.name in model_router._PARALLEL_READ_TOOLS
        if is_read:
            with lock:
                active_reads += 1
                peak_reads = max(peak_reads, active_reads)
            if call.id in {"a", "b"}:
                first_wave.wait(timeout=2.0)
            with lock:
                active_reads -= 1
        else:
            with lock:
                barrier_active_reads.append(active_reads)
        return call, {"ok": True, "tool": call.name}

    executed = model_router._execute_tool_waves(calls, execute)

    assert [call.id for call, _payload in executed] == [call.id for call in calls]
    assert peak_reads >= 2
    assert barrier_active_reads == [0]
