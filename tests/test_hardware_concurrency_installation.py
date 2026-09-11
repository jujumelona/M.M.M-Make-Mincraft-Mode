from __future__ import annotations

from minecraft_mod_ai import hardware_concurrency_installation as hardware
from minecraft_mod_ai import scheduler_parallel_safety_contract as scheduler


def test_cpu_io_workers_scale_beyond_previous_four_worker_cap(monkeypatch):
    monkeypatch.delenv("MMM_CPU_IO_WORKERS", raising=False)

    assert hardware.recommended_cpu_io_workers(cpu_count=2) == 2
    assert hardware.recommended_cpu_io_workers(cpu_count=8) == 7
    assert hardware.recommended_cpu_io_workers(cpu_count=16) == 15
    assert hardware.recommended_cpu_io_workers(cpu_count=64) == 32


def test_cpu_io_worker_override_is_dynamic(monkeypatch):
    monkeypatch.setenv("MMM_CPU_IO_WORKERS", "12")
    hardware.install()

    assert hardware.recommended_cpu_io_workers(cpu_count=4) == 12
    assert scheduler._cpu_capacity() == 12


def test_central_ai_workers_follow_active_llama_width(monkeypatch):
    monkeypatch.delenv("MMM_CENTRAL_AI_WORKERS", raising=False)
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "6")

    assert hardware.recommended_central_ai_workers() == 6
