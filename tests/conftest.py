from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_test_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep unit tests deterministic while production caches remain durable."""

    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path / "planner-checkpoints"))
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path / "planner-traces"))
    monkeypatch.setenv(
        "MMM_RESEARCH_CHECKPOINT_ROOT",
        str(tmp_path / "research-checkpoints"),
    )
    monkeypatch.setenv(
        "MMM_RESEARCH_DOCUMENT_DIR",
        str(tmp_path / "research-evidence"),
    )
