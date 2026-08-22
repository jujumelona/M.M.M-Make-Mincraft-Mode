from __future__ import annotations

import pytest

from minecraft_mod_ai import mcp_transport_pool
from minecraft_mod_ai import runtime_finalization


def test_failed_late_finalization_is_process_poisoned(monkeypatch) -> None:
    calls = 0

    def fail_first_installer() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic late-finalization failure")

    monkeypatch.setattr(runtime_finalization, "_FINALIZED", False)
    monkeypatch.setattr(runtime_finalization, "_FINALIZING", False)
    monkeypatch.setattr(
        mcp_transport_pool,
        "install_agent_mcp_transport_pool",
        fail_first_installer,
    )

    with pytest.raises(RuntimeError, match="synthetic late-finalization failure"):
        runtime_finalization.finalize_runtime()

    assert calls == 1
    assert runtime_finalization._FINALIZED is False
    assert runtime_finalization._FINALIZING is True

    with pytest.raises(RuntimeError, match="restart the process"):
        runtime_finalization.finalize_runtime()

    # The failed prefix must never be replayed in the same process.
    assert calls == 1
