from __future__ import annotations

import asyncio
import threading

import pytest

from minecraft_mod_ai import external_mcp_bridge_safety_contract as safety
from minecraft_mod_ai import external_mcp_router as router_module


class _AliveBridge:
    def is_alive(self) -> bool:
        return True


def test_nested_call_refuses_second_bridge_when_prior_timeout_is_orphaned() -> None:
    safety.install(router_module)
    router = router_module.ExternalMCPRouter.__new__(router_module.ExternalMCPRouter)
    router._lock = threading.RLock()
    router.timeout_seconds = 1.0

    previous = safety._ORPHANED_BRIDGE
    safety._ORPHANED_BRIDGE = _AliveBridge()  # type: ignore[assignment]
    try:
        async def invoke_inside_running_loop() -> None:
            with pytest.raises(
                router_module.ExternalMCPError,
                match="refusing to leak another bridge thread",
            ):
                router._call_provider(
                    "provider",
                    {},
                    tool="lookup",
                    arguments={},
                )

        asyncio.run(invoke_inside_running_loop())
    finally:
        safety._ORPHANED_BRIDGE = previous
