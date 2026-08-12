from __future__ import annotations

import asyncio
import threading
from functools import wraps
from typing import Any, Mapping

import anyio


_BRIDGE_STATE_LOCK = threading.RLock()
_ORPHANED_BRIDGE: threading.Thread | None = None


def install(router_module: Any) -> None:
    """Allow at most one cancellation-broken nested MCP bridge thread globally.

    MMM's public API is synchronous, so a call made from an already-running asyncio
    loop needs a helper thread to host ``anyio.run``. A provider that ignores timeout
    cancellation can keep that helper thread alive after the caller's join deadline.
    The legacy path created another daemon thread on every later call, allowing an
    unbounded leak. This replacement quarantines the one timed-out bridge globally;
    while it remains alive all later nested-loop calls fail closed instead of spawning
    more orphan workers. Normal non-nested calls still execute directly via anyio.
    """

    cls = router_module.ExternalMCPRouter
    current = cls._call_provider
    if getattr(current, "_mmm_bounded_nested_bridge", False):
        return

    @wraps(current)
    def call_provider(
        self: Any,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self._call_provider_async(
                server_name,
                entry,
                tool=tool,
                arguments=arguments,
            )

        with self._lock:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return anyio.run(run)

            global _ORPHANED_BRIDGE
            with _BRIDGE_STATE_LOCK:
                if _ORPHANED_BRIDGE is not None:
                    if _ORPHANED_BRIDGE.is_alive():
                        raise router_module.ExternalMCPError(
                            "A previous external MCP nested bridge ignored cancellation "
                            "and is still running; refusing to leak another bridge thread."
                        )
                    _ORPHANED_BRIDGE = None

                value: dict[str, Any] = {}
                error: list[BaseException] = []

                def worker() -> None:
                    try:
                        value["result"] = anyio.run(run)
                    except BaseException as exc:  # pragma: no cover - thread bridge
                        error.append(exc)

                thread = threading.Thread(
                    target=worker,
                    name="mmm-external-mcp-nested-bridge",
                    daemon=True,
                )
                thread.start()
                thread.join(self.timeout_seconds + 5.0)
                if thread.is_alive():
                    _ORPHANED_BRIDGE = thread
                    raise router_module.ExternalMCPError(
                        f"External MCP {server_name} exceeded the synchronous bridge "
                        "timeout and did not terminate after cancellation."
                    )
                if error:
                    raise router_module.ExternalMCPError(str(error[0])) from error[0]
                if "result" not in value:
                    raise router_module.ExternalMCPError(
                        "External MCP nested bridge exited without a result."
                    )
                return value["result"]

    call_provider._mmm_bounded_nested_bridge = True  # type: ignore[attr-defined]
    call_provider.__wrapped__ = current  # type: ignore[attr-defined]
    cls._call_provider = call_provider


__all__ = ["install"]
