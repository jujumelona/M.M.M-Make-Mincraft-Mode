from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any, Mapping

from minecraft_mod_ai.mcp_transport_pool import MCPTransportPool


class _Tracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.opens: list[str] = []
        self.closes: list[str] = []
        self.active = 0
        self.max_active = 0


class _FakeSession:
    def __init__(self, tracker: _Tracker, stage: str) -> None:
        self._tracker = tracker
        self._stage = stage

    async def list_tools(self) -> Any:
        return type("ListedTools", (), {"tools": ()})()

    async def call_tool(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._tracker.lock:
            self._tracker.active += 1
            self._tracker.max_active = max(
                self._tracker.max_active,
                self._tracker.active,
            )
        try:
            await asyncio.sleep(0.02)
            return {
                "name": name,
                "stage": self._stage,
                "arguments": dict(arguments),
            }
        finally:
            with self._tracker.lock:
                self._tracker.active -= 1


class _FakeSessionContext:
    def __init__(self, tracker: _Tracker, stage: str) -> None:
        self._tracker = tracker
        self._stage = stage
        self._session = _FakeSession(tracker, stage)

    async def __aenter__(self) -> _FakeSession:
        with self._tracker.lock:
            self._tracker.opens.append(self._stage)
        return self._session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        with self._tracker.lock:
            self._tracker.closes.append(self._stage)


class MCPTransportPoolTests(unittest.TestCase):
    def test_reuses_bounded_workers_and_preserves_parallelism(self) -> None:
        tracker = _Tracker()

        def factory(
            stage: str,
            env: Mapping[str, str],
            timeout_seconds: float,
        ) -> _FakeSessionContext:
            del env, timeout_seconds
            return _FakeSessionContext(tracker, stage)

        pool = MCPTransportPool(worker_count=2, session_factory=factory)
        try:
            async def exercise() -> None:
                await asyncio.gather(
                    *(
                        pool.call_tool(
                            stage="research",
                            env={"MMM_WORKSPACE": "/tmp/workspace"},
                            timeout_seconds=1.0,
                            name=f"read_{index}",
                            arguments={"index": index},
                        )
                        for index in range(4)
                    )
                )
                await pool.call_tool(
                    stage="research",
                    env={"MMM_WORKSPACE": "/tmp/workspace"},
                    timeout_seconds=1.0,
                    name="read_again",
                    arguments={},
                )

            asyncio.run(exercise())

            self.assertEqual(tracker.opens.count("research"), 2)
            self.assertEqual(tracker.max_active, 2)
        finally:
            pool.close()

        self.assertEqual(tracker.closes.count("research"), 2)

    def test_stage_change_recycles_a_worker_without_growing_the_pool(self) -> None:
        tracker = _Tracker()

        def factory(
            stage: str,
            env: Mapping[str, str],
            timeout_seconds: float,
        ) -> _FakeSessionContext:
            del env, timeout_seconds
            return _FakeSessionContext(tracker, stage)

        pool = MCPTransportPool(worker_count=1, session_factory=factory)
        try:
            async def exercise() -> None:
                await pool.call_tool(
                    stage="research",
                    env={"MMM_WORKSPACE": "/tmp/workspace"},
                    timeout_seconds=1.0,
                    name="read",
                    arguments={},
                )
                await pool.call_tool(
                    stage="quality",
                    env={"MMM_WORKSPACE": "/tmp/workspace"},
                    timeout_seconds=1.0,
                    name="verify",
                    arguments={},
                )

            asyncio.run(exercise())
            self.assertEqual(tracker.opens, ["research", "quality"])
            self.assertEqual(tracker.closes, ["research"])
        finally:
            pool.close()

        self.assertEqual(tracker.closes, ["research", "quality"])

    def test_worker_count_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            MCPTransportPool(worker_count=0)
        with self.assertRaises(ValueError):
            MCPTransportPool(worker_count=17)


if __name__ == "__main__":
    unittest.main()
