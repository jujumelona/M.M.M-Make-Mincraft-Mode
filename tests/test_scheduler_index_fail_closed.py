from __future__ import annotations

import pytest

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.complete_orchestrator_support import CompleteProductionError
from minecraft_mod_ai.work_graph import WorkNode


class _Ledger:
    def __init__(self) -> None:
        self.state = "running"
        self.events: list[str] = []

    def cached_receipt(self, *_args, **_kwargs):
        return None

    def task(self, _node_id):
        return {"state": self.state}

    def raise_if_cancelled(self):
        pass

    def begin(self, *_args, **_kwargs):
        self.state = "running"

    def retry(self, *_args, **_kwargs):
        self.state = "pending"

    def invalidate(self, *_args, **_kwargs):
        pass

    def succeed(self, *_args, **_kwargs):
        self.events.append("succeed")
        self.state = "succeeded"

    def fail(self, *_args, **_kwargs):
        self.events.append("fail")
        self.state = "failed"


class _BrokenIndex:
    def update_files(self, _paths):
        raise RuntimeError("index write failed")

    def write_manifest(self):
        raise AssertionError("manifest must not run after update failure")


def test_shared_index_failure_cannot_publish_succeeded_state() -> None:
    ledger = _Ledger()
    node = WorkNode(
        node_id="node",
        stage="generate:content",
        input_hash="sha256:test",
        dependencies=(),
        payload={"resource_class": "cpu_io"},
        resource_class="cpu_io",
    )

    with pytest.raises(CompleteProductionError, match="Shared ProjectIndex commit failed"):
        CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=lambda: {
                "status": "PASS",
                "touched_paths": ["src/main/java/X.java"],
            },
            validate_cached=lambda _cached: False,
            shared_index=_BrokenIndex(),
        )

    assert ledger.state == "failed"
    assert ledger.events == ["fail"]
