from __future__ import annotations

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.scheduler_parallel_safety_contract import _receipt_touched_paths
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
        self.events.append("cancel-check")

    def begin(self, *_args, **_kwargs):
        self.state = "running"

    def retry(self, *_args, **_kwargs):
        self.state = "pending"

    def invalidate(self, *_args, **_kwargs):
        pass

    def succeed(self, *_args, **_kwargs):
        self.events.append("ledger-succeed")
        self.state = "succeeded"

    def fail(self, *_args, **_kwargs):
        self.state = "failed"


class _Index:
    def __init__(self, ledger: _Ledger) -> None:
        self.ledger = ledger
        self.paths: tuple[str, ...] = ()

    def update_files(self, paths):
        self.paths = tuple(paths)
        self.ledger.events.append("index-update")

    def write_manifest(self):
        self.ledger.events.append("index-manifest")


def _node() -> WorkNode:
    return WorkNode(
        node_id="nested",
        stage="generate:content",
        input_hash="sha256:nested",
        dependencies=(),
        payload={"kind": "module-shard", "resource_class": "cpu_io"},
        resource_class="cpu_io",
    )


def _nested_receipt() -> dict:
    return {
        "status": "SUCCEEDED",
        "receipts": [
            {
                "status": "GENERATED",
                "touched_paths": ["src/main/java/A.java"],
            },
            {
                "status": "fabric_binding_generated",
                "files": ["src/main/java/B.java"],
                "receipts": {
                    "metadata": {
                        "status": "APPLIED",
                        "operations": [
                            {
                                "operation": "replace",
                                "path": "src/main/resources/fabric.mod.json",
                            },
                            {
                                "operation": "delete",
                                "path": "src/main/java/Old.java",
                            },
                        ],
                    }
                },
            },
        ],
    }


def test_nested_generator_receipts_expose_all_touched_source_paths() -> None:
    assert _receipt_touched_paths(_nested_receipt()) == (
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/resources/fabric.mod.json",
        "src/main/java/Old.java",
    )


def test_nested_paths_are_committed_before_node_success() -> None:
    ledger = _Ledger()
    index = _Index(ledger)

    receipt = CompleteProductionOrchestrator._run_work_node(
        ledger,
        _node(),
        action=_nested_receipt,
        validate_cached=lambda _cached: False,
        shared_index=index,
    )

    assert receipt["status"] == "SUCCEEDED"
    assert index.paths == (
        "src/main/java/A.java",
        "src/main/java/B.java",
        "src/main/resources/fabric.mod.json",
        "src/main/java/Old.java",
    )
    assert ledger.events.index("index-update") < ledger.events.index("ledger-succeed")
    assert ledger.events.index("index-manifest") < ledger.events.index("ledger-succeed")
