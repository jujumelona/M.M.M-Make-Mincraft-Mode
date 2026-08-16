import sqlite3
import threading
import time
from pathlib import Path

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.scheduler_parallel_safety_contract import install
from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphPlan,
    WorkNode,
)

install(
    work_graph_module=work_graph_module,
    orchestrator_module=orchestrator_module,
)


def _plan(*nodes: WorkNode) -> WorkGraphPlan:
    return WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:parallel-safety",
        graph_hash="sha256:parallel-safety-graph",
        module_count=0,
        nodes=nodes,
    )


def _node(
    node_id: str,
    stage: str,
    resource_class: str,
) -> WorkNode:
    return WorkNode(
        node_id=node_id,
        stage=stage,
        input_hash=f"sha256:{node_id}",
        dependencies=(),
        payload={
            "kind": "test",
            "resource_class": resource_class,
        },
        resource_class=resource_class,
    )


def test_safety_layer_owns_fairness_without_legacy_wrapper() -> None:
    claimant = work_graph_module.DurableWorkLedger.claim_ready
    assert getattr(claimant, "_mmm_parallel_lane_claim", False)
    assert getattr(claimant, "_mmm_exact_executor_fairness", False)
    assert getattr(claimant, "_mmm_stage_lock_admission", False)
    assert getattr(claimant, "_mmm_max_efficiency_claim", False)


def test_ledger_reuses_one_sqlite_connection_per_thread(
    tmp_path: Path,
) -> None:
    plan = _plan(_node("node", "generate:content", "cpu_io"))
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)

    with ledger._connect() as first:
        first_id = id(first)
    with ledger._connect() as second:
        assert id(second) == first_id
        assert second.execute("SELECT 1").fetchone()[0] == 1

    worker_connection_ids: list[int] = []

    def worker() -> None:
        with ledger._connect() as first_worker:
            worker_connection_ids.append(id(first_worker))
        with ledger._connect() as second_worker:
            worker_connection_ids.append(id(second_worker))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(worker_connection_ids) == 2
    assert worker_connection_ids[0] == worker_connection_ids[1]
    assert worker_connection_ids[0] != first_id


def test_orchestrator_claim_does_not_overqueue_a_saturated_lane(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _node("a-image", "generate:assets", "image_gpu"),
        _node("b-image", "generate:assets", "image_gpu"),
        _node("c-cpu", "generate:content", "cpu_io"),
    )
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    stages = ("generate:assets", "generate:content")

    first = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert first is not None
    assert first["node_id"] == "a-image"

    second = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert second is not None
    assert second["node_id"] == "c-cpu"

    assert (
        ledger.claim_ready(
            "mmm-orchestrator",
            stages=stages,
            lease_seconds=60,
        )
        is None
    )
    assert ledger.task("b-image")["state"] == "pending"

    ledger.succeed("a-image", {"status": "PASS"})
    third = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert third is not None
    assert third["node_id"] == "b-image"


def test_serial_cpu_stage_does_not_occupy_multiple_workers(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _node("a-content", "generate:content", "cpu_io"),
        _node("b-content", "generate:content", "cpu_io"),
        _node("c-system", "generate:system", "cpu_io"),
    )
    ledger = DurableWorkLedger(
        tmp_path / "serial-stage.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    stages = ("generate:content", "generate:system")

    first = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert first is not None
    assert first["node_id"] == "a-content"

    second = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert second is not None
    assert second["node_id"] == "c-system"
    assert ledger.task("b-content")["state"] == "pending"

    ledger.succeed("a-content", {"status": "PASS"})
    third = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert third is not None
    assert third["node_id"] == "b-content"


def test_orchestrator_polling_heartbeats_live_leases_but_reclaims_expired(
    tmp_path: Path,
) -> None:
    plan = _plan(
        _node("a-image", "generate:assets", "image_gpu"),
        _node("b-image", "generate:assets", "image_gpu"),
    )
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    stages = ("generate:assets",)

    claimed = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert claimed is not None
    assert claimed["node_id"] == "a-image"
    owner = str(claimed["lease_owner"])
    assert owner.startswith("mmm-orchestrator:")
    assert owner != "mmm-orchestrator"

    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE tasks SET lease_until = ? WHERE node_id = ?",
            (time.time() + 5.0, "a-image"),
        )
        connection.commit()
    before = float(ledger.task("a-image")["lease_until"])

    assert (
        ledger.claim_ready(
            "mmm-orchestrator",
            stages=stages,
            lease_seconds=60,
        )
        is None
    )
    after = float(ledger.task("a-image")["lease_until"])
    assert after > before + 40.0

    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE tasks SET lease_until = ? WHERE node_id = ?",
            (time.time() - 1.0, "a-image"),
        )
        connection.commit()

    reclaimed = ledger.claim_ready(
        "mmm-orchestrator",
        stages=stages,
        lease_seconds=60,
    )
    assert reclaimed is not None
    assert reclaimed["node_id"] == "a-image"
    assert reclaimed["attempt"] == 2


class _FakeLedger:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.state = "running"

    def cached_receipt(self, *_args, **_kwargs):
        return None

    def task(self, _node_id):
        return {"state": self.state}

    def raise_if_cancelled(self):
        self.events.append("cancel-check")

    def begin(self, *_args, **_kwargs):
        self.events.append("begin")
        self.state = "running"

    def retry(self, *_args, **_kwargs):
        self.events.append("retry")
        self.state = "pending"

    def invalidate(self, *_args, **_kwargs):
        self.events.append("invalidate")

    def succeed(self, *_args, **_kwargs):
        self.events.append("ledger-succeed")
        self.state = "succeeded"

    def fail(self, *_args, **_kwargs):
        self.events.append("ledger-fail")
        self.state = "failed"


class _FakeIndex:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update_files(self, _paths):
        self.events.append("index-update")

    def write_manifest(self):
        self.events.append("index-manifest")


def test_shared_index_commit_precedes_dependency_visible_success(tmp_path: Path) -> None:
    from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator

    node = _node("node", "generate:content", "cpu_io")
    plan = _plan(node)
    ledger = DurableWorkLedger(tmp_path / "commit.sqlite", proposal_hash=plan.proposal_hash)
    ledger.sync_plan(plan)
    events: list[str] = []

    class Index:
        def update_files(self, _paths):
            assert ledger.task("node")["state"] == "running"
            events.append("index-update")

        def write_manifest(self):
            assert ledger.task("node")["state"] == "running"
            events.append("index-manifest")

    receipt = CompleteProductionOrchestrator._run_work_node(
        ledger,
        node,
        action=lambda: {
            "status": "PASS",
            "touched_paths": ["src/main/java/X.java"],
        },
        validate_cached=lambda _cached: False,
        shared_index=Index(),
    )

    assert receipt["status"] == "PASS"
    assert events == ["index-update", "index-manifest"]
    assert ledger.task("node")["state"] == "succeeded"
