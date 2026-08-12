from __future__ import annotations

import threading
import time
from pathlib import Path

import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.project_write_lock import project_write_lock
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher
from minecraft_mod_ai.work_graph import WorkNode


class _Ledger:
    def __init__(self) -> None:
        self.state = "running"

    def cached_receipt(self, *_args, **_kwargs):
        return None

    def invalidate(self, *_args, **_kwargs):
        pass

    def task(self, _node_id):
        return {"state": self.state}

    def retry(self, *_args, **_kwargs):
        self.state = "pending"

    def raise_if_cancelled(self):
        pass

    def begin(self, *_args, **_kwargs):
        self.state = "running"

    def succeed(self, *_args, **_kwargs):
        self.state = "succeeded"

    def fail(self, *_args, **_kwargs):
        self.state = "failed"


class _Index:
    def __init__(self, root: Path) -> None:
        self.root = root

    def update_files(self, _paths):
        pass

    def write_manifest(self):
        pass


def test_transactional_patchers_serialize_same_project_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    guard = threading.Lock()
    active = 0
    max_active = 0

    def probe(self, operations):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return {"status": "APPLIED", "operations": []}

    monkeypatch.setattr(TransactionalSourcePatcher, "_apply_locked", probe)
    patchers = [TransactionalSourcePatcher(root), TransactionalSourcePatcher(root)]
    results: list[dict] = []

    threads = [
        threading.Thread(target=lambda patcher=patcher: results.append(patcher.apply([])))
        for patcher in patchers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert max_active == 1


def test_transactional_patchers_do_not_block_independent_projects(monkeypatch, tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    barrier = threading.Barrier(2)
    guard = threading.Lock()
    active = 0
    max_active = 0

    def probe(self, operations):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with guard:
            active -= 1
        return {"status": "APPLIED", "operations": []}

    monkeypatch.setattr(TransactionalSourcePatcher, "_apply_locked", probe)
    patchers = [TransactionalSourcePatcher(left), TransactionalSourcePatcher(right)]
    threads = [threading.Thread(target=patcher.apply, args=([],)) for patcher in patchers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active == 2


def test_shared_mutating_module_stages_use_commit_lane_but_custom_stays_llm() -> None:
    for stage in ("content", "system", "entity", "audio-binding"):
        node = work_graph_module._node(
            f"node-{stage}",
            f"generate:{stage}",
            (),
            {"kind": "module-shard", "generation_stage": stage},
        )
        assert node.resource_class == "commit", stage

    custom = work_graph_module._node(
        "node-custom",
        "generate:custom",
        (),
        {"kind": "module-shard", "generation_stage": "custom"},
    )
    assert custom.resource_class == "llm"

    asset = work_graph_module._node(
        "node-assets",
        "generate:assets",
        (),
        {"kind": "asset-shard"},
    )
    assert asset.resource_class == "image_gpu"


def test_commit_lane_action_holds_same_project_lock(tmp_path: Path) -> None:
    ledger = _Ledger()
    index = _Index(tmp_path)
    action_entered = threading.Event()
    release_action = threading.Event()
    competitor_entered = threading.Event()

    node = WorkNode(
        node_id="commit-node",
        stage="generate:content",
        input_hash="sha256:commit-node",
        dependencies=(),
        payload={"kind": "module-shard", "resource_class": "commit"},
        resource_class="commit",
    )

    def action() -> dict:
        action_entered.set()
        assert release_action.wait(timeout=2)
        return {"status": "PASS"}

    worker = threading.Thread(
        target=lambda: CompleteProductionOrchestrator._run_work_node(
            ledger,
            node,
            action=action,
            validate_cached=lambda _cached: False,
            shared_index=index,
        )
    )
    worker.start()
    assert action_entered.wait(timeout=2)

    def competitor() -> None:
        with project_write_lock(tmp_path):
            competitor_entered.set()

    contender = threading.Thread(target=competitor)
    contender.start()
    time.sleep(0.05)
    assert not competitor_entered.is_set()

    release_action.set()
    worker.join(timeout=2)
    contender.join(timeout=2)
    assert not worker.is_alive()
    assert not contender.is_alive()
    assert competitor_entered.is_set()
