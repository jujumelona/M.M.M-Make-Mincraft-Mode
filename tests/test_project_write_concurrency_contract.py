from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import minecraft_mod_ai.work_graph as work_graph_module
from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator
from minecraft_mod_ai.project_edit import FabricProjectInfo, ensure_main_initializer_call
from minecraft_mod_ai.project_write_lock import project_write_lock
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher
from minecraft_mod_ai.work_graph import WorkNode


class _LedgerCursor:
    def __init__(self, *, row=None, rowcount: int = 1) -> None:
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _LedgerConnection:
    def __init__(self, ledger: "_Ledger") -> None:
        self._ledger = ledger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params=()):
        normalized = " ".join(sql.split()).upper()
        if normalized.startswith("BEGIN"):
            return _LedgerCursor()
        if normalized.startswith("SELECT STATE, ATTEMPT, LEASE_OWNER"):
            return _LedgerCursor(
                row=(self._ledger.state, self._ledger.attempt, self._ledger.lease_owner)
            )
        if normalized.startswith("UPDATE TASKS"):
            if params:
                self._ledger.state = str(params[0])
            if "LEASE_OWNER = NULL" in normalized:
                self._ledger.lease_owner = ""
            return _LedgerCursor(rowcount=1)
        raise AssertionError(f"Unexpected fenced-ledger SQL in test: {normalized}")

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class _Ledger:
    def __init__(self) -> None:
        self.state = "running"
        self.attempt = 1
        self.lease_owner = "test-worker"

    def cached_receipt(self, *_args, **_kwargs):
        return None

    def invalidate(self, *_args, **_kwargs):
        pass

    def task(self, _node_id):
        return {
            "state": self.state,
            "attempt": self.attempt,
            "lease_owner": self.lease_owner,
        }

    def retry(self, *_args, **_kwargs):
        self.state = "pending"
        self.lease_owner = ""

    def raise_if_cancelled(self):
        pass

    def begin(self, *_args, worker_id="complete-orchestrator", **_kwargs):
        self.state = "running"
        self.attempt += 1
        self.lease_owner = worker_id

    def succeed(self, *_args, **_kwargs):
        self.state = "succeeded"
        self.lease_owner = ""

    def fail(self, *_args, **_kwargs):
        self.state = "failed"
        self.lease_owner = ""

    def _connect(self):
        return _LedgerConnection(self)


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


def test_shared_initializer_edits_merge_atomically_under_parallel_generation(tmp_path: Path) -> None:
    root = tmp_path / "project"
    main_java = root / "src/main/java/example/FrostWorksMod.java"
    metadata = root / "src/main/resources/fabric.mod.json"
    main_java.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    main_java.write_text(
        "package example;\n\n"
        "public final class FrostWorksMod {\n"
        "    public void onInitialize() {\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "id": "frostworks",
                "entrypoints": {"main": ["example.FrostWorksMod"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    info = FabricProjectInfo(
        root=root,
        mod_id="frostworks",
        main_entrypoint="example.FrostWorksMod",
        package_name="example",
        main_class="FrostWorksMod",
        main_java=main_java,
        fabric_mod_json=metadata,
        main_entrypoints=("example.FrostWorksMod",),
    )
    errors: list[BaseException] = []

    def bind(import_line: str, call_line: str, marker: str) -> None:
        try:
            ensure_main_initializer_call(
                info,
                import_line=import_line,
                call_line=call_line,
                marker=marker,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(
            target=bind,
            args=("import example.ext.GeneratedExtendedContent", "GeneratedExtendedContent.register()", "extended:content"),
        ),
        threading.Thread(
            target=bind,
            args=("import example.geckolib.GeneratedGeckoEntities", "GeneratedGeckoEntities.register()", "geckolib:entities"),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    text = main_java.read_text(encoding="utf-8")
    assert "import example.ext.GeneratedExtendedContent;" in text
    assert "import example.geckolib.GeneratedGeckoEntities;" in text
    assert "// MMM:extended:content" in text
    assert "// MMM:geckolib:entities" in text


def test_deterministic_module_stages_use_cpu_lanes_but_custom_stays_llm() -> None:
    for stage in ("content", "system", "entity"):
        node = work_graph_module._node(
            f"node-{stage}",
            f"generate:{stage}",
            (),
            {"kind": "module-shard", "generation_stage": stage, "members": []},
        )
        assert node.resource_class == "cpu_io", stage

    audio_binding = work_graph_module._node(
        "node-audio-binding",
        "generate:audio-binding",
        (),
        {"kind": "module-shard", "generation_stage": "audio-binding"},
    )
    assert audio_binding.resource_class == "commit"

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
        stage="generate:audio-binding",
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
