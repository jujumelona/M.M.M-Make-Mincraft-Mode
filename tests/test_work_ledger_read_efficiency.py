from __future__ import annotations

from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphPlan,
    WorkNode,
    WorkState,
)


def _ledger(tmp_path):
    ledger = DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash="sha256:proposal",
    )
    plan = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:proposal",
        graph_hash="sha256:graph",
        module_count=2,
        nodes=(
            WorkNode(
                node_id="a",
                stage="prepare",
                input_hash="sha256:a",
                dependencies=(),
                payload={"kind": "prepare"},
            ),
            WorkNode(
                node_id="b",
                stage="build",
                input_hash="sha256:b",
                dependencies=("a",),
                payload={"kind": "build"},
            ),
        ),
    )
    ledger.sync_plan(plan)
    return ledger


def test_task_page_uses_one_connection_and_preserves_dependencies(monkeypatch, tmp_path) -> None:
    ledger = _ledger(tmp_path)
    original_connect = ledger._connect
    opens = 0

    def counted_connect():
        nonlocal opens
        opens += 1
        return original_connect()

    monkeypatch.setattr(ledger, "_connect", counted_connect)
    page = ledger.tasks(limit=100)

    assert opens == 1
    assert [task["node_id"] for task in page["tasks"]] == ["a", "b"]
    assert page["tasks"][0]["dependencies"] == []
    assert page["tasks"][1]["dependencies"] == ["a"]
    assert page["next_cursor"] == ""


def test_task_page_filter_and_cursor_keep_original_contract(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    ledger.succeed("a", {"ok": True})

    succeeded = ledger.tasks(limit=1, state=WorkState.SUCCEEDED)
    assert [task["node_id"] for task in succeeded["tasks"]] == ["a"]
    assert succeeded["next_cursor"] == ""

    first = ledger.tasks(limit=1)
    assert [task["node_id"] for task in first["tasks"]] == ["a"]
    assert first["next_cursor"] == "a"
    second = ledger.tasks(cursor=first["next_cursor"], limit=1)
    assert [task["node_id"] for task in second["tasks"]] == ["b"]


def test_summary_reads_counts_and_metadata_with_one_connection(monkeypatch, tmp_path) -> None:
    ledger = _ledger(tmp_path)
    original_connect = ledger._connect
    opens = 0

    def counted_connect():
        nonlocal opens
        opens += 1
        return original_connect()

    monkeypatch.setattr(ledger, "_connect", counted_connect)
    summary = ledger.summary()

    assert opens == 1
    assert summary["graph_hash"] == "sha256:graph"
    assert summary["module_count"] == 2
    assert summary["task_count"] == 2
    assert summary["counts"] == {WorkState.PENDING.value: 2}
