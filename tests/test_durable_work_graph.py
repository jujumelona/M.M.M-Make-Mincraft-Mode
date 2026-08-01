import json
import sqlite3
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphError,
    WorkGraphPlan,
    WorkNode,
    WorkState,
    build_production_work_plan,
    run_named_checkpoint,
)


def _proposal(module_count: int = 4):
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one scalable item"
    )
    modules = []
    for index in range(module_count):
        module_id = f"content_{index:06d}"
        dependencies = () if index == 0 else (f"content_{index - 1:06d}",)
        modules.append(
            ProductionModule(
                module_id,
                "item",
                {"display_name": f"Content {index}"},
                dependencies,
            )
        )
    return complete_proposal_from_parts(
        requested_prompt="Create a scalable content graph",
        base_proposal=base,
        game_design={"title": "Scale graph"},
        modules=tuple(modules),
        acceptance_tests=("all requested content is registered",),
    )


def test_large_proposal_becomes_more_bounded_shards_without_global_cap() -> None:
    proposal = _proposal(20_000)
    policy = ScalePolicy(java_shard_size=37)
    plan = build_production_work_plan(proposal, policy=policy)
    content = [node for node in plan.nodes if node.stage == "generate:content"]

    assert plan.module_count == 20_000
    assert len(content) == 541
    assert all(len(node.payload["members"]) <= 37 for node in content)
    assert content[0].dependencies == ("prepare-project",)
    assert content[-1].dependencies == (
        "generate-content-00000539",
        "prepare-project",
    )
    assert plan.graph_hash == build_production_work_plan(
        proposal, policy=policy
    ).graph_hash


def test_work_plan_can_bind_to_the_normalized_execution_modules() -> None:
    proposal = _proposal(4)
    selected = proposal.modules[1:]
    # Removing the first module also requires removing its now-satisfied edge,
    # matching bootstrap deduplication in the orchestrator.
    normalized = (
        ProductionModule(
            selected[0].module_id,
            selected[0].kind,
            selected[0].config,
            (),
        ),
        *selected[1:],
    )

    plan = build_production_work_plan(
        proposal,
        policy=ScalePolicy(java_shard_size=2),
        modules=normalized,
    )
    generated_ids = {
        item["module_id"]
        for node in plan.nodes
        if node.payload.get("kind") == "module-shard"
        for item in node.payload["members"]
    }

    assert plan.proposal_hash == proposal.calculate_hash()
    assert plan.module_count == 3
    assert generated_ids == {
        "content_000001",
        "content_000002",
        "content_000003",
    }


def test_ledger_resumes_and_invalidates_only_changed_node_and_descendants(
    tmp_path: Path,
) -> None:
    proposal = _proposal(80)
    plan = build_production_work_plan(
        proposal,
        policy=ScalePolicy(java_shard_size=20),
    )
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)

    claimed = ledger.claim_ready("worker-one")
    assert claimed is not None
    assert claimed["node_id"] == "prepare-project"
    ledger.succeed("prepare-project", {"project_root": "project"})

    first = ledger.claim_ready("worker-one")
    assert first is not None
    ledger.succeed(first["node_id"], {"files": ["first.java"]})
    reopened = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    assert reopened.cached_receipt(first["node_id"]) == {
        "files": ["first.java"]
    }

    affected = reopened.invalidate(first["node_id"])
    assert first["node_id"] in affected
    assert "validate-source" in affected
    assert "prepare-project" not in affected
    assert reopened.task("prepare-project")["state"] == WorkState.SUCCEEDED.value
    assert reopened.task(first["node_id"])["state"] == WorkState.PENDING.value


def test_sync_plan_atomically_updates_inputs_and_prunes_obsolete_graph_rows(
    tmp_path: Path,
) -> None:
    initial = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:proposal",
        graph_hash="sha256:graph-one",
        module_count=2,
        nodes=(
            WorkNode("root", "generate", "sha256:root-one", (), {"v": 1}),
            WorkNode(
                "child",
                "validate",
                "sha256:child",
                ("root",),
                {"v": 1},
            ),
            WorkNode(
                "obsolete",
                "generate",
                "sha256:obsolete",
                (),
                {"v": 1},
            ),
        ),
    )
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=initial.proposal_hash,
    )
    ledger.sync_plan(initial)
    ledger.begin("root")
    ledger.succeed("root", {"output": "old-root"})
    ledger.begin("child")
    ledger.succeed("child", {"output": "old-child"})

    revised = WorkGraphPlan(
        schema_version=initial.schema_version,
        proposal_hash=initial.proposal_hash,
        graph_hash="sha256:graph-two",
        module_count=2,
        nodes=(
            WorkNode("root", "generate", "sha256:root-two", (), {"v": 2}),
            initial.nodes[1],
            WorkNode(
                "new-leaf",
                "package",
                "sha256:new-leaf",
                ("child",),
                {"v": 1},
            ),
        ),
    )
    result = ledger.sync_plan(revised)

    root = ledger.task("root")
    assert root["input_hash"] == "sha256:root-two"
    assert root["payload"] == {"v": 2}
    assert root["state"] == WorkState.PENDING.value
    assert root["receipt"] is None
    assert ledger.task("child")["state"] == WorkState.PENDING.value
    assert ledger.task("new-leaf")["dependencies"] == ["child"]
    with pytest.raises(WorkGraphError, match="Unknown work node"):
        ledger.task("obsolete")
    assert result["invalidated_nodes"] == ("root",)
    assert result["pruned_nodes"] == ("obsolete",)
    assert ledger.summary()["task_count"] == 3


def test_sync_plan_rolls_back_every_graph_change_when_metadata_swap_fails(
    tmp_path: Path,
) -> None:
    initial = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:proposal",
        graph_hash="sha256:graph-one",
        module_count=1,
        nodes=(
            WorkNode("root", "generate", "sha256:root-one", (), {"v": 1}),
            WorkNode("obsolete", "generate", "sha256:old", (), {"v": 1}),
        ),
    )
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=initial.proposal_hash,
    )
    ledger.sync_plan(initial)
    ledger.begin("root")
    ledger.succeed("root", {"output": "old-root"})
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_graph_hash_swap
            BEFORE INSERT ON metadata
            WHEN NEW.key = 'graph_hash'
            BEGIN
                SELECT RAISE(ABORT, 'simulated graph metadata failure');
            END
            """
        )

    revised = WorkGraphPlan(
        schema_version=initial.schema_version,
        proposal_hash=initial.proposal_hash,
        graph_hash="sha256:graph-two",
        module_count=1,
        nodes=(
            WorkNode("root", "generate", "sha256:root-two", (), {"v": 2}),
            WorkNode("new", "package", "sha256:new", ("root",), {"v": 1}),
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="metadata failure"):
        ledger.sync_plan(revised)

    root = ledger.task("root")
    assert root["input_hash"] == "sha256:root-one"
    assert root["state"] == WorkState.SUCCEEDED.value
    assert root["receipt"] == {"output": "old-root"}
    assert ledger.task("obsolete")["input_hash"] == "sha256:old"
    with pytest.raises(WorkGraphError, match="Unknown work node"):
        ledger.task("new")
    assert ledger.summary()["graph_hash"] == "sha256:graph-one"


def test_ledger_rejects_cross_proposal_resume_and_pages_tasks(
    tmp_path: Path,
) -> None:
    first = build_production_work_plan(_proposal(5))
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=first.proposal_hash,
    )
    ledger.sync_plan(first)

    with pytest.raises(WorkGraphError):
        DurableWorkLedger(
            tmp_path / "run.sqlite",
            proposal_hash=build_production_work_plan(_proposal(6)).proposal_hash,
        )

    page = ledger.tasks(limit=2)
    assert len(page["tasks"]) == 2
    assert page["next_cursor"]
    second = ledger.tasks(cursor=page["next_cursor"], limit=2)
    assert second["tasks"]
    assert {
        item["node_id"] for item in page["tasks"]
    }.isdisjoint(item["node_id"] for item in second["tasks"])


def test_named_checkpoint_reuses_valid_output_and_rebuilds_missing_output(
    tmp_path: Path,
) -> None:
    plan = build_production_work_plan(_proposal(2))
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    target = tmp_path / "artifact.txt"
    calls = 0

    def produce() -> Path:
        nonlocal calls
        calls += 1
        target.write_text(f"attempt {calls}", encoding="utf-8")
        return target

    def execute() -> Path:
        return run_named_checkpoint(
            ledger,
            "content-generation",
            stage="generate",
            input_value={"proposal": plan.proposal_hash},
            action=produce,
            encode=lambda path: {"path": str(path)},
            decode=lambda receipt: Path(receipt["path"]),
            validate_cached=lambda path: path.is_file(),
        )

    assert execute() == target
    assert execute() == target
    assert calls == 1
    target.unlink()
    assert execute() == target
    assert calls == 2


def test_receipt_export_is_streamed_and_portable(tmp_path: Path) -> None:
    plan = build_production_work_plan(_proposal(5))
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    ledger.begin("prepare-project")
    ledger.succeed("prepare-project", {"status": "PASS"})

    target = ledger.export_receipts(tmp_path / "receipts.jsonl")
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["record_type"] == "summary"
    assert any(
        row.get("record_type") == "task"
        and row.get("node_id") == "prepare-project"
        and row.get("state") == "succeeded"
        for row in rows
    )


def test_run_cancellation_stops_checkpoints_and_can_resume(
    tmp_path: Path,
) -> None:
    plan = build_production_work_plan(_proposal(5))
    ledger = DurableWorkLedger(
        tmp_path / "run.sqlite",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    cancelled = ledger.cancel_run(reason="user requested stop")
    assert cancelled["cancel_requested"] == "user requested stop"
    assert cancelled["counts"]["cancelled"] == cancelled["task_count"]

    with pytest.raises(WorkGraphError, match="cancelled"):
        run_named_checkpoint(
            ledger,
            "after-cancel",
            stage="test",
            input_value={},
            action=lambda: "should not run",
            encode=lambda value: {"value": value},
            decode=lambda receipt: receipt["value"],
        )

    resumed = ledger.resume_run()
    assert resumed["cancel_requested"] is None
    assert run_named_checkpoint(
        ledger,
        "after-cancel",
        stage="test",
        input_value={},
        action=lambda: "resumed",
        encode=lambda value: {"value": value},
        decode=lambda receipt: receipt["value"],
    ) == "resumed"
