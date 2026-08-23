from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.evidence_first_handoff import build_evidence_first_handoff
from minecraft_mod_ai.evidence_first_planning import (
    compile_evidence_first_plan,
    validate_evidence_first_plan,
)
from minecraft_mod_ai.project_inventory import inspect_project_inventory


def _write_project(root: Path) -> None:
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/test/java/example").mkdir(parents=True)
    (root / "src/main/resources/assets/evidencee2e/models/item").mkdir(parents=True)

    (root / "settings.gradle.kts").write_text(
        'rootProject.name = "evidence-e2e"\n',
        encoding="utf-8",
    )
    (root / "gradle.properties").write_text(
        "minecraft_version=1.21.1\n"
        "loader_version=0.16.14\n"
        "org.gradle.jvmargs=-Xmx1G\n",
        encoding="utf-8",
    )
    (root / "build.gradle.kts").write_text(
        'plugins { id("fabric-loom") version "1.8-SNAPSHOT" }\n'
        "dependencies {\n"
        '    minecraft("com.mojang:minecraft:1.21.1")\n'
        '    modImplementation("net.fabricmc:fabric-loader:0.16.14")\n'
        "}\n",
        encoding="utf-8",
    )
    (root / "src/main/resources/fabric.mod.json").write_text(
        """{
  "schemaVersion": 1,
  "id": "evidencee2e",
  "version": "1.0.0",
  "name": "Evidence E2E",
  "environment": "*",
  "entrypoints": {
    "main": ["example.TradeLedger"]
  },
  "depends": {
    "fabricloader": ">=0.16.14",
    "minecraft": "1.21.1"
  }
}
""",
        encoding="utf-8",
    )
    (root / "src/main/java/example/TradeLedger.java").write_text(
        """package example;

public final class TradeLedger {
    public int balance() {
        return 7;
    }
}
""",
        encoding="utf-8",
    )
    (root / "src/test/java/example/TradeLedgerTest.java").write_text(
        """package example;

public final class TradeLedgerTest {
    public void verifiesTradeLedger() {
        new TradeLedger().balance();
    }
}
""",
        encoding="utf-8",
    )
    (
        root
        / "src/main/resources/assets/evidencee2e/models/item/trade_ledger.json"
    ).write_text(
        '{"parent":"minecraft:item/generated","textures":{"layer0":"evidencee2e:item/trade_ledger"}}\n',
        encoding="utf-8",
    )


def test_existing_project_e2e_retains_verified_components_and_generates_only_true_gap(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)
    inventory = inspect_project_inventory(tmp_path)

    assert "1.21.1" in inventory.target.minecraft_versions
    assert "fabric" in inventory.target.loaders

    prompt = "Keep TradeLedger exactly as it is. Add AuditTrail."
    design = {
        "_existing_project_inventory": inventory.to_dict(),
        "modules": [
            {
                "capability": "trade_ledger",
                "description": "Keep TradeLedger exactly as it is.",
            },
            {
                "capability": "audit_trail",
                "description": "Add AuditTrail.",
            },
        ],
        "acceptance_tests": [
            "TradeLedger remains unchanged.",
            "AuditTrail is implemented and verified.",
        ],
    }
    target_decision = {
        "target": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "java_version": "21",
            "source_api_family": "fabric_live_ai",
        },
        "preserved_existing_target": True,
        "migration_requested": False,
        "reason": "preserve inspected existing project target",
    }

    plan = compile_evidence_first_plan(
        prompt,
        design,
        target_decision=target_decision,
    )
    validate_evidence_first_plan(plan, prompt=prompt)
    handoff = build_evidence_first_handoff(plan)

    assert plan["target_decision"]["policy"] == "preserve"
    assert plan["target_decision"]["coordinates"]["minecraft_version"] == "1.21.1"
    assert plan["target_decision"]["coordinates"]["loader"] == "fabric"

    decisions = {item["capability"]: item for item in plan["reuse_decisions"]}
    assert decisions["trade_ledger"]["action"] == "retain"
    assert decisions["audit_trail"]["action"] == "fresh"

    retained_refs = set(decisions["trade_ledger"]["component_refs"])
    retained_kinds = {
        item["kind"]
        for item in plan["component_catalog"]
        if item["component_id"] in retained_refs
    }
    assert {"symbol", "resource", "test"} <= retained_kinds

    assert len(plan["gap_catalog"]) == 1
    assert plan["gap_catalog"][0]["capability"] == "audit_trail"
    assert {
        item["capability"]
        for item in plan["acceptance_release_bindings"]
        if item["status"] == "retained"
    } == {"trade_ledger"}

    assert plan["tasks"]
    assert {
        requirement_ref
        for task in plan["tasks"]
        for requirement_ref in task["requirement_refs"]
    } == {decisions["audit_trail"]["requirement_ref"]}

    assert handoff["source_plan_sha256"] == plan["plan_sha256"]
    assert {
        item["requirement_ref"] for item in handoff["retain_receipts"]
    } == {decisions["trade_ledger"]["requirement_ref"]}
    assert all(
        item["task_ref"] in {task["task_id"] for task in plan["tasks"]}
        for item in handoff["production_modules"]
    )


def test_branch_fixtures_activate_only_applicable_subsystems() -> None:
    from minecraft_mod_ai.evidence_first_planning import _branch_predicates

    cases = {
        "ui.config_screen": {"needs_client_render"},
        "performance.optimization": {"needs_mixin"},
        "worldgen.placement": {"needs_datagen", "needs_worldgen"},
        "storage.saved_state": {"needs_persistence"},
        "machine.block_entity": {"needs_registry"},
    }

    for capability, expected_active in cases.items():
        requirement = {
            "requirement_id": "req_fixture",
            "capability": capability,
            "statement": capability,
            "source_span": {"text": capability},
        }
        branches = _branch_predicates(
            [requirement],
            (),
            {"project_topology": {"loaders": ["fabric"]}},
        )
        active = {
            name for name, value in branches.items() if value["status"] == "ACTIVE"
        }
        assert active == expected_active, capability
        for name, value in branches.items():
            if name not in expected_active:
                assert value["status"] == "NOT_APPLICABLE"


def test_multiloader_fixture_activates_only_loader_leaf_for_neutral_requirement() -> None:
    from minecraft_mod_ai.evidence_first_planning import _branch_predicates

    requirement = {
        "requirement_id": "req_neutral",
        "capability": "trade_rules",
        "statement": "trade_rules",
        "source_span": {"text": "trade_rules"},
    }
    branches = _branch_predicates(
        [requirement],
        (),
        {"project_topology": {"loaders": ["fabric", "neoforge"]}},
    )

    assert {
        name for name, value in branches.items() if value["status"] == "ACTIVE"
    } == {"needs_loader_leaf"}


def test_durable_ledger_preserves_completed_unaffected_work_when_plan_grows(
    tmp_path: Path,
) -> None:
    from minecraft_mod_ai.work_graph import (
        DurableWorkLedger,
        WorkGraphPlan,
        WorkNode,
        WorkState,
    )

    def node(
        node_id: str,
        input_hash: str,
        dependencies: tuple[str, ...] = (),
    ) -> WorkNode:
        return WorkNode(
            node_id=node_id,
            stage="generate:semantic",
            input_hash=input_hash,
            dependencies=dependencies,
            payload={"kind": "semantic-task", "task_id": node_id},
        )

    first = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="proposal-sha",
        graph_hash="graph-1",
        module_count=2,
        nodes=(
            node("task-a", "input-a"),
            node("task-b", "input-b", ("task-a",)),
        ),
    )
    ledger_path = tmp_path / "work-ledger.sqlite3"
    ledger = DurableWorkLedger(ledger_path, proposal_hash="proposal-sha")
    ledger.sync_plan(first)
    ledger.begin("task-a")
    ledger.succeed("task-a", {"task_id": "task-a", "status": "done"})
    ledger.begin("task-b")
    ledger.succeed("task-b", {"task_id": "task-b", "status": "done"})

    second = WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="proposal-sha",
        graph_hash="graph-2",
        module_count=3,
        nodes=(
            node("task-a", "input-a"),
            node("task-b", "input-b-changed", ("task-a",)),
            node("task-c", "input-c", ("task-b",)),
        ),
    )
    sync = ledger.sync_plan(second)

    assert ledger.task("task-a")["state"] == WorkState.SUCCEEDED.value
    assert ledger.task("task-b")["state"] == WorkState.PENDING.value
    assert ledger.task("task-c")["state"] == WorkState.PENDING.value
    assert "task-b" in sync["invalidated_nodes"]
    assert "task-a" not in sync["invalidated_nodes"]

    resumed = DurableWorkLedger.open_existing(ledger_path)
    assert resumed.task("task-a")["state"] == WorkState.SUCCEEDED.value
    assert resumed.task("task-b")["state"] == WorkState.PENDING.value
    assert resumed.task("task-c")["state"] == WorkState.PENDING.value
