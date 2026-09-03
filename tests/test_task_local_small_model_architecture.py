from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from minecraft_mod_ai import runtime_bootstrap
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    _materialize_owned_reuse_context,
    _task_local_module_contract,
    _verify_reuse_application,
)


def test_evidence_task_projection_does_not_replay_unrelated_global_config() -> None:
    task = {
        "task_id": "task_example",
        "semantic_outcome": "Implement one independently verifiable outcome",
        "requirement_refs": ["req_example"],
        "owned_anchors": [{"kind": "symbol", "locator": "src/main/java/X.java#X"}],
        "reuse_refs": [],
        "consumes": ["root:example"],
        "provides": ["example"],
        "depends_on": [],
        "acceptance": ["task_example: all declared provides exist"],
        "required_gates": ["source"],
        "impact_probes": ["changed_symbols"],
        "task_sha256": "sha256:" + "0" * 64,
    }
    module = ProductionModule(
        module_id="task_example",
        kind="custom_java",
        config={
            "implementation": "custom",
            "evidence_task": task,
            "unrelated_global_design": {"huge": "must stay host-side"},
        },
    )
    projected = _task_local_module_contract(module)
    assert projected["evidence_task"] == task
    assert "config" not in projected
    assert "unrelated_global_design" not in repr(projected)


def test_runtime_bootstrap_does_not_install_heuristic_tool_top_k() -> None:
    source = inspect.getsource(runtime_bootstrap._install_post_bootstrap_contracts)
    assert "_install_tool_retrieval" not in source
    assert "_install_repair_context" in source


def test_planner_binds_verified_reuse_before_live_lowering() -> None:
    source = inspect.getsource(CompleteGameDesignPlanner._plan_in_session)

    assert source.index("proposal = bind_reuse_plan(proposal)") < source.index(
        "return lower_live_modules(self, proposal)"
    )


def test_selected_donor_becomes_bounded_code_context(monkeypatch, tmp_path: Path) -> None:
    donor_root = tmp_path / ".minecraft_ai" / "reuse" / "donors" / "donor-key"
    source_path = donor_root / "src/main/java/donor/AlienCombatCapability.java"
    source_path.parent.mkdir(parents=True)
    source_text = (
        "package donor;\n"
        "public final class AlienCombatCapability {\n"
        "  public int damage() { return 12; }\n"
        "}\n"
    )
    source_path.write_text(source_text, encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    manifest = {
        "repository": "example/alien-combat",
        "commit_sha": "a" * 40,
        "license_id": "MIT",
        "capability": "alien.combat",
        "files": [
            {
                "path": str(source_path),
                "sha256": digest,
                "size_bytes": len(source_text.encode("utf-8")),
            }
        ],
    }
    (donor_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    materialization = {
        "schema_version": "mmm/reuse-materialization-v1",
        "donors": [manifest],
        "count": 1,
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.source_transplant.materialize_source_slices",
        lambda project_root, reuse_plan: materialization,
    )
    module = ProductionModule(
        module_id="task_alien_combat",
        kind="custom_java",
        config={
            "_owned_reuse_plan": {
                "capabilities": [
                    {
                        "capability": "alien.combat",
                        "mode": "source_transplant",
                        "donor": {
                            "repository": "example/alien-combat",
                            "commit_sha": "a" * 40,
                        },
                    }
                ]
            }
        },
    )

    context = _materialize_owned_reuse_context(tmp_path, module, byte_budget=2048)

    assert context is not None
    assert context["materialization"] == materialization
    assert context["snippets"][0]["content"] == source_text
    assert context["snippets"][0]["sha256"] == digest
    assert context["bytes_used"] <= 2048
    assert "_owned_reuse_plan" not in repr(context)


def test_reuse_application_receipt_requires_donor_code_in_generated_source(tmp_path: Path) -> None:
    target = tmp_path / "src/main/java/example/Connected.java"
    target.parent.mkdir(parents=True)
    target.write_text(
        "final class Connected { TradeEngine engine; int settleTrade() { return engine.commitTrade(); } }\n",
        encoding="utf-8",
    )
    context = {
        "snippets": [
            {
                "sha256": "sha256:" + "a" * 64,
                "symbols": ["TradeEngine"],
                "content": (
                    "public final class TradeEngine { "
                    "int settleTrade() { return commitTrade(); } }"
                ),
            }
        ]
    }

    receipt = _verify_reuse_application(
        context,
        tmp_path,
        ["src/main/java/example/Connected.java"],
    )

    assert receipt["status"] == "APPLIED"
    assert receipt["matched_declared_symbols"] == ["TradeEngine"]
    assert receipt["touched_paths"] == ["src/main/java/example/Connected.java"]


def test_reuse_application_rejects_fresh_rewrite_that_ignored_donor(tmp_path: Path) -> None:
    target = tmp_path / "src/main/java/example/Unrelated.java"
    target.parent.mkdir(parents=True)
    target.write_text(
        (
            "final class Unrelated { int settleTrade; "
            "int commitTrade() { return completelyDifferent(); } }\n"
        ),
        encoding="utf-8",
    )
    context = {
        "snippets": [
            {
                "sha256": "sha256:" + "a" * 64,
                "symbols": ["TradeEngine"],
                "content": (
                    "final class TradeEngine { "
                    "int settleTrade() { return commitTrade(); } }"
                ),
            }
        ]
    }

    with pytest.raises(CustomModuleGenerationError, match="REUSE_NOT_APPLIED"):
        _verify_reuse_application(
            context,
            tmp_path,
            ["src/main/java/example/Unrelated.java"],
        )
