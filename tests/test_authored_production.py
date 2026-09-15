import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.custom_module_generator import _task_local_module_contract
from minecraft_mod_ai.planning_pipeline import PlanningPipeline
from minecraft_mod_ai.small_model_atomic_coder_execution import atomicize_coder_messages
from minecraft_mod_ai.work_graph import build_production_work_plan


@pytest.mark.parametrize("version", ["1.21.11", "26.2"])
def test_saved_design_compiler_preserves_target_through_coder_handoff(version):
    from minecraft_mod_ai.custom_generation_research import _target_values
    from minecraft_mod_ai.platform_catalog import adapter_for_target

    plan = AuthoredPlan(f"Create a token item for Fabric {version}", "Add one token item.")
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    adapter = adapter_for_target(version, "fabric")
    expected = (version, "fabric", adapter.yarn_mappings)
    assert _target_values(proposal.game_design) == expected
    assert _target_values(proposal.modules[0].config) == expected
    assert proposal.game_design["authored_plan"] == plan.to_dict()
    assert proposal.modules[0].config["authored_plan"] == plan.to_dict()
    assert proposal.game_design["_platform_selection"]["target"] == adapter.public_dict()


@pytest.mark.parametrize("text", [
    "우주선을 만들고 행성마다 다른 광물을 거래한다.",
    '```json\n{"capability_label":"trade"}\n```',
    "0",
    "# 설계\n" + "선원, 무기, 연료, 수리, 거래를 구현한다.\n" * 1000,
], ids=["plain", "fenced", "zero", "long"])
def test_real_compiler_hands_saved_text_to_coder_without_replanning(monkeypatch, text):
    def forbidden(*args, **kwargs):
        pytest.fail("saved design entered planner again")

    monkeypatch.setattr(PlanningPipeline, "prepare", forbidden)
    monkeypatch.setattr(PlanningPipeline, "_semantic_design", forbidden)
    router = SimpleNamespace(generate_text=forbidden, generate_tool_decision=forbidden)
    plan = AuthoredPlan("Make a space trading mod for Fabric 1.21.11", text)
    proposal = CompleteGameDesignPlanner(router).compile_for_production(plan)
    assert proposal.requested_prompt == plan.requested_prompt
    assert proposal.game_design["authored_plan"] == plan.to_dict()
    assert proposal.base_proposal.spec.contents == ()
    assert proposal.base_proposal.spec.boss is None
    graph = build_production_work_plan(proposal)
    generation = [node for node in graph.nodes if node.stage == "generate:custom"]
    assert len(generation) == 1
    module = proposal.modules[0]
    contract = _task_local_module_contract(module)
    assert contract["authored_plan"] == plan.to_dict()
    messages = [{"role": "user", "content": json.dumps({
        "phase": "implement_authored_design", "module": contract,
    }, ensure_ascii=False)}]
    assert atomicize_coder_messages(messages) == (tuple(messages),)


def test_real_orchestrator_accepts_authored_handoff(monkeypatch, tmp_path):
    from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator

    plan = AuthoredPlan("Space mod for Fabric 1.21.11", "행성과 광물 거래")
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path)

    class ReachedProjectCreation(Exception):
        pass

    def prepare(approved, **kwargs):
        assert approved.modules[0].config["authored_plan"] == plan.to_dict()
        raise ReachedProjectCreation

    monkeypatch.setattr(orchestrator, "_prepare_project", prepare)
    with pytest.raises(ReachedProjectCreation):
        orchestrator.execute(proposal, approval_hash=proposal.calculate_hash(), run_name="authored")
