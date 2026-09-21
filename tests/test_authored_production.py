import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.custom_module_generator import _task_local_module_contract
from minecraft_mod_ai.planning_pipeline import PlanningPipeline
from minecraft_mod_ai.small_model_atomic_coder_execution import atomicize_coder_messages
from minecraft_mod_ai.small_model_task_capsule_contract import compile_task_capsule
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
    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["policy"] == "host_exact_task_queue_no_coder_file_planning"
    assert len(proposal.modules) == manifest["unit_count"] + 1
    assert proposal.modules[-1].module_id == "authored_entrypoint"
    assert proposal.modules[-1].required_gates == ("project build",)
    assert all("authored_plan" not in module.config for module in proposal.modules)
    assert all("evidence_task" in module.config for module in proposal.modules)
    assert all(_target_values(module.config) == expected for module in proposal.modules)
    for module in proposal.modules:
        capsule = compile_task_capsule(module)
        assert capsule is not None
        assert capsule.primary_path
        assert capsule.writable_paths == (capsule.primary_path,)
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

    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["unit_count"] >= 1
    feature_modules = proposal.modules[:-1]
    assert len(feature_modules) == manifest["unit_count"]
    reconstructed = "".join(
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["text"]
        for module in feature_modules
    )
    assert reconstructed == text

    graph = build_production_work_plan(proposal)
    generation = [node for node in graph.nodes if node.stage == "generate:custom"]
    assert generation

    paths = []
    for module in proposal.modules:
        contract = _task_local_module_contract(module)
        assert "evidence_task" in contract
        messages = [{"role": "user", "content": json.dumps({
            "phase": "implement_module",
            "module": contract,
        }, ensure_ascii=False)}]
        batches = atomicize_coder_messages(messages)
        assert len(batches) == 1
        payload = json.loads(batches[0][-1]["content"])
        atomic = payload["module"]["evidence_task"]["coder_execution_contract"]
        assert atomic["schema_version"] == "mmm/atomic-coder-step"
        refs = atomic["step"]["target_refs"]
        assert len(refs) == 1
        paths.append(refs[0].split("#", 1)[0])

    assert len(paths) == len(set(paths))
    assert paths[-1] == manifest["entrypoint"]["path"]


def test_fresh_authored_execution_is_exact_path_dependency_queue():
    text = ("행성 경제와 우주선 업그레이드를 구현한다.\n" * 300)
    plan = AuthoredPlan("우주 모드", text)
    package = "ai.minecraft.generated.authored_test"
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="authored_test",
        package_name=package,
        target={
            "minecraft_version": "1.21.11",
            "loader": "fabric",
            "mappings": "1.21.11+build.1",
        },
    )

    assert len(modules) == manifest["unit_count"] + 1
    assert manifest["unit_count"] > 1
    assert "".join(
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["text"]
        for module in modules[:-1]
    ) == text

    prior = ""
    paths = set()
    for index, module in enumerate(modules[:-1], start=1):
        assert module.module_id == f"authored_feature_{index:03d}"
        assert module.depends_on == ((prior,) if prior else ())
        capsule = compile_task_capsule(module)
        assert capsule is not None
        assert len(capsule.writable_paths) == 1
        assert capsule.primary_path not in paths
        paths.add(capsule.primary_path)
        prior = module.module_id

    entry = modules[-1]
    assert entry.module_id == "authored_entrypoint"
    assert entry.depends_on == (prior,)
    entry_capsule = compile_task_capsule(entry)
    assert entry_capsule is not None
    assert entry_capsule.primary_path == manifest["entrypoint"]["path"]
    assert entry_capsule.primary_symbol == manifest["entrypoint"]["symbol"]
    assert entry_capsule.primary_path not in paths


def test_real_orchestrator_accepts_authored_handoff(monkeypatch, tmp_path):
    from minecraft_mod_ai.complete_orchestrator import (
        CompleteExecutionOptions,
        CompleteProductionOrchestrator,
    )

    plan = AuthoredPlan("Space mod for Fabric 1.21.11", "행성과 광물 거래")
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    assert proposal.external_runtime_required is False
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path)

    class ReachedProjectCreation(Exception):
        pass

    def prepare(approved, **kwargs):
        assert approved.game_design["authored_plan"] == plan.to_dict()
        assert "_authored_execution_manifest" in approved.game_design
        assert all("evidence_task" in module.config for module in approved.modules)
        raise ReachedProjectCreation

    monkeypatch.setattr(orchestrator, "_prepare_project", prepare)
    with pytest.raises(ReachedProjectCreation):
        orchestrator.execute(
            proposal,
            approval_hash=proposal.calculate_hash(),
            run_name="authored",
            options=CompleteExecutionOptions(
                run_blockbench=False,
                run_runtime=False,
                run_client=False,
                run_mineflayer=False,
                run_visual_review=False,
            ),
        )
