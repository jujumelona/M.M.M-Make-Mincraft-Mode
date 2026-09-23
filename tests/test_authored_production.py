import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import (
    _compile_new_authored_modules,
    materialize_authored_execution_scaffold,
)
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.custom_module_generator import _task_local_module_contract
from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    compile_direct_task_mutation_authority,
)
from minecraft_mod_ai.mutation_authority import MutationAuthorityMode
from minecraft_mod_ai.planning_pipeline import PlanningPipeline
from minecraft_mod_ai.progress_aware_tool_loop import _task_authority_context
from minecraft_mod_ai.scale_policy import ScalePolicy
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
    assert len(proposal.modules) == manifest["unit_count"]
    assert manifest["entrypoint"]["owner"] == "host_scaffold"
    assert all("authored_plan" not in module.config for module in proposal.modules)
    assert all("evidence_task" in module.config for module in proposal.modules)
    assert all(_target_values(module.config) == expected for module in proposal.modules)
    for module in proposal.modules:
        capsule = compile_task_capsule(module)
        assert capsule is not None
        assert capsule.primary_path
        assert capsule.writable_paths == (capsule.primary_path,)
        assert capsule.creatable_paths == ()
        authority = compile_direct_task_mutation_authority(module)
        assert authority is not None
        assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
        assert authority.writable_paths == (capsule.primary_path,)
        assert authority.creatable_paths == ()
        context = _task_authority_context({
            "module": _task_local_module_contract(module),
            "primary_path": capsule.primary_path,
            "writable_paths": list(capsule.writable_paths),
            "reuse_action": capsule.reuse_action,
        })
        assert context is not None
        assert context.target_path == capsule.primary_path
        assert context.is_new_file is False
        assert context.creatable_paths == ()
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
    feature_modules = proposal.modules
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
    assert manifest["entrypoint"]["path"] not in paths


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

    assert len(modules) == manifest["unit_count"]
    assert manifest["unit_count"] > 1
    assert "".join(
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["text"]
        for module in modules
    ) == text

    paths = set()
    for index, module in enumerate(modules, start=1):
        assert module.module_id == f"authored_feature_{index:03d}"
        assert module.depends_on == ()
        capsule = compile_task_capsule(module)
        assert capsule is not None
        assert len(capsule.writable_paths) == 1
        assert capsule.primary_path not in paths
        paths.add(capsule.primary_path)

    assert manifest["entrypoint"]["owner"] == "host_scaffold"
    assert manifest["entrypoint"]["path"] not in paths
    assert manifest["entrypoint"]["feature_symbols"] == [
        f"AuthoredFeature{index:03d}"
        for index in range(1, manifest["unit_count"] + 1)
    ]



def test_authored_execution_uses_markdown_sections_not_arbitrary_byte_packing():
    text = (
        "# Economy\nCredits, trade and prices.\n"
        "# Ship Building\nParts, upgrades and crew.\n"
        "# Planets\nMining, aliens and colonies.\n"
    )
    plan = AuthoredPlan("space mod", text)
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="authored_test",
        package_name="ai.minecraft.generated.authored_test",
        target={
            "minecraft_version": "1.21.11",
            "loader": "fabric",
            "mappings": "1.21.11+build.1",
        },
    )

    assert manifest["unit_count"] == 3
    assert [item["section"] for item in manifest["units"]] == [
        "Economy",
        "Ship Building",
        "Planets",
    ]
    assert [
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["section"]
        for module in modules
    ] == ["Economy", "Ship Building", "Planets"]
    assert all(module.depends_on == () for module in modules)
    assert "".join(
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["text"]
        for module in modules
    ) == text


def test_fresh_authored_work_graph_checkpoints_each_exact_task_independently(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    text = (
        "# Economy\nCredits and trade.\n"
        "# Ships\nParts and upgrades.\n"
        "# Planets\nMining and colonies.\n"
        "# Combat\nWeapons and aliens.\n"
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(
        AuthoredPlan("space mod", text)
    )

    graph = build_production_work_plan(
        proposal,
        policy=ScalePolicy(java_shard_size=48),
    )
    custom = [node for node in graph.nodes if node.stage == "generate:custom"]

    assert len(custom) == len(proposal.modules) == 4
    assert all(node.resource_class == "llm" for node in custom)
    assert all(len(node.payload["members"]) == 1 for node in custom)
    assert [
        node.payload["members"][0]["module_id"]
        for node in custom
    ] == [f"authored_feature_{index:03d}" for index in range(1, 5)]
    assert all(node.dependencies == ("prepare-project",) for node in custom)


def test_authored_scaffold_materializes_existing_exact_targets_and_host_entrypoint(tmp_path):
    plan = AuthoredPlan("Space mod for Fabric 1.21.11", "경제\n우주선\n행성")
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="authored_test",
        package_name="ai.minecraft.generated.authored_test",
        target={
            "minecraft_version": "1.21.11",
            "loader": "fabric",
            "mappings": "1.21.11+build.1",
        },
    )
    base = SimpleNamespace(
        spec=SimpleNamespace(
            mod_id="authored_test",
            package_name="ai.minecraft.generated.authored_test",
        )
    )
    proposal = SimpleNamespace(
        game_design={"_authored_execution_manifest": manifest},
        base_proposal=base,
    )
    main = (
        tmp_path
        / "src/main/java/ai/minecraft/generated/authored_test/AuthoredTestMod.java"
    )
    main.parent.mkdir(parents=True)
    main.write_text(
        "package ai.minecraft.generated.authored_test;\n"
        "import net.fabricmc.api.ModInitializer;\n"
        "public final class AuthoredTestMod implements ModInitializer {\n"
        "    @Override\n"
        "    public void onInitialize() {\n"
        "        // host baseline\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    assert materialize_authored_execution_scaffold(proposal, tmp_path) == tmp_path.resolve()
    first_main = main.read_text(encoding="utf-8")
    assert first_main.count("// MMM_AUTHORED_HOST_ENTRYPOINT_BINDING") == 1

    for index, module in enumerate(modules, start=1):
        capsule = compile_task_capsule(module)
        assert capsule is not None
        assert capsule.creatable_paths == ()
        target = tmp_path / capsule.primary_path
        assert target.is_file()
        source = target.read_text(encoding="utf-8")
        assert f"public final class AuthoredFeature{index:03d}" in source
        assert "public static void initialize()" in source
        call = f"AuthoredFeature{index:03d}.initialize();"
        assert first_main.count(call) == 1

    # Preparation is idempotent and must not overwrite coder-filled feature sources.
    first_capsule = compile_task_capsule(modules[0])
    assert first_capsule is not None
    first_target = tmp_path / first_capsule.primary_path
    filled = first_target.read_text(encoding="utf-8").replace(
        "// MMM_AUTHORED_FEATURE_BODY_001",
        'System.out.println("filled");',
    )
    first_target.write_text(filled, encoding="utf-8")
    materialize_authored_execution_scaffold(proposal, tmp_path)
    assert first_target.read_text(encoding="utf-8") == filled
    assert main.read_text(encoding="utf-8") == first_main


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


def test_existing_authored_plan_requires_localize_freeze_before_coder():
    plan = AuthoredPlan(
        "Modify the existing space mod",
        "# Economy\nPreserve trading and add ship upgrades.\n",
        existing_input_sha256="sha256:" + "a" * 64,
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    assert len(proposal.modules) == 1
    module = proposal.modules[0]
    assert module.module_id == "authored_existing_001"
    assert module.config["authored_localization_required"] is True
    assert "evidence_task" not in module.config
    assert module.required_gates == ("target_compile", "project build")
    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["policy"] == "host_localize_freeze_exact_targets_before_coder"
    assert manifest["units"][0]["module_id"] == "authored_existing_001"
    assert manifest["units"][0]["end_byte"] == len(plan.text.encode("utf-8"))


def test_existing_authored_design_is_semantic_and_serial_for_small_model():
    plan = AuthoredPlan(
        "Modify the existing space mod",
        (
            "# Economy\nPreserve trade and prices.\n"
            "# Ships\nAdd upgrade behavior.\n"
            "# Planets\nExtend mining and colonies.\n"
        ),
        existing_input_sha256="sha256:" + "b" * 64,
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)

    assert [module.module_id for module in proposal.modules] == [
        "authored_existing_001",
        "authored_existing_002",
        "authored_existing_003",
    ]
    assert [module.depends_on for module in proposal.modules] == [
        (),
        ("authored_existing_001",),
        ("authored_existing_002",),
    ]
    assert [
        module.config["authored_unit"]["section"] for module in proposal.modules
    ] == ["Economy", "Ships", "Planets"]
    assert "".join(
        module.config["authored_plan"]["text"] for module in proposal.modules
    ) == plan.text

    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["unit_count"] == 3
    assert [unit["module_id"] for unit in manifest["units"]] == [
        module.module_id for module in proposal.modules
    ]

    graph = build_production_work_plan(
        proposal,
        policy=ScalePolicy(java_shard_size=48),
    )
    custom = [node for node in graph.nodes if node.stage == "generate:custom"]
    assert len(custom) == 3
    assert [len(node.payload["members"]) for node in custom] == [1, 1, 1]
    assert custom[0].dependencies == ("prepare-project",)
    assert custom[1].dependencies == ("generate-custom-00000000", "prepare-project")
    assert custom[2].dependencies == ("generate-custom-00000001", "prepare-project")


def test_materialize_authored_scaffold_is_noop_without_game_design(tmp_path) -> None:
    from types import SimpleNamespace

    proposal = SimpleNamespace(
        base_proposal=SimpleNamespace(
            spec=SimpleNamespace(package_name="demo.mod")
        )
    )
    project_root = tmp_path / "project"
    project_root.mkdir()

    result = materialize_authored_execution_scaffold(proposal, project_root)

    assert result == project_root.resolve()
