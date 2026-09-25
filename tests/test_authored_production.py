import hashlib
import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import (
    _authored_execution_units,
    _compile_new_authored_modules,
    _implementation_authored_plan,
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
    text = (
        "# Design\n"
        "## Wallet\nPersist each player's credit balance.\n"
        "## Purchase\nReject purchases without enough credits and deduct exactly once.\n"
        "## Reward\nGrant the purchased reward exactly once.\n"
    )
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

    assert len(modules) == manifest["unit_count"] == 3
    assert [item["section"] for item in manifest["units"]] == [
        "Wallet",
        "Purchase",
        "Reward",
    ]
    assert "".join(
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["text"]
        for module in modules
    ) == text

    paths = set()
    for index, module in enumerate(modules, start=1):
        assert module.module_id == f"authored_feature_{index:03d}"
        expected_dependency = () if index == 1 else (f"authored_feature_{index - 1:03d}",)
        assert module.depends_on == expected_dependency
        task = module.config["evidence_task"]
        expected_consumes = [] if index == 1 else [f"authored_feature_{index - 1:03d}_ready"]
        assert task["consumes"] == expected_consumes
        assert manifest["units"][index - 1]["depends_on"] == list(expected_dependency)
        assert manifest["units"][index - 1]["consumes"] == expected_consumes
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

def test_named_design_document_descends_into_feature_sections():
    text = (
        "# Galaxy Ascension: Minecraft Mod Design Document\n"
        "## Intro\n"
        "**Mod Title:** Galactic Ascension\n"
        "**Version:** 1.0.0-alpha\n"
        "**Description:** A progression mod transforming survival into space exploration.\n"
        "## Resource Economy\n"
        "Persist credits and trade mined resources.\n"
        "## Ship Construction\n"
        "Spend credits to install modular ship parts.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 2
    assert [unit["section"] for unit in units] == [
        "Resource Economy",
        "Ship Construction",
    ]
    assert units[0]["text"].startswith(
        "# Galaxy Ascension: Minecraft Mod Design Document\n"
        "## Intro\n"
    )
    assert "## Resource Economy\n" in units[0]["text"]
    assert units[0]["implementation_text"].startswith(
        "## Resource Economy\n"
    )
    assert "Galaxy Ascension: Minecraft Mod Design Document" not in units[0]["implementation_text"]
    assert "## Intro\n" not in units[0]["implementation_text"]
    assert "".join(unit["text"] for unit in units) == text

    modules, manifest = _compile_new_authored_modules(
        AuthoredPlan("space mod", text),
        mod_id="authored_test",
        package_name="ai.minecraft.generated.authored_test",
        target={
            "minecraft_version": "1.21.11",
            "loader": "fabric",
            "mappings": "1.21.11+build.1",
        },
    )
    assert manifest["unit_count"] == 2
    first_task = modules[0].config["evidence_task"]
    first_obligation = first_task["implementation_obligations"][0]
    authored_unit = first_task["engineering_worksheet"]["authored_unit"]
    assert "unit 1/2" in first_obligation
    assert "## Resource Economy\n" in first_obligation
    assert "Galaxy Ascension: Minecraft Mod Design Document" not in first_obligation
    assert "## Intro\n" not in first_obligation
    assert authored_unit["text"].startswith("# Galaxy Ascension: Minecraft Mod Design Document\n")
    assert authored_unit["implementation_text"].startswith("## Resource Economy\n")


def test_named_document_intro_plus_generic_systems_descends_to_real_features():
    text = (
        "# Galaxy Ascension: Minecraft Mod Design Document\n"
        "## Intro\n"
        "**Version:** 1.0.0-alpha\n"
        "## Gameplay Systems\n"
        "### Resource Economy\n"
        "Persist credits and trade resources.\n"
        "### Ship Construction\n"
        "Spend credits on modular ship parts.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 2
    assert [unit["section"] for unit in units] == [
        "Resource Economy",
        "Ship Construction",
    ]
    assert units[0]["implementation_text"].startswith("### Resource Economy\n")
    assert "## Intro\n" not in units[0]["implementation_text"]
    assert "## Gameplay Systems\n" not in units[0]["implementation_text"]
    assert "".join(unit["text"] for unit in units) == text


def test_generic_document_wrapper_keeps_single_semantic_feature_with_subheadings():
    text = (
        "# Design\n"
        "## Trading\n"
        "### Trigger\nExchange ore for credits.\n"
        "### State\nPersist credits across relog.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 1
    assert units[0]["section"] == "Trading"
    assert units[0]["implementation_text"].startswith("## Trading\n")
    assert "### Trigger\n" in units[0]["implementation_text"]
    assert "### State\n" in units[0]["implementation_text"]
    assert units[0]["text"] == text


def test_document_metadata_preamble_is_context_not_a_standalone_feature():
    text = (
        "# Stellar Odyssey Mod Design Document\n"
        "**Mod Name:** Stellar Odyssey (별유람선)\n"
        "**Version:** 1.0\n"
        "**Target Minecraft Version:** 1.21+\n"
        "**Genre:** Sci-Fi, Survival, Base Building, Space Combat\n"
        "---\n"
        "# Resource Economy\n"
        "Persist player credits and exchange mined resources for credits.\n"
        "# Ship Construction\n"
        "Spend credits to install ship parts and upgrades.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 2
    assert [unit["section"] for unit in units] == [
        "Resource Economy",
        "Ship Construction",
    ]
    assert units[0]["text"].startswith("# Stellar Odyssey Mod Design Document\n")
    assert "# Resource Economy\n" in units[0]["text"]
    assert "Persist player credits" in units[0]["text"]
    assert "".join(unit["text"] for unit in units) == text


@pytest.mark.parametrize(
    "prefix",
    [
        "\n",
        "---\n",
        "Saved authored design\n\n",
        "\ufeff",
    ],
)
def test_document_preamble_with_leading_source_bytes_is_lossless(prefix):
    text = (
        prefix
        + "# Stellar Odyssey Mod Design Document\n"
        "**Mod Name:** Stellar Odyssey\n"
        "**Version:** 1.0\n"
        "---\n"
        "# Resource Economy\n"
        "Persist credits and mined resources.\n"
        "# Ship Construction\n"
        "Install ship parts and upgrades.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 2
    assert [unit["section"] for unit in units] == [
        "Resource Economy",
        "Ship Construction",
    ]
    assert units[0]["text"].startswith(prefix)
    assert "".join(unit["text"] for unit in units) == text


def test_generic_wrapper_with_leading_prose_is_lossless():
    text = (
        "Saved design context that must not be dropped.\n\n"
        "# Design\n"
        "**Mod Name:** Example\n"
        "## Wallet\nPersist each player's balance.\n"
        "## Purchase\nDeduct once and grant once.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 2
    assert [unit["section"] for unit in units] == ["Wallet", "Purchase"]
    assert units[0]["text"].startswith("Saved design context")
    assert "".join(unit["text"] for unit in units) == text


def test_single_generic_wrapper_metadata_attaches_to_first_child_feature():
    text = (
        "# Design\n"
        "**Mod Name:** Example\n"
        "**Version:** 1.0\n"
        "## Wallet\nPersist each player's balance.\n"
        "## Purchase\nDeduct once and grant once.\n"
    )

    units = _authored_execution_units(text)

    assert len(units) == 2
    assert [unit["section"] for unit in units] == ["Wallet", "Purchase"]
    assert units[0]["text"].startswith("# Design\n**Mod Name:** Example\n")
    assert "## Wallet\n" in units[0]["text"]
    assert "".join(unit["text"] for unit in units) == text


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
    assert [module.depends_on for module in modules] == [
        (),
        ("authored_feature_001",),
        ("authored_feature_002",),
    ]
    assert "".join(
        module.config["evidence_task"]["engineering_worksheet"]["authored_unit"]["text"]
        for module in modules
    ) == text


def test_nested_authored_behavior_reaches_one_manifest_target_and_atomic_coder():
    text = (
        "# Trading\n"
        "## Trigger\nRight click a trader to exchange one ore for 10 credits.\n"
        "## State\nKeep each player credit balance across relog.\n"
        "# Travel\n"
        "## Trigger\nSpend 20 credits to launch the player's ship.\n"
        "## Rejection\nInsufficient credits leaves both balance and ship unchanged.\n"
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(
        AuthoredPlan("Space trading for Fabric 1.21.11", text)
    )
    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["unit_count"] == len(proposal.modules) == 2
    assert [unit["section"] for unit in manifest["units"]] == ["Trading", "Travel"]
    source = text.encode("utf-8")
    next_start = 0
    for module, record in zip(proposal.modules, manifest["units"], strict=True):
        capsule = compile_task_capsule(module)
        batches = atomicize_coder_messages([{"role": "user", "content": json.dumps({
            "phase": "implement_module", "module": _task_local_module_contract(module),
        })}])
        atomic = json.loads(batches[0][-1]["content"])["module"]["evidence_task"]["coder_execution_contract"]
        unit = atomic["engineering_worksheet"]["authored_unit"]
        assert unit["start_byte"] == record["start_byte"] == next_start
        next_start = unit["end_byte"]
        exact = source[unit["start_byte"]:next_start]
        assert unit["text"].encode("utf-8") == exact
        assert unit["source_text_sha256"] == record["text_sha256"] == "sha256:" + hashlib.sha256(exact).hexdigest()
        assert unit["text"].strip() in atomic["step"]["obligation"]
        assert capsule.primary_path == record["path"]
        assert atomic["step"]["target_refs"] == [record["path"] + "#" + record["symbol"]]
        if unit["section"] == "Trading":
            assert "10 credits" in unit["text"] and "across relog" in unit["text"]
        else:
            assert "20 credits" in unit["text"] and "Insufficient credits" in unit["text"]
    assert next_start == len(source)


@pytest.mark.parametrize("text", [
    "# Trading\n## Trigger\nExchange ore for credits.\n## State\nPersist credits.\n",
    "# Overview\n# Trading\nExchange ore for credits.\n# Appendix\n",
    "# Trading\nExchange ore for credits.\n```markdown\n# Not a feature\n```\n",
])
def test_authored_headings_do_not_create_empty_or_code_fence_tasks(text):
    units = _authored_execution_units(text)
    assert len(units) == 1
    assert units[0]["text"] == text


def test_oversized_semantic_authored_section_is_not_split_by_bytes():
    text = "# Trading\n## Trigger\n" + "광석 하나를 열 크레딧으로 교환한다.\n" * 160
    units = _authored_execution_units(text)

    assert len(units) == 1
    assert units[0]["section"] == "Trading"
    assert units[0]["text"] == text
    assert units[0]["end_byte"] == len(text.encode("utf-8"))

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
    assert custom[0].dependencies == ("prepare-project",)
    for previous, node in zip(custom[:-1], custom[1:], strict=True):
        assert set(node.dependencies) == {"prepare-project", previous.node_id}


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
    assert module.required_gates == ("target_compile",)
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

def test_model_reasoning_prefix_is_not_lowered_as_gameplay_work() -> None:
    source = (
        "Thinking Process:\n\n"
        "1. **Analyze the Request:** space mod\n"
        "2. **Deconstruct the Template Sections for Content:** details\n"
        "3. **Drafting Content (Mental Outline):** draft\n\n"
        "# behavior_contract\n"
        "- player trades resources for ship parts\n"
        "# verification\n"
        "- launch without fuel is rejected\n"
    )
    plan = AuthoredPlan(requested_prompt="space mod", text=source)

    projected, provenance = _implementation_authored_plan(plan)

    assert projected.text.startswith("# behavior_contract\n")
    assert "Thinking Process" not in projected.text
    assert provenance is not None
    assert provenance["stripped_prefix_bytes"] > 0
    assert provenance["source_text_sha256"] != provenance["implementation_text_sha256"]
    units = _authored_execution_units(projected.text)
    assert all("Thinking Process" not in unit["text"] for unit in units)


def test_user_intro_is_preserved_without_model_reasoning_markers() -> None:
    source = (
        "Starship economy design.\n\n"
        "# behavior_contract\n"
        "- trade ore for credits\n"
    )
    plan = AuthoredPlan(requested_prompt="space mod", text=source)

    projected, provenance = _implementation_authored_plan(plan)

    assert projected is plan
    assert provenance is None


def test_authored_units_preserve_projected_text_exactly() -> None:
    source = "# behavior_contract\n- A\n# state_model\n- B\n"
    units = _authored_execution_units(source)

    assert "".join(unit["text"] for unit in units) == source


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("prefix,design", [
    (("Thinking Process:\n1. **Analyze the Request:** space mod\n"
      "instruction injected into the prompt. As an AI text generator...\n\n"),
     "## behavior_contract\nTrade ore for 10 credits.\n## state_model\nPersist credits.\n"),
    ("**Thinking Process:**\n1. **Analyze the Request:** space mod\n\n",
     "### 행동 계약\n광석을 10 크레딧으로 교환한다.\n### 상태 모델\n잔액을 저장한다.\n"),
    ("Thinking Process:\n1. **Analyze the Request:** space mod\n\n",
     "# 우주 모드 설계\n## behavior_contract\nTrade ore for 10 credits.\n"),
    (("<think>Analyze the Request: space mod\n"
      "## behavior_contract\nThis heading is still inside reasoning.\n</think>\n\n"),
     "# Economy\nTrade ore for 10 credits.\n"),
    ("<analysis>Draft the Korean design.</analysis>\n",
     "우주선의 연료가 없으면 출발을 거부한다.\n"),
])
def test_compiler_projects_reasoning_before_both_production_routes(prefix, design, existing):
    plan = AuthoredPlan(
        "Space mod for Fabric 1.21.11", prefix + design,
        existing_input_sha256="sha256:" + "a" * 64 if existing else "",
        media_paths=("reference.png",),
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    saved = proposal.game_design["authored_plan"]
    assert saved["text"] == design
    assert saved["media_paths"] == ["reference.png"]
    projection = proposal.game_design["_authored_source_projection"]
    assert projection["source_plan"] == plan.to_dict()
    assert projection["stripped_prefix_bytes"] == len(prefix.encode("utf-8"))
    def sha(text):
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert projection["source_text_sha256"] == sha(prefix + design)
    assert projection["implementation_text_sha256"] == sha(design)
    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["source_text_sha256"] == sha(design)
    parts = []
    for module, record in zip(proposal.modules, manifest["units"], strict=True):
        if existing:
            part = module.config["authored_plan"]["text"]
        else:
            batches = atomicize_coder_messages([{"role": "user", "content": json.dumps({
                "phase": "implement_module", "module": _task_local_module_contract(module),
            })}])
            task = json.loads(batches[0][-1]["content"])["module"]["evidence_task"]
            part = task["coder_execution_contract"]["engineering_worksheet"]["authored_unit"]["text"]
            assert "Thinking Process" not in json.dumps(task)
            assert "instruction injected" not in json.dumps(task)
        assert part.encode("utf-8") == design.encode("utf-8")[record["start_byte"]:record["end_byte"]]
        assert record["text_sha256"] == sha(part)
        parts.append(part)
    assert "".join(parts) == design


@pytest.mark.parametrize("text", [
    ("NPC dialogue says Thinking Process: and Analyze the Request:\n"
     "# behavior_contract\nDisplay that dialogue when the player trades.\n"),
    ("Thinking Process:\nAnalyze the Request: this is the NPC's dialogue.\n"
     "```markdown\n# behavior_contract\nAn in-game document example.\n```\n"),
    "Thinking Process:\nAnalyze the Request: a draft with no final boundary.\n",
    ("# Manual\n<think>is a literal tag in the manual</think>\n"
     "# behavior_contract\nRender the manual unchanged.\n"),
])
def test_compiler_preserves_ambiguous_or_quoted_reasoning_text(text):
    plan = AuthoredPlan("Space mod for Fabric 1.21.11", text)
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    assert proposal.game_design["authored_plan"] == plan.to_dict()
    assert "_authored_source_projection" not in proposal.game_design


def test_contract_shaped_authored_design_stays_one_coherent_bounded_module(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved design entered planner again")

    monkeypatch.setattr(PlanningPipeline, "prepare", forbidden)
    monkeypatch.setattr(PlanningPipeline, "_semantic_design", forbidden)
    router = SimpleNamespace(generate_text=forbidden, generate_tool_decision=forbidden)
    sections = (
        "state_model",
        "algorithm",
        "integration",
        "authority_and_network",
        "persistence",
        "resources_and_ui",
        "failure_and_limits",
        "reuse_assessment",
        "verification",
    )
    text = "\n".join(
        f"# {section}\n- concrete_{section}: "
        + ("observable behavior, owned state, and exact constraints. " * 12)
        for section in sections
    )
    assert len(text.encode("utf-8")) > 2048
    plan = AuthoredPlan("Make a space trading mod for Fabric 1.21.11", text)

    proposal = CompleteGameDesignPlanner(router).compile_for_production(plan)

    assert len(proposal.modules) == 1
    module = proposal.modules[0]
    assert module.module_id == "authored_design"
    assert module.config["authored_execution_mode"] == "bounded_coherent"
    assert module.config["authored_bounded_scope"] is True
    assert module.config["authored_plan"]["text"] == text
    assert compile_task_capsule(module) is None

    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["policy"] == "host_bounded_coherent_authored_design"
    assert manifest["unit_count"] == 1
    assert manifest["units"][0]["module_id"] == "authored_design"

    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS

    package_path = proposal.base_proposal.spec.package_name.replace(".", "/")
    mod_id = proposal.base_proposal.spec.mod_id
    assert authority.mutation_authority.authorizes(
        f"src/main/java/{package_path}/economy/TradeService.java",
        operation="create_file",
    )
    assert authority.mutation_authority.authorizes(
        f"src/main/resources/assets/{mod_id}/lang/ko_kr.json",
        operation="create_file",
    )
    assert authority.mutation_authority.authorizes(
        f"src/main/resources/data/{mod_id}/recipes/ship_part.json",
        operation="create_file",
    )
    assert not authority.mutation_authority.authorizes(
        "src/main/resources/fabric.mod.json",
        operation="replace_exact",
    )
    assert not authority.mutation_authority.authorizes(
        "src/main/java/com/example/Foreign.java",
        operation="create_file",
    )
    assert not authority.mutation_authority.authorizes(
        f"src/main/java/{package_path}/Old.java",
        operation="delete_file",
    )

    contract = _task_local_module_contract(module)
    assert contract["authored_execution_mode"] == "bounded_coherent"
    assert contract["authored_write_scope"]["mod_id"] == mod_id
    assert contract["authored_write_scope"]["java_package"] == proposal.base_proposal.spec.package_name

    messages = [{
        "role": "user",
        "content": json.dumps({
            "phase": "implement_authored_design",
            "module": contract,
        }, ensure_ascii=False),
    }]
    batches = atomicize_coder_messages(messages)
    assert len(batches) == 1
    assert json.loads(batches[0][0]["content"])["module"]["authored_plan"]["text"] == text

