import hashlib
import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import (
    _authored_execution_units,
    _compile_new_authored_modules,
    _execution_plan_projection,
    _implementation_authored_plan,
    materialize_authored_execution_scaffold,
)
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.custom_module_generator import _task_local_module_contract
from minecraft_mod_ai.planning_pipeline import PlanningPipeline
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.work_graph import build_production_work_plan


@pytest.mark.parametrize("version", ["1.21.11", "26.2"])
def test_existing_design_localization_preserves_target_contract(version):
    from minecraft_mod_ai.custom_generation_research import _target_values
    from minecraft_mod_ai.platform_catalog import adapter_for_target

    plan = AuthoredPlan(
        f"Modify a token item for Fabric {version}",
        "Preserve the token item and update its behavior.",
        existing_input_sha256="sha256:" + "a" * 64,
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    adapter = adapter_for_target(version, "fabric")
    expected = (version, "fabric", adapter.yarn_mappings)

    assert _target_values(proposal.game_design) == expected
    assert proposal.game_design["authored_plan"] == plan.to_dict()
    assert proposal.game_design["_platform_selection"]["target"] == adapter.public_dict()
    assert all(
        module.config.get("authored_localization_required") is True
        for module in proposal.modules
    )

@pytest.mark.parametrize("text", [
    "우주선을 만들고 행성마다 다른 광물을 거래한다.",
    '```json\n{"capability_label":"trade"}\n```',
    "0",
    "# 설계\n" + "선원, 무기, 연료, 수리, 거래를 구현한다.\n" * 1000,
], ids=["plain", "fenced", "zero", "long"])

def test_existing_saved_text_uses_localization_without_replanning(monkeypatch, text):
    def forbidden(*args, **kwargs):
        pytest.fail("existing saved design entered gameplay planning again")

    monkeypatch.setattr(PlanningPipeline, "prepare", forbidden)
    monkeypatch.setattr(PlanningPipeline, "_semantic_design", forbidden)
    router = SimpleNamespace(generate_text=forbidden, generate_tool_decision=forbidden)
    plan = AuthoredPlan(
        "Modify the existing space trading mod for Fabric 1.21.11",
        text,
        existing_input_sha256="sha256:" + "b" * 64,
    )
    proposal = CompleteGameDesignPlanner(router).compile_for_production(plan)

    assert proposal.game_design["authored_plan"] == plan.to_dict()
    assert "".join(
        module.config["authored_plan"]["text"] for module in proposal.modules
    ).encode() == text.encode()
    graph = build_production_work_plan(proposal)
    generation = [n for n in graph.nodes if n.stage == "generate:custom"]
    assert generation
    assert all(
        member["config"].get("authored_localization_required") is True
        for node in generation
        for member in node.payload["members"]
    )

def test_fresh_markdown_graph_request_hashes_empty_structured_authority():
    plan = AuthoredPlan(
        "space mod",
        "# behavior_contract\n- actors: player controls the ship.\n"
        "## state_model\n- variables: credits are server-owned.\n",
    )
    modules, _manifest = _compile_new_authored_modules(
        plan,
        mod_id="authored_test",
        package_name="example",
        target={},
        production_state_section={},
    )

    request = modules[0].config["implementation_graph_request"]
    assert request["structured_sections"] == {}
    expected_sha = "sha256:" + hashlib.sha256(b"{}").hexdigest()
    assert request["structured_sections_sha256"] == expected_sha


def test_fresh_authored_execution_defers_source_ownership_until_ir():
    plan = AuthoredPlan("space mod", "# Design\n## Wallet\nPersist credits.\n## Purchase\nSpend credits.")
    modules, manifest = _compile_new_authored_modules(plan, mod_id="authored_test", package_name="example", target={})
    assert manifest["units"] == []
    assert len(modules) == 1
    task = modules[0].config["evidence_task"]
    assert task["engineering_worksheet"]["implementation_graph_request"]["text"] == plan.text
    assert task["depends_on"] == []
    assert task["owned_anchors"][0]["locator"] == manifest["entrypoint"]["path"] + "#" + manifest["entrypoint"]["symbol"]

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


def test_fresh_authored_document_sections_are_provenance_not_source_units():
    text = "# Economy\nCredits and trade.\n# Ships\nParts and upgrades.\n# Verification\nCheck failures."
    modules, manifest = _compile_new_authored_modules(AuthoredPlan("space mod", text), mod_id="authored_test", package_name="example", target={})
    assert manifest["unit_count"] == 0
    assert modules[0].config["implementation_graph_request"]["text"] == text
    assert "AuthoredFeature" not in json.dumps(manifest)


def test_nested_existing_authored_behavior_reaches_localization_without_loss():
    text = (
        "# Trading\n## Trigger\nTrade for 10 credits.\n## State\nPersist credits.\n"
        "# Travel\nSpend 20 credits.\n## Failure\nKeep balance unchanged."
    )
    plan = AuthoredPlan(
        "Modify Fabric 1.21.11 mod",
        text,
        existing_input_sha256="sha256:" + "c" * 64,
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)

    assert "".join(
        module.config["authored_plan"]["text"] for module in proposal.modules
    ) == text
    assert proposal.game_design["_authored_execution_manifest"]["source_text_sha256"] == (
        "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    )


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

def test_fresh_authored_work_graph_schedules_ir_execution_after_project_preparation(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(
        AuthoredPlan("space mod", "# Economy\nCredits.\n# Ships\nParts.\n# Planets\nMining.")
    )
    graph = build_production_work_plan(
        proposal,
        policy=ScalePolicy(java_shard_size=48),
    )
    custom = [node for node in graph.nodes if node.stage == "generate:custom"]
    assert len(custom) == 1
    assert custom[0].resource_class == "llm"
    assert custom[0].dependencies == ("prepare-project",)
    assert custom[0].payload["members"][0]["module_id"] == "authored_implementation_graph"

def test_authored_scaffold_defers_source_materialization_until_ir(tmp_path):
    modules, manifest = _compile_new_authored_modules(AuthoredPlan("space mod", "Economy, ships, planets"),
        mod_id="authored_test", package_name="example", target={})
    proposal = SimpleNamespace(game_design={"_authored_execution_manifest": manifest})
    main = tmp_path / manifest["entrypoint"]["path"]
    main.parent.mkdir(parents=True)
    original = b"package example; public final class AuthoredTestMod { public void onInitialize() {} }"
    main.write_bytes(original)
    for _ in range(2):
        assert materialize_authored_execution_scaffold(proposal, tmp_path) == tmp_path.resolve()
        assert main.read_bytes() == original
        assert list(main.parent.glob("*.java")) == [main]
    assert modules[0].config["implementation_graph_request"]["entrypoint_path"] == manifest["entrypoint"]["path"]


def test_real_orchestrator_accepts_existing_authored_localization_handoff(monkeypatch, tmp_path):
    from minecraft_mod_ai.complete_orchestrator import (
        CompleteExecutionOptions,
        CompleteProductionOrchestrator,
    )

    plan = AuthoredPlan(
        "Modify the existing Fabric 1.21.11 space mod",
        "행성과 광물 거래",
        existing_input_sha256="sha256:" + "d" * 64,
    )
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path)

    class ReachedProjectCreation(Exception):
        pass

    def prepare(approved, **kwargs):
        assert approved.game_design["authored_plan"] == plan.to_dict()
        assert "_authored_execution_manifest" in approved.game_design
        assert all(
            module.config.get("authored_localization_required") is True
            for module in approved.modules
        )
        raise ReachedProjectCreation

    monkeypatch.setattr(orchestrator, "_prepare_project", prepare)
    with pytest.raises(ReachedProjectCreation):
        orchestrator.execute(
            proposal,
            approval_hash=proposal.calculate_hash(),
            run_name="authored-existing",
            options=CompleteExecutionOptions(
                run_blockbench=False,
                run_runtime=False,
                run_client=False,
                run_mineflayer=False,
                run_visual_review=False,
            ),
        )


def _structured_state_plan(existing_sha: str = "") -> AuthoredPlan:
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS

    state_spec = {
        **{name: [] for name in DETAIL_RECORDS["state_model"]},
        "variables": [{
            "name": "credits",
            "owner": "Player",
            "type": "Int",
            "unit": "credits",
            "default": "0",
            "domain": "economy",
        }],
        "inapplicable_concerns": [
            {"concern": name, "reason": "not required"}
            for name in DETAIL_RECORDS["state_model"]
            if name != "variables"
        ],
    }
    structured = {
        "state_model": {
            "specification": state_spec,
            "constraint_evidence_refs": [],
        }
    }
    return AuthoredPlan(
        "Modify existing economy",
        "## state_model\n- variables: name owner type unit default domain\n",
        existing_input_sha256=existing_sha,
        structured_sections=structured,
    )


def test_execution_document_normalization_preserves_structured_semantic_authority() -> None:
    base = _structured_state_plan()
    plan = AuthoredPlan(
        base.requested_prompt,
        "# Wrapper\n# state_model\n- variables: name owner type unit default domain\n",
        structured_sections=base.structured_sections,
    )

    projected, _report = _execution_plan_projection(plan)

    assert projected.structured_sections == base.structured_sections
    assert projected.schema_version == plan.schema_version


def test_existing_authored_units_preserve_structured_semantic_authority() -> None:
    plan = _structured_state_plan("sha256:" + "c" * 64)
    proposal = CompleteGameDesignPlanner(SimpleNamespace()).compile_for_production(plan)

    assert len(proposal.modules) == 1
    saved = proposal.modules[0].config["authored_plan"]
    assert saved["schema_version"] == "mmm/authored-plan-v2"
    assert saved["structured_sections"] == plan.structured_sections


def test_reasoning_projection_preserves_structured_semantic_authority() -> None:
    base = _structured_state_plan()
    plan = AuthoredPlan(
        base.requested_prompt,
        (
            "Thinking Process:\n"
            "1. **Analyze the Request:** economy\n\n"
            + base.text
        ),
        structured_sections=base.structured_sections,
    )

    projected, provenance = _implementation_authored_plan(plan)

    assert provenance is not None
    assert projected.structured_sections == base.structured_sections
    assert projected.schema_version == plan.schema_version


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
        "Space mod for Fabric 1.21.11",
        prefix + design,
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

    if not existing:
        request = proposal.modules[0].config["implementation_graph_request"]
        assert request["text"] == design
        assert "instruction injected" not in request["text"]
        assert manifest["units"] == []
        return

    parts = []
    for module, record in zip(proposal.modules, manifest["units"], strict=True):
        if existing:
            part = module.config["authored_plan"]["text"]
        else:
            task = _task_local_module_contract(module)
            part = task["engineering_worksheet"]["authored_unit"]["text"]
            rendered = json.dumps(task, ensure_ascii=False)
            assert "Thinking Process" not in rendered
            assert "instruction injected" not in rendered
            anchors = task["owned_anchors"]
            assert len(anchors) == 1
            assert anchors[0]["locator"] == f"{record['path']}#{record['symbol']}"
        assert part.encode("utf-8") == design.encode("utf-8")[
            record["start_byte"]:record["end_byte"]
        ]
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


def test_worksheet_with_supplementary_section_is_not_split_into_feature_classes():
    from minecraft_mod_ai.authored_production import _contract_shaped_authored_design

    text = (
        "# StarForge\n## 개요 (Overview)\nSpace trading.\n"
        "## behavior_contract\nBuild ships.\n## state_model\nStore credits.\n"
        "## integration\nUse server events.\n## verification\nTest trades.\n"
        "## 추가 설명\nPreserve this custom design note.\n"
    )
    assert _contract_shaped_authored_design(text)


def test_orphan_reasoning_close_before_canonical_design_is_projected() -> None:
    prefix = (
        "## behavior_contract\n"
        "... list ...\n"
        "## state_model\n"
        "... list ...\n"
        "etc.\n\n"
        "I need to make sure I use these exact headers.\n"
        "Okay, ready to write.\n"
        "</think>\n\n"
    )
    design = (
        "# 우주 제국 (Space Frontier) 모드 디자인 문서\n\n"
        "## behavior_contract\nBuild ships.\n"
        "## state_model\nPersist credits.\n"
        "## algorithm\nCalculate launch requirements.\n"
        "## integration\nBind server events.\n"
        "## authority_and_network\nServer validates actions.\n"
        "## persistence\nSave player state.\n"
        "## resources_and_ui\nRender trade UI.\n"
        "## failure_and_limits\nReject invalid launch.\n"
        "## reuse_assessment\nReuse host APIs.\n"
        "## verification\nCompile and test.\n"
    )
    plan = AuthoredPlan("space mod", prefix + design)

    projected, provenance = _implementation_authored_plan(plan)

    assert projected.text == design
    assert provenance is not None
    assert provenance["stripped_prefix_bytes"] == len(prefix.encode("utf-8"))


def test_inline_or_fenced_reasoning_close_is_preserved() -> None:
    inline = (
        "# Manual\n"
        "The literal token </think> is shown to players.\n"
        "## behavior_contract\nBuild ships.\n"
        "## state_model\nPersist credits.\n"
        "## algorithm\nCalculate launch requirements.\n"
        "## integration\nBind server events.\n"
    )
    fenced = (
        "```text\n"
        "</think>\n"
        "```\n"
        "## behavior_contract\nBuild ships.\n"
        "## state_model\nPersist credits.\n"
        "## algorithm\nCalculate launch requirements.\n"
        "## integration\nBind server events.\n"
    )

    for source in (inline, fenced):
        plan = AuthoredPlan("space mod", source)
        projected, provenance = _implementation_authored_plan(plan)
        assert projected is plan
        assert provenance is None
