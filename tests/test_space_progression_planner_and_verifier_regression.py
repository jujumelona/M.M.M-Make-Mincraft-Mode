from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import evidence_first_planning as planning
from minecraft_mod_ai import planning_authority
from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ToolCall,
)
from minecraft_mod_ai.production_contract import _infer_dimensions
from minecraft_mod_ai.progress_aware_tool_loop import generate_with_tools
from minecraft_mod_ai.small_model_task_capsule_contract import (
    _TaskBoundAdapter,
    compile_task_capsule,
)

PROMPT = (
    "자원파밍으로 돈을 모으고 거래하여 우주선을 부위마다 제작하고 무기를 업그레이드하고 "
    "선원을 고용하고 우주선 성능을 업그레이드하고 우주선을 확장한 뒤 우주로 나가서 "
    "다른 행성 광물을 채굴하고 외계인과 전투하고 다른 행성을 식민지화한다"
)


def _semantic_item(
    capability: str,
    anchor: str,
    given: str,
    when: str,
    then: str,
    *,
    required=(),
    semantic_type="gameplay_mechanic",
):
    return {
        "source_clause_index": 0,
        "capability_id": capability,
        "source_anchor": anchor,
        "semantic_statement": f"Player-visible {capability} behavior",
        "given": given,
        "when": when,
        "then": then,
        "semantic_type": semantic_type,
        "required_prerequisite_capabilities": list(required),
        "optional_prerequisite_capabilities": [],
    }


class _SemanticRouter:
    def generate_tool_decision(self, role, messages, **kwargs):
        assert role == "planner"
        return {
            "requirements": [
                _semantic_item("resource.farming", "자원파밍", "farmable resources exist", "the player farms resources", "resource inventory increases"),
                _semantic_item("economy.currency", "돈을 모으고", "the player owns gathered resources", "the player earns money", "currency balance increases", required=("resource.farming",)),
                _semantic_item("economy.trade", "거래하여", "resources and currency are available", "the player accepts a priced stocked trade", "inventory, stock and balance change atomically", required=("resource.farming", "economy.currency")),
                _semantic_item("spacecraft.component_construction", "우주선을 부위마다 제작하고", "resources, currency and trading are available", "the player acquires and assembles compatible ship parts", "the assembled spacecraft records its parts", required=("resource.farming", "economy.currency", "economy.trade")),
                _semantic_item("spacecraft.weapon_upgrade", "무기를 업그레이드하고", "a spacecraft and trading are available", "the player buys and installs a weapon tier", "the weapon slot and combat stats increase", required=("spacecraft.component_construction", "economy.trade")),
                _semantic_item("crew.recruitment", "선원을 고용하고", "a spacecraft and trading are available", "the player hires and assigns crew", "crew roles and skills affect the spacecraft", required=("spacecraft.component_construction", "economy.trade")),
                _semantic_item("spacecraft.performance_upgrade", "우주선 성능을 업그레이드하고", "a spacecraft and trading are available", "the player buys a performance tier", "thrust, speed, fuel capacity or durability increases", required=("spacecraft.component_construction", "economy.trade"), semantic_type="software_quality"),
                _semantic_item("spacecraft.expansion", "우주선을 확장한 뒤", "a spacecraft and trading are available", "the player buys and installs expansion modules", "cargo or module capacity increases", required=("spacecraft.component_construction", "economy.trade")),
                _semantic_item("space.launch", "우주로 나가서", "the spacecraft, weapons, crew, performance and expansion meet launch requirements", "the player spends fuel and selects a destination", "the player and spacecraft enter space", required=("spacecraft.component_construction", "spacecraft.weapon_upgrade", "crew.recruitment", "spacecraft.performance_upgrade", "spacecraft.expansion")),
                _semantic_item("planet.special_mineral", "다른 행성 광물을 채굴하고", "the player is in space and has reached another planet", "the player mines a special mineral", "the special mineral enters inventory", required=("space.launch",)),
                _semantic_item("alien.combat", "외계인과 전투하고", "the player is in space on an alien planet", "the player and an alien exchange attacks", "combat damage, death and drops are observable", required=("space.launch",)),
                _semantic_item("colony.colonization", "다른 행성을 식민지화한다", "the player is in space on a colonizable planet", "the player establishes a colony", "colony ownership, storage and development persist", required=("space.launch",)),
            ]
        }


def test_space_progression_semantics_dependencies_and_obligations_are_complete() -> None:
    catalog = planning_authority.build_authoritative_request_catalog(PROMPT, _SemanticRouter())
    by_capability = {item["capability"]: item for item in catalog["requirements"]}

    assert len(by_capability) == 12
    performance = by_capability["spacecraft.performance_upgrade"]
    assert performance["semantic_type"] == "gameplay_mechanic"
    assert "spacecraft.gameplay_stat_schema" in performance["implementation_capabilities"]
    assert performance["artifact_task_ids"]
    assert performance["artifact_obligations"]
    assert performance["runtime_acceptance"]
    assert any("thrust/speed" in item for item in performance["design_resolution_obligations"])
    assert any(
        "buy/sell prices" in item
        for item in by_capability["economy.trade"]["design_resolution_obligations"]
    )
    assert any(
        "death/dismissal" in item
        for item in by_capability["crew.recruitment"]["design_resolution_obligations"]
    )
    assert any(
        "development stages" in item
        for item in by_capability["colony.colonization"]["design_resolution_obligations"]
    )

    launch = by_capability["space.launch"]
    launch_dependencies = {
        next(item["capability"] for item in catalog["requirements"] if item["requirement_id"] == ref)
        for ref in launch["depends_on"]
    }
    assert launch_dependencies == {
        "spacecraft.component_construction",
        "spacecraft.weapon_upgrade",
        "crew.recruitment",
        "spacecraft.performance_upgrade",
        "spacecraft.expansion",
    }
    assert launch["unlock_policy"]["optional_requirement_refs"] == []
    for capability in ("planet.special_mineral", "alien.combat", "colony.colonization"):
        assert by_capability[capability]["depends_on"] == [launch["requirement_id"]]
    construction_dependencies = set(by_capability["spacecraft.component_construction"]["depends_on"])
    assert by_capability["economy.trade"]["requirement_id"] in construction_dependencies

    branches = planning._branch_predicates(catalog["requirements"], (), {"project_topology": {}})
    assert branches["needs_mixin"]["status"] == "NOT_APPLICABLE"
    steps = planning._semantic_steps("spacecraft.performance_upgrade", branches)
    assert [step.name for step in steps] == [
        "gameplay_schema",
        "domain_service",
        "state_sync",
        "resource_binding",
        "runtime_scenario",
    ]
    assert not any("build_config" in step.anchor_kinds for step in steps)
    assert "behavior_equivalence" not in planning._required_gates(
        "spacecraft.performance_upgrade", branches
    )
    ownership = {
        "module_id": ":",
        "source_set": "main",
        "source_root": "src/main/java",
        "resource_root": "src/main/resources",
        "test_root": "src/test/java",
        "namespace": "generated.mod",
        "mod_id": "space_mod",
        "extension": "java",
        "topology_module_ids": [],
        "topology_source_sets": [],
    }
    anchor_probe = planning._anchors(
        "spacecraft.performance_upgrade", steps[-1], "task_probe", ownership
    )
    assert {anchor["source_set"] for anchor in anchor_probe if anchor["kind"] == "test"} == {"test"}

    gap = {
        "gap_id": "gap_ship_performance",
        "requirement_ref": performance["requirement_id"],
        "capability": performance["capability"],
        "missing_provides": performance["provides"],
        "acceptance": performance["acceptance"],
        "runtime_acceptance": performance["runtime_acceptance"],
        "implementation_capabilities": performance["implementation_capabilities"],
        "artifact_obligations": performance["artifact_obligations"],
        "semantic_type": performance["semantic_type"],
        "unlock_policy": performance["unlock_policy"],
    }
    tasks = planning._compile_tasks(
        (gap,),
        ({"requirement_ref": performance["requirement_id"], "component_refs": [], "source_refs": []},),
        {"coordinates": {"minecraft_version": "1.21.1", "loader": "fabric"}},
        branches,
        ownership,
    )
    final_task = tasks[-1]
    assert "runtime_gameplay_validation" in final_task["required_gates"]
    assert "public_acceptance_observed" in final_task["done_predicate"]["checks"]
    assert final_task["runtime_acceptance"]
    assert {item["kind"] for item in final_task["artifact_obligations"]} >= {
        "item_model",
        "recipe",
        "tag",
        "lang",
    }

    dimensions, _reasons = _infer_dimensions(
        requested_prompt="우주선 성능을 거래 구매로 업그레이드한다",
        game_design={"capability": "spacecraft.performance_upgrade"},
        research_brief=None,
        modules=(),
        assets=(),
    )
    assert "performance" not in dimensions
    assert "state_save_migration" not in dimensions


def _module():
    java_path = "src/main/java/generated/mod/ShipPerformance.java"
    test_path = "src/test/java/generated/mod/ShipPerformanceTest.java"
    main_anchor = {"kind": "symbol", "locator": f"{java_path}#ShipPerformance", "ownership": "exclusive", "status": "host_reserved", "module_id": "root", "source_set": "main"}
    test_anchor = {"kind": "test", "locator": f"{test_path}#ShipPerformanceTest", "ownership": "exclusive", "status": "host_reserved", "module_id": "root", "source_set": "test"}
    task = {
        "task_id": "task_ship_performance",
        "task_sha256": "sha256:" + "a" * 64,
        "requirement_refs": ["req_ship_performance"],
        "gap_refs": ["gap_ship_performance"],
        "owned_anchors": [main_anchor, test_anchor],
        "provides": ["capability:spacecraft.performance_upgrade"],
        "acceptance": ["ship performance stats increase after purchase"],
        "production_bindings": [{"task_ref": "task_ship_performance", "reuse_action": "fresh", "owned_anchors": [main_anchor]}],
    }
    return SimpleNamespace(module_id="task_ship_performance", kind="custom_java", config={"evidence_task": task}, depends_on=(), required_gates=("source_static_validation", "target_compile", "runtime_gameplay_validation"))


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
        },
    }


def test_logged_malformed_diagnostics_path_is_rebound_and_verification_completes(capsys) -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None

    class Adapter:
        def __init__(self):
            self.turn = 0

        def generate_turn(self, request):
            self.turn += 1
            if self.turn == 1:
                args = {"operation": "create_file", "path": "src/main/java/wrong/.java", "content": "package generated.mod; public final class ShipPerformance {}"}
                return GenerationResponse(tool_calls=(ToolCall(id="edit", name="apply_source_edit", arguments=args, raw_arguments=json.dumps(args)),))
            if self.turn == 2:
                args = {"action": "java_diagnostics", "arguments": {"diagnostics_config": {"allowlist": ["/wrong/.java"]}}}
                return GenerationResponse(tool_calls=(ToolCall(id="verify", name="java_diagnostics", arguments=args, raw_arguments=json.dumps(args)),))
            return GenerationResponse(content="implemented and verified")

    class Runtime:
        def __init__(self):
            self.calls = []

        def call(self, stage, name, arguments):
            self.calls.append((stage, name, dict(arguments)))
            if name == "apply_source_edit":
                return {"schema_version": "mmm/source-patch-receipt-v1", "status": "APPLIED", "operations": [{"operation": "create", "path": capsule.primary_path, "before_sha256": "sha256:" + "a" * 64, "after_sha256": "sha256:" + "b" * 64}]}
            assert name == "java_diagnostics"
            assert arguments == {"project_root": ".", "relative_files": [capsule.primary_path], "timeout_seconds": 60}
            return {"status": "PASS", "diagnostics": {}}

    router = SimpleNamespace(_agent_require_fresh_evidence=False, _generation_scope=lambda *args, **kwargs: nullcontext())
    config = SimpleNamespace(adapter="llama_cpp", max_context=16384, max_input_tokens=0, max_new_tokens=4096, extra={"runtime_contract": "qwen", "qwen_family": "qwen3.5"})
    request = GenerationRequest(
        messages=({"role": "user", "content": json.dumps({"phase": "implement_module", "operation": "create_file", "path": capsule.primary_path})},),
        tools=(_schema("apply_source_edit"), _schema("java_diagnostics")),
        tool_choice=None,
        parallel_tool_calls=False,
    )
    runtime = Runtime()

    result = generate_with_tools(router, config=config, adapter=_TaskBoundAdapter(Adapter(), capsule), request=request, runtime=runtime, stage="generation", role="coder")

    assert result == "implemented and verified"
    assert [name for _stage, name, _args in runtime.calls] == ["apply_source_edit", "java_diagnostics"]
    stderr = capsys.readouterr().err
    assert '"event":"task_tool_arguments_bound"' in stderr
    assert '"event":"verification_adjudicated"' in stderr
    assert '"reason":"PASS"' in stderr
    assert "VERIFIER_UNAVAILABLE" not in stderr


def test_agent_runtime_binds_diagnostics_to_the_actual_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src/main/java/generated/mod").mkdir(parents=True)
    (project / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    target = "src/main/java/generated/mod/ShipPerformance.java"
    (project / target).write_text("package generated.mod; final class ShipPerformance {}\n", encoding="utf-8")
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    dispatched = []

    def fake_run_async(function, *args):
        dispatched.append((function.__name__, args))
        return {"status": "PASS", "diagnostics": {}}

    runtime._run_async = fake_run_async  # type: ignore[method-assign]
    runtime.call(
        "generation",
        "java_diagnostics",
        {
            "project_root": "/hallucinated/project",
            "relative_files": [target],
            "diagnostics_config": {"allowlist": ["/wrong/.java"]},
        },
    )

    assert len(dispatched) == 1
    _function, (_stage, _name, arguments) = dispatched[0]
    assert arguments == {
        "project_root": str(project.resolve()),
        "relative_files": [target],
    }
