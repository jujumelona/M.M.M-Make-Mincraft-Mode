from __future__ import annotations

from minecraft_mod_ai.evidence_first_planning import (
    _BRANCHES,
    _compile_tasks,
    build_request_catalog,
)
from minecraft_mod_ai.reuse_planner import decompose_capability_graph


def test_prompt_only_unknown_requirement_is_one_opaque_provisional() -> None:
    graph = decompose_capability_graph("Add seasonal rune banking.")
    provisional = [node for node in graph.nodes if node.startswith("provisional:")]
    assert len(provisional) == 1
    assert not any(part in provisional[0] for part in ("primary", "state", "logic"))
    source_map = dict(graph.sources)
    assert source_map[provisional[0]] == "prompt_resolution.provisional_opaque"


def test_authoritative_design_does_not_gain_opaque_prompt_noise() -> None:
    design = {"capabilities": [f"system.feature_{index}" for index in range(96)]}
    graph = decompose_capability_graph("Implement the declared systems.", design=design)
    assert len(graph.nodes) == 96
    assert not any(node.startswith("provisional:") for node in graph.nodes)


def test_requirement_acceptance_never_copies_whole_prompt() -> None:
    prompt = (
        "Build a space progression mod with mining, trading, modular ships, "
        "colonies, bosses, and research across several planets."
    )
    design = {
        "modules": [
            {"capability": "resource.mining"},
            {"capability": "economy.trade"},
            {"capability": "spaceship.modular_build"},
        ]
    }
    catalog = build_request_catalog(prompt, design)
    acceptance = [
        item
        for requirement in catalog["requirements"]
        for item in requirement["acceptance"]
    ]
    assert acceptance
    assert all(prompt not in item for item in acceptance)
    assert all("Verify the observable player-facing behavior for capability" in item for item in acceptance)


def test_task_acceptance_keeps_internal_checks_for_dag_validation() -> None:
    public_acceptance = "Verify the observable player-facing behavior for capability economy.trade."
    gaps = [{"gap_id":"gap_trade","requirement_ref":"req_trade","capability":"economy.trade","missing_provides":["capability:economy.trade"],"acceptance":[public_acceptance]}]
    reuse = [{"requirement_ref":"req_trade","component_refs":[],"source_refs":[]}]
    target = {"coordinates": {"minecraft_version":"1.21.1","loader":"neoforge"}}
    branches = {name: {"status":"NOT_APPLICABLE"} for name in _BRANCHES}
    ownership = {"module_id":":","source_set":"main","source_root":"src/main/java","resource_root":"src/main/resources","test_root":"src/test/java","namespace":"generated.test","mod_id":"test","extension":"java","topology_module_ids":[],"topology_source_sets":[]}
    tasks = _compile_tasks(gaps, reuse, target, branches, ownership)
    assert tasks
    assert all(task["acceptance"] for task in tasks)
    assert all(any("declared provides" in item.casefold() for item in task["acceptance"]) for task in tasks)
    assert public_acceptance in tasks[-1]["acceptance"]

