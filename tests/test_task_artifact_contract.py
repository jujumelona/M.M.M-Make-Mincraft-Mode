from __future__ import annotations

from minecraft_mod_ai import task_artifact_contract as contract


def _task(task_id, req, *, depends=(), acceptance=(), predicates=(), anchor_kind="symbol"):
    return {
        "task_id": task_id,
        "semantic_outcome": "Implement one independently verifiable outcome for example.capability",
        "gap_refs": ["gap_x"],
        "requirement_refs": [req],
        "target_cell": {},
        "owned_anchors": [
            {
                "kind": anchor_kind,
                "locator": f"locator:{task_id}",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": "root",
                "source_set": "main",
            }
        ],
        "reuse_refs": [],
        "consumes": [],
        "provides": [f"provide:{task_id}"],
        "depends_on": list(depends),
        "conditional_predicates": list(predicates),
        "required_gates": ["target_compile"],
        "acceptance": list(acceptance) or [
            f"{task_id}: all declared provides exist and all owned anchors pass their integrity checks"
        ],
        "done_predicate": {"operator": "all", "checks": ["required_gates_passed"]},
        "impact_probes": [],
        "state": "pending",
        "task_sha256": "",
    }


def test_task_acceptance_is_split_into_public_and_internal_fields():
    task = _task(
        "task_child",
        "req_child",
        acceptance=(
            "task_child: all declared provides exist and all owned anchors pass their integrity checks",
            "Given a player has currency; when they trade; then the requested exchange completes.",
        ),
    )
    gap = {
        "requirement_ref": "req_child",
        "capability": "economy.trade",
        "acceptance": [
            "Given a player has currency; when they trade; then the requested exchange completes."
        ],
    }
    contract._REQUIREMENT_DEPS.set({"req_child": ()})
    contract._REQUIREMENT_PROVIDES.set({"req_child": ("provide:task_child",)})
    result = contract._postprocess_tasks([task], [gap])[0]

    assert result["acceptance"] == result["internal_invariants"]
    assert result["public_acceptance"] == [
        "Given a player has currency; when they trade; then the requested exchange completes."
    ]
    assert "Implement one independently verifiable outcome" not in result["semantic_outcome"]
    assert "Given a player has currency" in result["semantic_outcome"]


def test_gameplay_requirement_dependency_becomes_task_dependency_edge():
    parent = _task("task_parent", "req_parent")
    child = _task("task_child", "req_child")
    gaps = [
        {"requirement_ref": "req_parent", "capability": "resource.gather", "acceptance": ["Given ore; when mined; then resource is obtained."]},
        {"requirement_ref": "req_child", "capability": "economy.trade", "acceptance": ["Given resource; when traded; then exchange completes."]},
    ]
    contract._REQUIREMENT_DEPS.set(
        {"req_parent": (), "req_child": ("req_parent",)}
    )
    contract._REQUIREMENT_PROVIDES.set(
        {
            "req_parent": ("provide:task_parent",),
            "req_child": ("provide:task_child",),
        }
    )

    result = {item["task_id"]: item for item in contract._postprocess_tasks([parent, child], gaps)}

    assert result["task_child"]["consumes"] == ["provide:task_parent"]
    assert result["task_child"]["depends_on"] == ["task_parent"]
    assert result["task_child"]["dependency_reasons"]["task_parent"] == {
        "kind": "requirement_dataflow",
        "requirement_ref": "req_parent",
    }


def test_architecture_generates_artifact_obligations_without_authored_assets():
    task = _task(
        "task_ui",
        "req_ui",
        predicates=("needs_client_render", "needs_datagen"),
        anchor_kind="resource",
    )
    task["artifact_obligations"] = contract._artifact_obligations(task)
    plan = {"tasks": [task]}

    artifacts = contract._artifact_plan(plan, {"assets": []})

    assert artifacts["required_artifacts"]
    assert artifacts["asset_requirement_status"] == "REQUIRED_UNRESOLVED"
    assert artifacts["zero_asset_justification"] == ""
    kinds = {item["kind"] for item in artifacts["required_artifacts"]}
    assert "client_visual_or_ui_resource" in kinds
    assert "generated_data_resource" in kinds


def test_zero_assets_require_explicit_architecture_justification():
    task = _task("task_code", "req_code", anchor_kind="symbol")
    task["artifact_obligations"] = contract._artifact_obligations(task)
    artifacts = contract._artifact_plan({"tasks": [task]}, {"assets": []})

    assert artifacts["asset_requirement_status"] == "NOT_REQUIRED_BY_ARCHITECTURE"
    assert artifacts["zero_asset_justification"]


def test_design_choices_and_implementation_obligations_have_separate_provenance():
    task = _task(
        "task_screen",
        "req_trade",
        predicates=("needs_client_render",),
        anchor_kind="resource",
    )
    task["semantic_outcome"] = "Render and bind the client screen for economy.trade"
    task["artifact_obligations"] = contract._artifact_obligations(task)
    resolution = contract._design_resolution({"tasks": [task]})

    # A derived implementation branch is not a user-authored or evidence-selected
    # design alternative. Selection claims require an explicit comparative receipt.
    assert resolution["selected_design_alternatives"] == []
    assert resolution["derived_architecture_decisions"]
    assert all(
        item["provenance_role"] == "derived_architecture_decision"
        for item in resolution["derived_architecture_decisions"]
    )
    assert all(
        item["provenance_role"] == "implementation_obligation"
        for item in resolution["implementation_obligations"]
    )


def test_module_ownership_never_uses_gradle_root_path_as_logical_id():
    result = contract._normalize_ownership({}, {"module_id": ":", "topology_module_ids": [":", ":client"]})
    assert result["module_id"] == "root"
    assert result["gradle_project_path"] == ":"
    assert result["topology_module_ids"] == ["root", "client"]
