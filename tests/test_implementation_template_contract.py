from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.implementation_template_contract import (
    MODEL_FILL_FIELDS,
    SCHEMA,
    build_implementation_template,
    install,
    sanitize_hole_fills,
)


def _task() -> dict:
    return {
        "task_id": "task_trade_service",
        "task_sha256": "sha256:input",
        "semantic_outcome": "Implement server-authoritative trade",
        "requirement_refs": ["req_trade"],
        "gap_refs": ["gap_trade"],
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "java_version": 21,
        },
        "implementation_capabilities": [
            "economy.transaction_service",
            "network.server_authority",
            "persistence.trade_state",
        ],
        "design_resolution_obligations": ["Choose one atomic transaction boundary"],
        "artifact_obligations": [
            {
                "artifact_id": "artifact_source",
                "kind": "source_code",
                "locator": "src/main/java/example/TradeService.java",
            },
            {
                "artifact_id": "artifact_network",
                "kind": "network_protocol",
                "locator": "unresolved:network_protocol",
            },
            {
                "artifact_id": "artifact_state",
                "kind": "persistence_schema",
                "locator": "unresolved:persistence_schema",
            },
            {
                "artifact_id": "artifact_test",
                "kind": "verification_artifact",
                "locator": "src/test/java/example/TradeServiceTest.java",
            },
        ],
        "consumes": ["economy_model:trade"],
        "provides": ["capability:economy.trade"],
        "depends_on": [],
        "required_gates": [
            "target_compile",
            "network_protocol_validation",
            "runtime_gameplay_validation",
        ],
        "public_acceptance": [
            "Given sufficient funds, when buying, then debit exactly once and grant the item."
        ],
        "runtime_acceptance": [
            "Reject insufficient funds without mutating balance, stock, or inventory."
        ],
        "reuse_refs": ["donor:verified:trade"],
        "conditional_predicates": ["needs_network", "needs_persistence"],
        "owned_anchors": [],
    }


def test_template_is_dynamic_detailed_and_stable() -> None:
    first = build_implementation_template(_task())
    second = build_implementation_template(_task())

    assert first["schema_version"] == SCHEMA
    assert first["template_sha256"] == second["template_sha256"]
    assert [hole["hole_id"] for hole in first["holes"]] == [
        hole["hole_id"] for hole in second["holes"]
    ]

    kinds = [hole["kind"] for hole in first["holes"]]
    assert kinds.count("implementation_capability") == 3
    assert kinds.count("design_resolution") == 1
    assert kinds.count("artifact_implementation") == 4
    assert kinds.count("dataflow_input") == 1
    assert kinds.count("dataflow_output") == 1
    assert kinds.count("verification_gate") == 3
    assert kinds.count("public_acceptance") == 1
    assert kinds.count("runtime_acceptance") == 1
    assert kinds.count("reference_adaptation") == 1
    assert len(first["completion_policy"]["required_hole_ids"]) == len(first["holes"])

    check_ids = {item["check_id"] for item in first["minecraft_checklist"]}
    assert {
        "source_ownership",
        "persistent_state_round_trip",
        "server_authority",
        "network_side_safety",
        "verification_from_behavior",
        "runtime_acceptance",
    } <= check_ids
    assert first["target_constraints"]["minecraft_version"] == "1.21.1"
    assert first["target_constraints"]["loader"] == "fabric"


def test_model_cannot_add_holes_or_write_host_owned_fields() -> None:
    template = build_implementation_template(_task())
    first_id = template["holes"][0]["hole_id"]
    fills = sanitize_hole_fills(
        template,
        [
            {
                "hole_id": first_id,
                "implementation_decision": "Use one server-side transaction service.",
                "local_steps": ["validate", "debit", "grant", "sync"],
                "target_coordinates": {"minecraft_version": "1.20.1"},
                "evil": "ignored",
            },
            {
                "hole_id": "hole_invented_by_model",
                "implementation_decision": "not allowed",
            },
        ],
    )

    assert fills == [
        {
            "hole_id": first_id,
            "implementation_decision": "Use one server-side transaction service.",
            "local_steps": ["validate", "debit", "grant", "sync"],
        }
    ]
    assert set(fills[0]) <= {"hole_id", *MODEL_FILL_FIELDS}


def test_install_preserves_detailed_plan_in_small_model_capsule_surface() -> None:
    base_task = _task()

    def compile_tasks(_gaps, _reuse, _target, _branches, _ownership):
        return (dict(base_task),)

    def hash_without(value, field):
        payload = dict(value)
        payload[field] = ""
        return "sha256:" + str(len(repr(sorted(payload))))

    planning = SimpleNamespace(_compile_tasks=compile_tasks, _hash_without=hash_without)

    def build_batch_skeleton(*_args, **_kwargs):
        return {
            "modules": [],
            "acceptance_tests": ["fake_default"],
            "completed_deliverables": ["fake_default"],
        }

    def merge_model_output_into_skeleton(skeleton, _model_output, _catalog):
        return skeleton

    planner_template = SimpleNamespace(
        MODEL_TASK_DETAIL_KEYS=frozenset({"implementation_notes"}),
        build_batch_skeleton=build_batch_skeleton,
        merge_model_output_into_skeleton=merge_model_output_into_skeleton,
    )
    capsule = SimpleNamespace(_COMPACT_TASK_FIELDS=("task_id", "required_gates"))

    install(
        planning_module=planning,
        planner_template_module=planner_template,
        task_capsule_module=capsule,
    )

    compiled = planning._compile_tasks([], [], {}, {}, {})[0]
    assert compiled["implementation_template"]["schema_version"] == SCHEMA
    assert "implementation_capabilities" in capsule._COMPACT_TASK_FIELDS
    assert "design_resolution_obligations" in capsule._COMPACT_TASK_FIELDS
    assert "runtime_acceptance" in capsule._COMPACT_TASK_FIELDS
    assert "implementation_template" in capsule._COMPACT_TASK_FIELDS
    assert "hole_fills" in planner_template.MODEL_TASK_DETAIL_KEYS
