from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _ProductionBatch
from minecraft_mod_ai.implementation_template_contract import (
    MODEL_FILL_FIELDS,
    SCHEMA,
    build_implementation_template,
    sanitize_hole_fills,
)
from minecraft_mod_ai.planner_hole_filling import fill_evidence_page
from minecraft_mod_ai.planner_template_schema import (
    build_batch_skeleton,
    merge_model_output_into_skeleton,
)
from minecraft_mod_ai.small_model_task_capsule_contract import (
    compact_task_local_module_contract,
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
        "internal_invariants": [
            "Trade mutation is server-authoritative and atomic."
        ],
        "acceptance": [
            "Trade mutation is server-authoritative and atomic."
        ],
        "runtime_acceptance": [
            "Reject insufficient funds without mutating balance, stock, or inventory."
        ],
        "reuse_refs": ["donor:verified:trade"],
        "conditional_predicates": ["needs_network", "needs_persistence"],
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": "src/main/java/example/TradeService.java#TradeService",
                "status": "host_reserved",
                "ownership": "exclusive",
                "module_id": ":",
                "source_set": "main",
            }
        ],
        "production_bindings": [
            {
                "task_ref": "task_trade_service",
                "reuse_action": "adapt",
                "owned_anchors": [
                    {
                        "kind": "symbol",
                        "locator": "src/main/java/example/TradeService.java#TradeService",
                        "status": "host_reserved",
                    }
                ],
            }
        ],
    }


class _HoleRouter:
    def __init__(self, *, omit_once: bool = False) -> None:
        self.calls = 0
        self.omit_once = omit_once
        self.enable_tools: list[object] = []

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs.get("response_format") == "json"
        self.enable_tools.append(kwargs.get("enable_tools"))
        self.calls += 1
        packet = json.loads(messages[-1]["content"].split("\n", 1)[1])
        modules = []
        for module in packet["modules"]:
            holes = list(module["implementation_template"]["holes"])
            if self.omit_once and self.calls == 1 and holes:
                holes = holes[:-1]
            modules.append(
                {
                    "module_id": module["module_id"],
                    "config": {
                        "implementation_notes": "Implement the host sketch without changing authority.",
                        "hole_fills": [
                            {
                                "hole_id": hole["hole_id"],
                                "implementation_decision": f"Implement {hole['subject']}",
                                "local_steps": [
                                    "Read the host-owned anchors and constraints.",
                                    "Implement the bounded obligation.",
                                    "Run the declared verification gate.",
                                ],
                                "code_bindings": [],
                                "reference_uses": [],
                                "verification_intent": "Produce host-verifiable evidence.",
                                "uncertainties": [],
                            }
                            for hole in holes
                        ],
                    },
                }
            )
        return json.dumps({"modules": modules})


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


def test_small_model_capsule_gets_detailed_template_from_full_task() -> None:
    task = _task()
    module = SimpleNamespace(
        module_id=task["task_id"],
        kind="custom_java",
        config={"evidence_task": task},
        depends_on=[],
        required_gates=task["required_gates"],
    )

    payload = compact_task_local_module_contract(module)
    compact = payload["evidence_task"]
    template = compact["implementation_template"]

    assert template["schema_version"] == SCHEMA
    assert template["task_ref"] == task["task_id"]
    assert len(template["holes"]) == 16
    assert {
        hole["subject"]
        for hole in template["holes"]
        if hole["kind"] == "implementation_capability"
    } == set(task["implementation_capabilities"])
    assert template["host_owned"]["artifact_obligations"] == task["artifact_obligations"]
    assert template["host_owned"]["required_gates"] == task["required_gates"]


def test_planner_skeleton_uses_real_contract_and_sanitizes_hole_fills() -> None:
    task = _task()
    contracts = {task["task_id"]: task}
    skeleton = build_batch_skeleton(
        task["task_id"],
        task["semantic_outcome"],
        [],
        [task["task_id"]],
        host_module_contracts=contracts,
    )

    assert skeleton["acceptance_tests"] == [
        *task["public_acceptance"],
        *task["runtime_acceptance"],
        *task["acceptance"],
    ]
    assert skeleton["completed_deliverables"] == task["provides"]
    assert not any(
        item.startswith("test_task_trade_service_registers")
        for item in skeleton["acceptance_tests"]
    )

    config = skeleton["modules"][0]["config"]
    template = config["implementation_template"]
    first_id = template["holes"][0]["hole_id"]
    model_output = {
        "modules": [
            {
                "module_id": task["task_id"],
                "kind": "boss",
                "depends_on": ["invented_dependency"],
                "required_gates": ["invented_gate"],
                "config": {
                    "implementation_notes": "Adapt the exact host task.",
                    "hole_fills": [
                        {
                            "hole_id": first_id,
                            "implementation_decision": "Use the shared transaction service.",
                            "target_coordinates": {"minecraft_version": "1.20.1"},
                        },
                        {
                            "hole_id": "hole_model_invented",
                            "implementation_decision": "must disappear",
                        },
                    ],
                },
            }
        ],
        "acceptance_tests": ["model_replaced_acceptance"],
        "completed_deliverables": ["model_replaced_deliverable"],
    }
    merged = merge_model_output_into_skeleton(
        skeleton,
        model_output,
        {task["task_id"]},
    )
    module = merged["modules"][0]

    assert module["kind"] == "custom_java"
    assert module["depends_on"] == []
    assert module["required_gates"] == task["required_gates"]
    assert merged["acceptance_tests"] == skeleton["acceptance_tests"]
    assert merged["completed_deliverables"] == skeleton["completed_deliverables"]
    assert module["config"]["implementation_template"] == template
    assert module["config"]["model_fill"]["hole_fills"] == [
        {
            "hole_id": first_id,
            "implementation_decision": "Use the shared transaction service.",
        }
    ]


def test_hole_filler_repairs_only_missing_holes() -> None:
    task = _task()
    skeleton = build_batch_skeleton(
        task["task_id"],
        task["semantic_outcome"],
        task["provides"],
        [task["task_id"]],
        host_module_contracts={task["task_id"]: task},
    )
    router = _HoleRouter(omit_once=True)

    page = fill_evidence_page(
        router,
        skeleton,
        valid_module_catalog={task["task_id"]},
    )
    template = page["modules"][0]["config"]["implementation_template"]
    fills = page["modules"][0]["config"]["model_fill"]["hole_fills"]

    assert router.calls == 2
    assert router.enable_tools == [False, False]
    assert {item["hole_id"] for item in fills} == set(
        template["completion_policy"]["required_hole_ids"]
    )


def test_canonical_evidence_batch_invokes_bounded_hole_filler() -> None:
    task = _task()
    batch = _ProductionBatch(
        batch_id=task["task_id"],
        scope=task["semantic_outcome"],
        depends_on_batches=(),
        deliverables=tuple(task["provides"]),
        exports=(task["task_id"],),
        task_contract=task,
        evidence_plan_sha256="sha256:evidence",
    )
    router = _HoleRouter()

    modules, assets, tests = CompleteGameDesignPlanner(router)._expand_batches(
        (batch,),
        prompt="Create a server-authoritative trade system.",
        game_design={},
        evidence_mode=True,
    )

    assert router.calls == 1
    assert not assets
    assert tests
    assert len(modules) == 1
    config = modules[0].config
    required = set(
        config["implementation_template"]["completion_policy"]["required_hole_ids"]
    )
    filled = {
        item["hole_id"] for item in config["model_fill"]["hole_fills"]
    }
    assert filled == required
