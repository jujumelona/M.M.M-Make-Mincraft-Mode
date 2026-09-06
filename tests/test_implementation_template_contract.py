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
        "internal_invariants": ["Trade mutation is server-authoritative and atomic."],
        "acceptance": ["Trade mutation is server-authoritative and atomic."],
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
    def __init__(self, *, omit_last: bool = False, malformed: bool = False) -> None:
        self.calls = 0
        self.omit_last = omit_last
        self.malformed = malformed
        self.enable_tools: list[object] = []
        self.requested_hole_ids: list[list[str]] = []

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        assert kwargs.get("response_format") == "text"
        self.enable_tools.append(kwargs.get("enable_tools"))
        self.calls += 1
        if self.malformed:
            return '{"broken":'
        packet = json.loads(messages[-1]["content"].split("\n", 1)[1])
        if "modules" in packet:
            holes = list(packet["modules"][0]["implementation_template"]["holes"])
            self.requested_hole_ids.append([hole["hole_id"] for hole in holes])
            returned = holes[:-1] if self.omit_last and holes else holes
            return "\n".join(
                f"### Hole {index}\nDecision: Implement {hole['subject']} precisely\n"
                "Steps:\n- Apply the host-owned contract.\n"
                "Verification: Run the supplied host gate."
                for index, hole in enumerate(returned, 1)
            )
        pages = list(packet["pages"])
        holes = [hole for page in pages for hole in page["holes"]]
        self.requested_hole_ids.append([hole["hole_id"] for hole in holes])
        if self.omit_last and holes:
            omitted = holes[-1]["hole_id"]
        else:
            omitted = ""
        blocks = []
        for page in pages:
            for hole in page["holes"]:
                if hole["hole_id"] == omitted:
                    continue
                blocks.append(
                    f"BEGIN PAGE {page['page_id']}\nBEGIN {hole['hole_id']}\n"
                    f"Decision: Implement {hole['subject']} precisely\n"
                    "Steps:\n- Apply the host-owned contract.\n"
                    "Verification: Run the supplied host gate.\n"
                    f"END {hole['hole_id']}\nEND PAGE {page['page_id']}"
                )
        return "\n".join(blocks)


def test_template_is_dynamic_detailed_and_stable() -> None:
    first = build_implementation_template(_task())
    second = build_implementation_template(_task())

    assert first["schema_version"] == SCHEMA
    assert first["template_sha256"] == second["template_sha256"]
    assert [hole["hole_id"] for hole in first["holes"]] == [
        hole["hole_id"] for hole in second["holes"]
    ]
    assert len(first["completion_policy"]["required_hole_ids"]) == len(first["holes"])
    assert first["target_constraints"]["minecraft_version"] == "1.21.1"
    assert first["target_constraints"]["loader"] == "fabric"
    check_ids = {item["check_id"] for item in first["minecraft_checklist"]}
    assert {
        "source_ownership",
        "persistent_state_round_trip",
        "server_authority",
        "network_side_safety",
        "verification_from_behavior",
        "runtime_acceptance",
    } <= check_ids


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


def test_small_model_capsule_gets_host_template_from_full_task() -> None:
    task = _task()
    module = SimpleNamespace(
        module_id=task["task_id"],
        kind="custom_java",
        config={"evidence_task": task},
        depends_on=[],
        required_gates=task["required_gates"],
    )
    template = compact_task_local_module_contract(module)["evidence_task"][
        "implementation_template"
    ]
    assert template["schema_version"] == SCHEMA
    assert template["task_ref"] == task["task_id"]
    assert template["host_owned"]["artifact_obligations"] == task["artifact_obligations"]
    assert template["host_owned"]["required_gates"] == task["required_gates"]


def test_planner_skeleton_keeps_host_owned_contract_when_model_refines() -> None:
    task = _task()
    skeleton = build_batch_skeleton(
        task["task_id"],
        task["semantic_outcome"],
        [],
        [task["task_id"]],
        host_module_contracts={task["task_id"]: task},
    )
    config = skeleton["modules"][0]["config"]
    template = config["implementation_template"]
    first_id = template["holes"][0]["hole_id"]
    merged = merge_model_output_into_skeleton(
        skeleton,
        {
            "modules": [
                {
                    "module_id": task["task_id"],
                    "kind": "boss",
                    "depends_on": ["invented_dependency"],
                    "required_gates": ["invented_gate"],
                    "config": {
                        "hole_fills": [
                            {
                                "hole_id": first_id,
                                "implementation_decision": "Use the shared transaction service.",
                                "target_coordinates": {"minecraft_version": "1.20.1"},
                            }
                        ]
                    },
                }
            ]
        },
        {task["task_id"]},
    )
    module = merged["modules"][0]
    assert module["kind"] == "custom_java"
    assert module["depends_on"] == []
    assert module["required_gates"] == task["required_gates"]
    assert module["config"]["implementation_template"] == template
    assert module["config"]["model_fill"]["hole_fills"] == [
        {
            "hole_id": first_id,
            "implementation_decision": "Use the shared transaction service.",
        }
    ]


def test_hole_filler_uses_one_optional_refinement_and_host_defaults_for_omissions() -> None:
    task = _task()
    skeleton = build_batch_skeleton(
        task["task_id"],
        task["semantic_outcome"],
        task["provides"],
        [task["task_id"]],
        host_module_contracts={task["task_id"]: task},
    )
    router = _HoleRouter(omit_last=True)

    page = fill_evidence_page(
        router,
        skeleton,
        valid_module_catalog={task["task_id"]},
    )
    config = page["modules"][0]["config"]
    template = config["implementation_template"]
    fills = config["model_fill"]["hole_fills"]

    assert router.calls >= 1
    assert router.enable_tools == [False] * router.calls
    assert {item["hole_id"] for item in fills} == set(
        template["completion_policy"]["required_hole_ids"]
    )
    assert all(item["implementation_decision"] for item in fills)
    assert all(item["verification_intent"] for item in fills)


def test_malformed_hole_refinement_does_not_destroy_plan() -> None:
    task = _task()
    skeleton = build_batch_skeleton(
        task["task_id"],
        task["semantic_outcome"],
        task["provides"],
        [task["task_id"]],
        host_module_contracts={task["task_id"]: task},
    )
    page = fill_evidence_page(
        _HoleRouter(malformed=True),
        skeleton,
        valid_module_catalog={task["task_id"]},
    )
    config = page["modules"][0]["config"]
    required = set(config["implementation_template"]["completion_policy"]["required_hole_ids"])
    fills = config["model_fill"]["hole_fills"]
    assert {item["hole_id"] for item in fills} == required
    assert all(item["implementation_decision"] for item in fills)


def test_canonical_evidence_batch_invokes_only_bounded_optional_refinement() -> None:
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

    assert router.calls >= 1
    assert all(0 < len(request) <= 12 for request in router.requested_hole_ids)
    assert router.enable_tools == [False] * router.calls
    assert not assets
    assert tests
    assert len(modules) == 1
    config = modules[0].config
    required = set(config["implementation_template"]["completion_policy"]["required_hole_ids"])
    filled = {item["hole_id"] for item in config["model_fill"]["hole_fills"]}
    assert filled == required
