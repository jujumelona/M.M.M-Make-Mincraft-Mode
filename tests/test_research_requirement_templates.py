from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.research_derived_requirements import _model_facet_augmentation
from minecraft_mod_ai.research_requirement_template import (
    FACET_AUGMENTATION_RESPONSE_SCHEMA,
    build_facet_slot,
    build_host_planning_context,
)


def _selection() -> dict:
    return {
        "source": "platform_resolver",
        "explicit_version": True,
        "explicit_loader": True,
        "preserved_existing_target": False,
        "migration_requested": False,
        "target": {
            "minecraft_version": "1.21.8",
            "loader": "fabric",
            "java_version": 21,
            "mappings": "1.21.8+build.1",
            "fabric_loader": "0.17.2",
            "fabric_api": "0.136.0+1.21.8",
            "fabric_loom": "1.11-SNAPSHOT",
            "datapack_version": 81,
            "resource_pack_version": 64,
        },
    }


def _slot(planning_context: dict) -> dict:
    return build_facet_slot(
        planning_context=planning_context,
        requirement={
            "requirement_id": "req_space_travel",
            "capability": "space_travel_mechanic",
            "statement": "Travel to another planet with a completed spacecraft.",
            "implementation_capabilities": None,
            "artifact_obligations": None,
            "acceptance": ["Arrival occurs only after valid launch."],
        },
        facet="server_network_authority",
        baseline={
            "facet": "server_network_authority",
            "disposition": "implemented",
            "statement": "Host task already owns authoritative launch validation.",
            "rationale": "Existing task slice contains server authority.",
            "evidence_refs": [],
            "acceptance": ["Client cannot force launch."],
            "implementation_obligations": ["Validate launch on server."],
        },
        task_slice=[
            {
                "task_id": "task_space_travel_authority",
                "kind": "domain_service",
                "acceptance": ["Reject invalid launch request."],
            }
        ],
        evidence_catalog=[
            {
                "evidence_ref": "evidence:server-authority",
                "summary": {"claim": "Server validates travel requests."},
            }
        ],
        allowed_evidence_refs=["evidence:server-authority"],
    )


def test_notebook_target_is_host_owned_and_auto_never_moves_to_model() -> None:
    explicit_router = SimpleNamespace(
        _mmm_requested_minecraft_version="1.21.8",
        _mmm_requested_loader="fabric",
    )
    explicit = build_host_planning_context(
        explicit_router,
        {"_platform_selection": _selection()},
    )
    assert explicit["target"]["minecraft_version"] == "1.21.8"
    assert explicit["target"]["loader"] == "fabric"
    assert explicit["target"]["java_version"] == 21
    assert explicit["requested_constraints"]["minecraft_version_source"] == "user_notebook"
    assert explicit["requested_constraints"]["loader_source"] == "user_notebook"

    auto = build_host_planning_context(
        SimpleNamespace(
            _mmm_requested_minecraft_version=None,
            _mmm_requested_loader=None,
        ),
        {"_platform_selection": _selection()},
    )
    assert auto["target"] == explicit["target"]
    assert auto["requested_constraints"]["minecraft_version"] is None
    assert auto["requested_constraints"]["loader"] is None
    assert auto["requested_constraints"]["minecraft_version_source"] == "host_selector"
    assert auto["requested_constraints"]["loader_source"] == "host_selector"


def test_small_model_schema_contains_only_semantic_slot_fields() -> None:
    properties = set(FACET_AUGMENTATION_RESPONSE_SCHEMA["properties"])
    assert properties == {
        "decision",
        "rationale",
        "evidence_refs",
        "implementation_obligations",
        "acceptance",
    }
    forbidden = {
        "minecraft_version",
        "loader",
        "java_version",
        "mappings",
        "facet",
        "requirement_id",
        "task_id",
        "path",
        "host_baseline",
    }
    assert not properties.intersection(forbidden)
    assert FACET_AUGMENTATION_RESPONSE_SCHEMA["additionalProperties"] is False


def test_host_template_keeps_versions_ids_tasks_and_baseline_outside_model_slot() -> None:
    context = build_host_planning_context(
        SimpleNamespace(
            _mmm_requested_minecraft_version="1.21.8",
            _mmm_requested_loader="fabric",
        ),
        {"_platform_selection": _selection()},
    )
    slot = _slot(context)
    host = slot["host_owned"]
    assert host["planning_context"]["target"]["minecraft_version"] == "1.21.8"
    assert host["requirement"]["requirement_id"] == "req_space_travel"
    assert host["facet"] == "server_network_authority"
    assert host["host_task_slice"][0]["task_id"] == "task_space_travel_authority"
    assert host["host_baseline"]["disposition"] == "implemented"
    assert host["requirement"]["implementation_capabilities"] == []
    assert host["requirement"]["artifact_obligations"] == []

    model_slot_text = json.dumps(slot["model_slot"], ensure_ascii=False)
    assert "1.21.8" not in model_slot_text
    assert "task_space_travel_authority" not in model_slot_text
    assert "req_space_travel" not in model_slot_text


class _BrokenRouter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, *_args, **_kwargs):
        self.calls += 1
        return ""


def test_empty_model_output_retries_once_then_keeps_host_baseline() -> None:
    context = build_host_planning_context(
        SimpleNamespace(
            _mmm_requested_minecraft_version="1.21.8",
            _mmm_requested_loader="fabric",
        ),
        {"_platform_selection": _selection()},
    )
    router = _BrokenRouter()
    candidate, calls, events = _model_facet_augmentation(
        router,
        parent="req_space_travel",
        facet="server_network_authority",
        slot=_slot(context),
        allowed_refs=["evidence:server-authority"],
    )
    assert candidate is None
    assert calls == 2
    assert router.calls == 2
    assert [event["status"] for event in events] == ["retry", "fallback_host_baseline"]
    assert events[-1]["fallback_used"] is True
    assert events[-1]["requirement_id"] == "req_space_travel"
    assert events[-1]["facet"] == "server_network_authority"
    assert events[-1]["error_type"]
    assert "raw_response_snippet" in events[-1]


class _BoundedRouter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, _role, messages, **kwargs):
        self.calls += 1
        assert kwargs["enable_tools"] is False
        assert kwargs["response_schema"] is FACET_AUGMENTATION_RESPONSE_SCHEMA
        host_payload = json.loads(messages[1]["content"])
        assert host_payload["host_owned"]["planning_context"]["target"]["minecraft_version"] == "1.21.8"
        return {
            "decision": "add_obligation",
            "rationale": "The supplied evidence adds one server validation rule.",
            "evidence_refs": ["evidence:server-authority"],
            "implementation_obligations": ["Validate destination eligibility on the server."],
            "acceptance": ["Invalid client destination requests are rejected."],
        }


def test_valid_model_call_can_only_return_bounded_semantic_addition() -> None:
    context = build_host_planning_context(
        SimpleNamespace(
            _mmm_requested_minecraft_version="1.21.8",
            _mmm_requested_loader="fabric",
        ),
        {"_platform_selection": _selection()},
    )
    router = _BoundedRouter()
    candidate, calls, events = _model_facet_augmentation(
        router,
        parent="req_space_travel",
        facet="server_network_authority",
        slot=_slot(context),
        allowed_refs=["evidence:server-authority"],
    )
    assert calls == 1
    assert router.calls == 1
    assert candidate == {
        "decision": "add_obligation",
        "rationale": "The supplied evidence adds one server validation rule.",
        "evidence_refs": ["evidence:server-authority"],
        "implementation_obligations": ["Validate destination eligibility on the server."],
        "acceptance": ["Invalid client destination requests are rejected."],
    }
    assert events[-1]["status"] == "accepted"
