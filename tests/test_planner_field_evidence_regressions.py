from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai import research_derived_requirements as research
from minecraft_mod_ai.design_markdown import _markdown_list, _section_field_body
from minecraft_mod_ai.platform_evidence_pipeline import capability_queries
from minecraft_mod_ai.planner_stage_trace import PlannerStageTrace
from minecraft_mod_ai.research_facet_response import decode_facet_response
from minecraft_mod_ai.research_requirement_evidence import evidence_catalog
from minecraft_mod_ai.research_requirement_plan_slice import host_facet_baseline
from minecraft_mod_ai.root_cause_trace import emit_root_cause, trace_scope
from minecraft_mod_ai.structured_output import StructuredOutputValidationError


def _response():
    return {
        "decision": "add_obligation",
        "rationale": "External source requires duplicate rejection.",
        "evidence_refs": ["evidence:known"],
        "implementation_obligations": ["Reject duplicate trade requests"],
        "acceptance": ["A duplicate request never causes a second debit"],
    }


def test_double_encoded_array_is_recovered_from_router_exception_without_model_retry():
    expected = _response()
    raw = dict(expected, acceptance=json.dumps(expected["acceptance"]))

    class Router:
        calls = 0

        def generate_text(self, *_args, **_kwargs):
            self.calls += 1
            raise StructuredOutputValidationError(
                output=json.dumps(raw), errors=["acceptance is not an array"]
            )

    router = Router()
    value, calls, events = research._model_facet_augmentation(
        router,
        parent="req_trade",
        facet="failure_edge_cases",
        slot={},
        allowed_refs=["evidence:known"],
    )
    assert value == expected
    assert calls == router.calls == 1
    assert events[-1]["status"] == "accepted"


@pytest.mark.parametrize("invalid", [None, "arbitrary prose", "[1]", '{"x":1}', [1]])
def test_transport_repair_never_coerces_invalid_array_content(invalid):
    with pytest.raises((ValueError, TypeError)):
        decode_facet_response(dict(_response(), acceptance=invalid))


def test_facet_retry_contains_original_output_and_precise_error():
    class Router:
        messages = []

        def generate_text(self, _role, messages, **_kwargs):
            self.messages.append(messages)
            if len(self.messages) == 1:
                raise StructuredOutputValidationError(
                    output='{"acceptance":42}', errors=["bad shape"]
                )
            return _response()

    router = Router()
    value, calls, _events = research._model_facet_augmentation(
        router,
        parent="req_trade",
        facet="failure_edge_cases",
        slot={},
        allowed_refs=["evidence:known"],
    )
    assert value == _response()
    assert calls == 2
    assert router.messages[1][-2] == {
        "role": "assistant",
        "content": '{"acceptance":42}',
    }
    assert "acceptance" in router.messages[1][-1]["content"]
    assert "array" in router.messages[1][-1]["content"]


def test_internal_platform_metadata_and_bare_source_ids_are_not_external_evidence():
    internal = {
        "_platform_selection": {
            "source_id": "platform:host",
            "summary": "Server registry and persistent state are compatible",
        }
    }
    assert (
        evidence_catalog(
            {"source_id": "internal-note", "claim": "Server state"}, {}, internal
        )
        == ()
    )
    assert (
        evidence_catalog(
            {"url": "https://example.org/metadata", "status": "compatible"}, {}, {}
        )
        == ()
    )
    receipts = evidence_catalog(
        {
            "url": "https://example.org/source",
            "children": [
                {
                    "body": "The server rejects duplicate transactions.",
                    "requirement_ref": "req_trade",
                }
            ],
        },
        {},
        {},
    )
    assert len(receipts) == 1
    assert receipts[0]["summary"]["url"] == "https://example.org/source"
    assert receipts[0]["provenance_kind"] == "external_source"


def test_required_facet_without_source_owner_is_missing_even_when_prose_claims_coverage():
    requirement = {
        "requirement_id": "req_trade",
        "implementation_capabilities": ["persistence.economy_state"],
    }
    tasks = [
        {
            "task_id": "test_only",
            "owned_anchors": [{"kind": "test"}],
            "semantic_outcome": "Verify persistence",
            "acceptance": ["State persists and reloads"],
        }
    ]
    baseline = host_facet_baseline(requirement, tasks)
    assert baseline["persistence_reload"]["disposition"] == "missing"
    assert baseline["server_network_authority"]["disposition"] == "not_applicable"


def test_search_uses_gameplay_capability_instead_of_internal_design_identifier():
    queries = capability_queries(
        "",
        design={
            "modules": [
                {
                    "plugin_id": "design_economy_trade_3",
                    "capability": "economy.trade",
                    "reason": "Trade goods",
                },
            ]
        },
    )
    assert queries
    assert all("design_" not in query for query in queries)
    assert any("economy.trade" in query for query in queries)


def test_malformed_heading_is_losslessly_normalized_and_subheadings_are_not_actions():
    raw = "# ## modules\n- implementation\n# ## assets\n"
    assert (
        _section_field_body(raw, "modules", ("modules", "assets")) == "- implementation"
    )
    assert _section_field_body(raw, "assets", ("modules", "assets")) == ""
    assert _markdown_list(
        "### Explore\n- gather resources\n#### Return\n- deliver cargo"
    ) == ["gather resources", "deliver cargo"]


def test_scoped_design_module_worker_receives_only_its_own_requirement(monkeypatch):
    ledger = tuple(
        {
            "requirement_id": f"req_{name}",
            "capability": f"gameplay.{name}",
            "semantic_statement": f"Implement {name}",
            "authored_text": name,
            "acceptance": [name],
        }
        for name in ("trade", "portal")
    )
    monkeypatch.setattr(design, "_active_requirement_ledger", lambda _prompt: ledger)

    class Router:
        calls = []

        def generate_text(self, _role, messages, **_kwargs):
            self.calls.append(messages)
            return "- Apply the authored state transition atomically"

    router = Router()
    section = design._generate_section(
        router,
        prompt="Trade and portals",
        section_id="modules_and_assets",
        fields=("modules",),
        research={},
        media_paths=(),
        trace_metadata=None,
    )
    assert len(router.calls) == 2
    for index, messages in enumerate(router.calls):
        content = messages[1]["content"]
        assert ledger[index]["requirement_id"] in content
        assert ledger[1 - index]["requirement_id"] not in content
        assert section["modules"][index]["requirement_refs"] == [
            ledger[index]["requirement_id"]
        ]


def test_trace_links_stages_and_preserves_deep_full_content_with_secret_redaction(
    tmp_path, monkeypatch
):
    journal = tmp_path / "root.jsonl"
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(journal))
    monkeypatch.setenv("MMM_PLANNER_TRACE_CONSOLE", "0")
    value = {
        "rows": [
            {"text": "a" * 4096, "access_token": "never-log-me"} for _ in range(100)
        ]
    }
    for _ in range(8):
        value = {"nested": value}
    with trace_scope("complete_planning", trace_id="same-run"):
        stage = PlannerStageTrace(stage="design", prompt="request")
        emit_root_cause("full_contract", details=value)
    record = json.loads(journal.read_text().splitlines()[-1])
    assert record["trace_id"] == stage.run_id == "same-run"
    receipt = record["details_artifact"]
    artifact = Path(receipt["path"]).read_text()
    assert "never-log-me" not in artifact
    assert "<depth-limit>" not in artifact
    restored = json.loads(artifact)
    for _ in range(8):
        restored = restored["nested"]
    assert len(restored["rows"]) == 100
    assert restored["rows"][99]["text"] == "a" * 4096
    assert restored["rows"][99]["access_token"] == "<redacted>"


def test_host_missing_owner_is_rejected_before_optional_research(monkeypatch):
    monkeypatch.setattr(
        research, "validate_evidence_first_plan", lambda *_a, **_k: None
    )

    class Router:
        def generate_text(self, *_a, **_k):
            pytest.fail("Host structural failure spent a model turn")

    plan = {
        "request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_trade",
                    "implementation_capabilities": ["persistence.economy_state"],
                }
            ]
        },
        "tasks": [],
    }
    with pytest.raises(
        research.ResearchRequirementError, match="missing implementation facets"
    ):
        research.derive_research_requirements(
            Router(),
            prompt="Trade",
            evidence_plan=plan,
            research_brief={},
            technical_evidence={},
            game_design={},
        )


def test_verified_retained_capability_needs_no_new_implementation_owner():
    requirement = {
        "requirement_id": "req_trade",
        "implementation_capabilities": ["persistence.economy_state"],
    }
    plan = {
        "reuse_decisions": [
            {
                "requirement_ref": "req_trade",
                "action": "retain",
                "component_refs": ["component:verified-trade"],
            }
        ]
    }
    baseline = research._baseline_for_requirement(plan, requirement, [])
    assert baseline["persistence_reload"]["disposition"] == "already_covered"
    assert "component:verified-trade" in baseline["persistence_reload"]["rationale"]
    plan["reuse_decisions"][0]["action"] = "adapt"
    assert (
        research._baseline_for_requirement(plan, requirement, [])["persistence_reload"][
            "disposition"
        ]
        == "missing"
    )
