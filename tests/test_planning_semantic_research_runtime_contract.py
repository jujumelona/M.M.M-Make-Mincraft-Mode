from types import SimpleNamespace

from minecraft_mod_ai import planning_semantic_research as semantic
from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.planning_candidate_evidence import (
    global_grounded_pool,
    requirement_candidate_trace,
)


def requirement():
    return {
        "decision_type": "requirement",
        "requirement_id": "req_runtime",
        "semantic_capability": "spacecraft trade upgrade",
        "statement": "Players trade resources to upgrade spacecraft.",
        "acceptance": ["Players can trade resources to upgrade spacecraft."],
    }


def pool_for(body):
    return global_grounded_pool({
        "r_runtime": {
            "queries": [{
                "query": "spacecraft trade upgrade",
                "evidence_records": [{
                    "source_id": "modrinth:runtime",
                    "content": body,
                    "url": "https://example.invalid/runtime",
                    "title": "Runtime fixture",
                }],
            }]
        }
    })


def test_runtime_assessment_contract_has_no_model_authored_prose():
    properties = semantic._ASSESSMENT_SCHEMA["properties"]
    assert set(properties) == {"verdict", "evidence_start", "evidence_end"}
    assert properties["verdict"]["enum"] == list(semantic._VERDICTS)
    assert semantic._VERIFICATION_SCHEMA["properties"] == {
        "verdict": {"type": "string", "enum": list(semantic._VERDICTS)}
    }
    assert "excerpt" not in properties
    assert "reason" not in properties


def test_structured_output_fixed_point_rejects_one_source_without_killing_planning(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(semantic, "request_message_budget", lambda *_: 4096)

    class Router:
        profile = "runtime"
        registry = SimpleNamespace(
            role=lambda *a: SimpleNamespace(adapter="llama_cpp", extra={})
        )

        def generate_tool_decision(self, *args, **kwargs):
            raise ModelConfigurationError(
                "Host-selected action 'assess_requirement_source' repeated-invalid forced "
                "argument-page fixed point on page 1/1; error=ToolCallValidationError: "
                "schema-invalid arguments at reason"
            )

    req = requirement()
    pool = pool_for(
        "A spacecraft exists, but this page does not document the requested trade upgrade."
    )
    result = semantic.review_requirement_sources(
        Router(), req, pool, requirement_candidate_trace(req, pool)
    )

    assert not result["complete"]
    assert result["schema_version"] == "mmm/semantic-research-review-v2"
    assert result["observations"]
    assert (
        result["observations"][0]["assessment_error"]
        == "model_structured_output_invalid"
    )
    assert result["observations"][0]["verdict"] == "invalid_output"


def test_host_reconstructs_exact_evidence_and_proof_text(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(semantic, "request_message_budget", lambda *_: 4096)
    calls = []

    def model(*args, **kwargs):
        calls.append(kwargs["tool_name"])
        if kwargs["tool_name"] == "assess_requirement_source":
            return {"verdict": "supported", "evidence_start": 0, "evidence_end": 0}
        return {"verdict": "supported"}

    monkeypatch.setattr(semantic, "generate_fixed_template_value", model)
    req = requirement()
    body = "Players can trade mined resources to upgrade spacecraft engines and weapons."
    pool = pool_for(body)
    result = semantic.review_requirement_sources(
        SimpleNamespace(), req, pool, requirement_candidate_trace(req, pool)
    )

    assert calls == ["assess_requirement_source", "verify_requirement_entailment"]
    assert result["complete"]
    proof = result["accepted_proofs"][0]
    assert proof["excerpt"] == body
    assert proof["source_id"] == "modrinth:runtime"
    assert "host-owned source span" in proof["reason"]


def test_out_of_range_model_span_never_becomes_proof(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(semantic, "request_message_budget", lambda *_: 4096)
    monkeypatch.setattr(
        semantic,
        "generate_fixed_template_value",
        lambda *a, **k: {
            "verdict": "supported",
            "evidence_start": 0,
            "evidence_end": 99,
        },
    )
    req = requirement()
    pool = pool_for("Players can trade resources to upgrade spacecraft.")
    result = semantic.review_requirement_sources(
        SimpleNamespace(), req, pool, requirement_candidate_trace(req, pool)
    )
    assert not result["complete"]
    assert result["accepted_proofs"] == []


def test_valid_evidence_span_is_not_artificially_capped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(semantic, "request_message_budget", lambda *_: 8192)
    body = "".join(
        f"Evidence section {index}: spacecraft trade upgrade behavior is documented. "
        + ("detail " * 80)
        + "\n"
        for index in range(6)
    )
    units = semantic._source_units(body)
    assert len(units) > 4

    def model(*args, **kwargs):
        if kwargs["tool_name"] == "assess_requirement_source":
            return {
                "verdict": "supported",
                "evidence_start": 0,
                "evidence_end": len(units) - 1,
            }
        return {"verdict": "supported"}

    monkeypatch.setattr(semantic, "generate_fixed_template_value", model)
    req = requirement()
    pool = pool_for(body)
    result = semantic.review_requirement_sources(
        SimpleNamespace(), req, pool, requirement_candidate_trace(req, pool)
    )

    assert result["complete"]
    assert result["accepted_proofs"][0]["excerpt"] == body
