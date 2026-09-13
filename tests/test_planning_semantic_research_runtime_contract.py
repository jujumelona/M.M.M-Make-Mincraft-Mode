import json
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


def test_semantic_windows_use_actual_serialized_budget_not_half():
    req = requirement()
    budget = 4096
    body = "Spacecraft trade upgrade behavior is documented in detail. " * 300
    windows = list(semantic._semantic_windows(
        body,
        requirement_statement=req["statement"],
        obligation=req["acceptance"][0],
        source_id="modrinth:runtime",
        assessment_budget=budget,
        verification_budget=budget,
    ))

    assert "".join(window for _, _, window in windows) == body
    assert len(windows[0][2].encode("utf-8")) > budget // 2
    for _, _, window in windows:
        units = semantic._source_units(window)
        last = len(units) - 1
        assert semantic._message_bytes(semantic._assessment_messages(
            req["statement"], req["acceptance"][0], "modrinth:runtime", units
        )) <= budget
        assert semantic._message_bytes(semantic._verification_messages(
            req["statement"],
            req["acceptance"][0],
            "modrinth:runtime",
            units,
            last,
            last,
        )) <= budget


def test_verifier_context_sends_source_once_with_host_selected_range():
    req = requirement()
    body = "Unsupported features:\nPlayers can trade resources to upgrade spacecraft."
    units = semantic._source_units(body)
    messages = semantic._verification_messages(
        req["statement"],
        req["acceptance"][0],
        "modrinth:runtime",
        units,
        0,
        len(units) - 1,
    )
    payload = json.loads(messages[1]["content"])

    assert "source_quote" not in payload
    assert "source_window" not in payload
    assert "".join(unit["text"] for unit in payload["source_units"]) == body
    assert payload["evidence_start"] == 0
    assert payload["evidence_end"] == len(units) - 1


def test_structured_output_fixed_point_rejects_one_source_without_killing_planning(
    monkeypatch,
    tmp_path,
    capsys,
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
    events = [json.loads(line.split("ROOT CAUSE TRACE: ", 1)[1])
              for line in capsys.readouterr().err.splitlines()
              if line.startswith("ROOT CAUSE TRACE: ")]
    summary = next(event for event in events if event["event"] == "planning_semantic_research_summary")
    assert summary["details"]["assessment_verdict_counts"] == {"invalid_output": 1}
    assert summary["details"]["missing_obligations"] == [req["acceptance"][0]]
    assert summary["details"]["accepted_proof_count"] == 0


def test_host_reconstructs_exact_evidence_and_proof_text(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(semantic, "request_message_budget", lambda *_: 4096)
    calls = []

    def model(*args, **kwargs):
        calls.append(kwargs["tool_name"])
        if kwargs["tool_name"] == "assess_requirement_source":
            return {"verdict": "supported", "evidence_start": 0, "evidence_end": 0}
        payload = json.loads(args[2][1]["content"])
        assert "source_quote" not in payload
        assert "source_window" not in payload
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
    monkeypatch.setattr(semantic, "request_message_budget", lambda *_: 16384)
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
