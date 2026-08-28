from __future__ import annotations

from pathlib import Path

path = Path("minecraft_mod_ai/semantic_requirement_authority.py")
text = path.read_text(encoding="utf-8")
start = text.index("def _generate_approved_nodes(\n")
end = text.index("\n\ndef _build_catalog(", start)
replacement = '''def _generate_approved_nodes(
    prompt: str,
    router: Any,
    clauses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Approve authored meaning one host clause at a time.

    The model owns only the semantic description of the current authored clause.
    The host owns provenance, graph identity, clause identity, and all cross-clause
    references. A malformed response therefore invalidates only its own clause.
    """

    parameters = {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "capability_id": {"type": "string", "minLength": 3},
                        "source_quote": {"type": "string", "minLength": 1},
                        "semantic_statement": {"type": "string", "minLength": 1},
                        "given": {"type": "string", "minLength": 1},
                        "when": {"type": "string", "minLength": 1},
                        "then": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "capability_id", "source_quote", "semantic_statement",
                        "given", "when", "then",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }

    approved: list[dict[str, Any]] = []
    for ordinal, clause in enumerate(clauses):
        clause_index = int(clause["clause_index"])
        diagnostic: dict[str, Any] | None = None
        seen_failures: set[str] = set()
        accepted = False
        for attempt in range(3):
            request_payload = {
                "current_clause_index": clause_index,
                "current_clause": str(clause["text"]),
                "previous_clause_context": (
                    str(clauses[ordinal - 1]["text"]) if ordinal > 0 else ""
                ),
                "next_clause_context": (
                    str(clauses[ordinal + 1]["text"])
                    if ordinal + 1 < len(clauses)
                    else ""
                ),
                "repair_diagnostic": diagnostic,
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Interpret exactly the current authored clause. Do not add design choices, "
                        "implementation classes, APIs, dependencies, or requirements not stated by "
                        "that clause. Split only when the current clause independently states more "
                        "than one observable requirement. capability_id must be a meaningful lower-"
                        "case dotted semantic identifier, never an opaque hash. source_quote must be "
                        "the smallest unique verbatim substring of current_clause that grounds that "
                        "requirement. Adjacent clauses are context only and may not become new output "
                        "requirements. Return concrete Given/When/Then observable behavior."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False, sort_keys=True),
                },
            ]
            try:
                native = getattr(router, "generate_tool_decision", None)
                if callable(native):
                    payload = native(
                        "planner",
                        messages,
                        tool_name="approve_semantic_clause",
                        parameters=parameters,
                        description=(
                            "Return only semantic requirements grounded in the current authored clause."
                        ),
                    )
                else:
                    raw = router.generate_text(
                        "planner",
                        messages,
                        response_format="text",
                        response_schema=None,
                        enable_tools=False,
                    )
                    payload = _parse_json(raw)
            except Exception as exc:
                payload = {}
                diagnostic = _diagnostic(
                    "REQ_CLAUSE_MODEL_RESPONSE",
                    f"$.clauses[{clause_index}]",
                    f"{type(exc).__name__}: {exc}",
                    "one compact clause-local semantic payload",
                    f"clause:{clause_index}",
                )
            else:
                raw_requirements = (
                    payload.get("requirements") if isinstance(payload, Mapping) else None
                )
                if not isinstance(raw_requirements, list) or not raw_requirements:
                    diagnostic = _diagnostic(
                        "REQ_CLAUSE_SCHEMA",
                        f"$.clauses[{clause_index}].requirements",
                        raw_requirements,
                        "non-empty clause-local requirements array",
                        f"clause:{clause_index}",
                    )
                else:
                    raw_nodes: list[dict[str, Any]] = []
                    for item_index, item in enumerate(raw_requirements):
                        if not isinstance(item, Mapping):
                            raw_nodes.append({})
                            continue
                        raw_nodes.append(
                            {
                                "local_id": f"c{clause_index}_{item_index}",
                                "capability_id": item.get("capability_id"),
                                "provenance_role": "explicit",
                                "source_clause_index": 0,
                                "source_quote": item.get("source_quote"),
                                "semantic_statement": item.get("semantic_statement"),
                                "derived_from": [],
                                "depends_on": [],
                                "derivation_reason": "",
                                "observable_behavior": {
                                    "given": item.get("given"),
                                    "when": item.get("when"),
                                    "then": item.get("then"),
                                },
                            }
                        )
                    local_clause = dict(clause)
                    local_clause["clause_index"] = 0
                    local_nodes, validation_error = _validate_candidate(
                        {"requirements": raw_nodes},
                        prompt=prompt,
                        clauses=(local_clause,),
                    )
                    if local_nodes is not None:
                        for node in local_nodes:
                            node["source_clause_index"] = clause_index
                        approved.extend(local_nodes)
                        accepted = True
                        break
                    diagnostic = dict(validation_error or {})
                    diagnostic["repair_scope"] = f"clause:{clause_index}"

            failure_state = _sha256({"diagnostic": diagnostic, "candidate": payload})
            if failure_state in seen_failures:
                raise _evidence.EvidencePlanError(
                    "semantic clause approval reached a no-progress fixed point: "
                    + _canonical(diagnostic)
                )
            seen_failures.add(failure_state)
            if attempt == 2:
                raise _evidence.EvidencePlanError(
                    "semantic clause approval exhausted bounded repair attempts: "
                    + _canonical(diagnostic)
                )

        if not accepted:
            raise _evidence.EvidencePlanError(
                f"semantic clause {clause_index} was not approved"
            )

    merged_payload = {
        "requirements": [
            {
                "local_id": node["local_id"],
                "capability_id": node["capability_id"],
                "provenance_role": "explicit",
                "source_clause_index": node["source_clause_index"],
                "source_quote": node["source_quote"],
                "semantic_statement": node["semantic_statement"],
                "derived_from": [],
                "depends_on": [],
                "derivation_reason": "",
                "observable_behavior": dict(node["observable_behavior"]),
            }
            for node in approved
        ]
    }
    merged, final_error = _validate_candidate(
        merged_payload,
        prompt=prompt,
        clauses=clauses,
    )
    if merged is None:
        raise _evidence.EvidencePlanError(
            "host-merged semantic clauses violated the requirement contract: "
            + _canonical(final_error)
        )
    return merged
'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

Path("tests/test_semantic_requirement_authority.py").write_text('''from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.evidence_first_planning import EvidencePlanError
from minecraft_mod_ai.semantic_requirement_authority import (
    build_approved_requirement_catalog,
    validate_approved_requirement_catalog,
)


def _item(capability: str, quote: str) -> dict:
    return {
        "capability_id": capability,
        "source_quote": quote,
        "semantic_statement": capability.replace(".", " "),
        "given": "the feature precondition exists",
        "when": "the player performs the requested behavior",
        "then": "the requested outcome is observable",
    }


class TextRouter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra semantic call")
        value = self.responses.pop(0)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class NativeRouter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        return self.responses.pop(0)

    def generate_text(self, *args, **kwargs):
        raise AssertionError("native semantic authority must not use free-form JSON")


def test_semantic_authority_requests_each_authored_clause_separately():
    prompt = "Players gather crystals. Players travel to a new region."
    router = TextRouter([
        {"requirements": [_item("resource.gathering", "gather crystals")]},
        {"requirements": [_item("world.travel", "travel to a new region")]},
    ])
    catalog = build_approved_requirement_catalog(prompt, router)
    validate_approved_requirement_catalog(catalog, prompt=prompt)
    assert len(router.calls) == 2
    first = json.loads(router.calls[0]["messages"][1]["content"])
    second = json.loads(router.calls[1]["messages"][1]["content"])
    assert first["current_clause"] != second["current_clause"]
    assert router.calls[0]["kwargs"]["response_format"] == "text"
    assert router.calls[0]["kwargs"]["response_schema"] is None
    assert {r["capability"] for r in catalog["requirements"]} == {"resource.gathering", "world.travel"}
    assert all(r["provenance_role"] == "explicit" for r in catalog["requirements"])
    assert all(r["depends_on"] == [] for r in catalog["requirements"])


def test_one_bad_json_retries_only_the_failed_clause():
    prompt = "Players gather crystals. Players open a portal."
    router = TextRouter([
        {"requirements": [_item("resource.gathering", "gather crystals")]},
        "not json",
        {"requirements": [_item("progression.portal", "open a portal")]},
    ])
    catalog = build_approved_requirement_catalog(prompt, router)
    assert len(catalog["requirements"]) == 2
    assert len(router.calls) == 3
    retry = json.loads(router.calls[2]["messages"][1]["content"])
    assert retry["current_clause_index"] == 1
    assert retry["repair_diagnostic"]["error_code"] == "REQ_CLAUSE_MODEL_RESPONSE"
    assert retry["repair_diagnostic"]["repair_scope"] == "clause:1"


def test_native_tool_decision_is_preferred_for_clause_semantics():
    prompt = "Players collect fragments."
    router = NativeRouter([{"requirements": [_item("resource.collection", "collect fragments")]}])
    catalog = build_approved_requirement_catalog(prompt, router)
    assert len(catalog["requirements"]) == 1
    assert router.calls[0]["kwargs"]["tool_name"] == "approve_semantic_clause"
    props = router.calls[0]["kwargs"]["parameters"]["properties"]["requirements"]["items"]["properties"]
    assert "provenance_role" not in props
    assert "depends_on" not in props


def test_multiple_meanings_in_one_clause_need_distinct_grounding_quotes():
    prompt = "Players gather crystals, then trade crystals."
    router = TextRouter([{"requirements": [
        _item("resource.gathering", "gather crystals"),
        _item("economy.trade", "trade crystals"),
    ]}])
    catalog = build_approved_requirement_catalog(prompt, router)
    assert len(catalog["requirements"]) == 2
    assert {r["source_span"]["text"] for r in catalog["requirements"]} == {"gather crystals", "trade crystals"}


def test_opaque_capability_retries_clause_and_fails_closed_on_repeat():
    prompt = "Players discover a hidden mechanic."
    invalid = {"requirements": [_item("semantic_13ee7693e9ed", "discover a hidden mechanic")]}
    router = TextRouter([invalid, invalid])
    with pytest.raises(EvidencePlanError, match="semantic clause approval reached a no-progress fixed point"):
        build_approved_requirement_catalog(prompt, router)
    second = json.loads(router.calls[1]["messages"][1]["content"])
    assert second["repair_diagnostic"]["error_code"] == "REQ_CAPABILITY_ID"
    assert second["repair_diagnostic"]["repair_scope"] == "clause:0"


def test_model_cannot_promote_design_provenance_or_cross_clause_dependencies():
    prompt = "Players exchange collected items."
    supplied = _item("economy.exchange", "exchange collected items")
    supplied["provenance_role"] = "selected_design_alternative"
    supplied["depends_on"] = ["anything"]
    router = TextRouter([{"requirements": [supplied]}])
    catalog = build_approved_requirement_catalog(prompt, router)
    requirement = catalog["requirements"][0]
    assert requirement["provenance_role"] == "explicit"
    assert requirement["depends_on"] == []
''', encoding="utf-8")
