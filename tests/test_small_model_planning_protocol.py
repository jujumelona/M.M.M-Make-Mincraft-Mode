from __future__ import annotations

import pytest

from minecraft_mod_ai import authored_scope_research_contract as retrieval
from minecraft_mod_ai import planning_authority
from minecraft_mod_ai import semantic_requirement_authority as semantic
from minecraft_mod_ai.forced_tool_execution_contract import _native_protocol_failure
from minecraft_mod_ai.model_adapters.qwen_tool_parser import ToolCallValidationError


class _StructuredRouter:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = dict(responses)
        self.calls: list[dict[str, object]] = []
        self.text_calls = 0

    def generate_tool_decision(
        self,
        role: str,
        messages: object,
        *,
        tool_name: str,
        parameters: object,
        description: str = "",
    ) -> object:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        if tool_name not in self.responses:
            raise AssertionError(f"unexpected structured turn: {tool_name}")
        return self.responses[tool_name]

    def generate_text(self, *_args: object, **_kwargs: object) -> str:
        self.text_calls += 1
        raise AssertionError("small-model planning must not fall back to free-form text")


def _semantic_payload() -> dict[str, object]:
    return {
        "requirements": [
            {
                "source_clause_index": 0,
                "capability_id": "resource.gathering",
                "source_anchor": "gather crystals",
                "semantic_statement": "the player can gather crystals",
                "given": "crystals are available",
                "when": "the player gathers crystals",
                "then": "the player obtains crystals",
            },
            {
                "source_clause_index": 1,
                "capability_id": "economy.trade",
                "source_anchor": "trade crystals",
                "semantic_statement": "the player can trade crystals",
                "given": "the player has crystals",
                "when": "the player trades crystals",
                "then": "the trade completes",
            },
        ]
    }


def test_semantic_compiler_is_one_structured_batch_for_many_clauses() -> None:
    router = _StructuredRouter(
        {"compile_semantic_requirements": _semantic_payload()}
    )
    clauses = [
        {"clause_index": 0, "text": "Players gather crystals."},
        {"clause_index": 1, "text": "Players trade crystals."},
    ]

    payload = planning_authority._call_semantic_compiler(router, clauses)

    assert payload == _semantic_payload()
    assert [call["tool_name"] for call in router.calls] == [
        "compile_semantic_requirements"
    ]
    assert router.text_calls == 0
    parameters = router.calls[0]["parameters"]
    assert isinstance(parameters, dict)
    source_index = parameters["properties"]["requirements"]["items"]["properties"][
        "source_clause_index"
    ]
    assert source_index["minimum"] == 0
    assert source_index["maximum"] == 1


def test_many_semantic_leaves_share_one_model_turn() -> None:
    payload = {
        "requirements": [
            {
                "source_clause_index": 0,
                "capability_id": "resource.gathering",
                "source_anchor": "gather crystals",
                "semantic_statement": "gather crystals",
                "given": "crystals exist",
                "when": "the player gathers them",
                "then": "crystals are obtained",
            },
            {
                "source_clause_index": 0,
                "capability_id": "economy.trade",
                "source_anchor": "trade them",
                "semantic_statement": "trade crystals",
                "given": "crystals are held",
                "when": "the player trades them",
                "then": "the trade completes",
            },
            {
                "source_clause_index": 0,
                "capability_id": "vehicle.launch",
                "source_anchor": "launch a spacecraft",
                "semantic_statement": "launch a spacecraft",
                "given": "a spacecraft exists",
                "when": "the player launches it",
                "then": "the spacecraft enters flight",
            },
        ]
    }
    router = _StructuredRouter({"compile_semantic_requirements": payload})

    result = semantic._call_semantic_model(
        router,
        [
            {
                "clause_index": 0,
                "text": "Players gather crystals, trade them, and launch a spacecraft.",
            }
        ],
    )

    assert result == payload
    assert len(router.calls) == 1
    assert router.calls[0]["tool_name"] == "compile_semantic_requirements"
    assert router.text_calls == 0


def test_retrieval_planner_is_host_deterministic_for_many_requirements() -> None:
    requirements = [
        {
            "requirement_id": "req-build",
            "capability": "vehicle.spacecraft.build",
            "depends_on": [],
        },
        {
            "requirement_id": "req-launch",
            "capability": "vehicle.spacecraft.launch",
            "depends_on": ["req-build"],
        },
    ]
    router = _StructuredRouter({})

    raw = planning_authority._call_retrieval_planner(
        router,
        "Build a spacecraft, then launch it.",
        requirements,
    )
    normalized = retrieval._normalize_retrieval_plan(
        "Build a spacecraft, then launch it.",
        requirements,
        raw,
    )

    assert router.calls == []
    assert router.text_calls == 0
    assert normalized["req-build"]["depends_on"] == []
    assert normalized["req-launch"]["depends_on"] == ["req-build"]
    assert len(normalized["req-build"]["search_queries"]) >= 2
    assert len(normalized["req-launch"]["search_queries"]) >= 2
    assert all(
        query.casefold().startswith("minecraft ")
        for item in normalized.values()
        for query in item["search_queries"]
    )


def test_retrieval_schema_requires_exact_requirement_cardinality() -> None:
    schema = retrieval._retrieval_plan_schema(["req-a", "req-b", "req-c"])
    requirements = schema["properties"]["requirements"]

    assert requirements["minItems"] == 3
    assert requirements["maxItems"] == 3
    requirement_id = requirements["items"]["properties"]["requirement_id"]
    assert requirement_id["enum"] == ["req-a", "req-b", "req-c"]


def test_retrieval_normalizer_rejects_missing_requirement() -> None:
    requirements = [
        {"requirement_id": "req-a"},
        {"requirement_id": "req-b"},
    ]
    payload = {
        "requirements": [
            {
                "requirement_id": "req-a",
                "depends_on": [],
                "search_queries": [
                    "minecraft crystal source",
                    "minecraft crystal implementation",
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="omitted requirements"):
        retrieval._normalize_retrieval_plan("request", requirements, payload)


def test_retrieval_normalizer_rejects_dependency_cycles() -> None:
    requirements = [
        {"requirement_id": "req-a"},
        {"requirement_id": "req-b"},
    ]
    payload = {
        "requirements": [
            {
                "requirement_id": "req-a",
                "depends_on": ["req-b"],
                "search_queries": [
                    "minecraft crystal source",
                    "minecraft crystal implementation",
                ],
            },
            {
                "requirement_id": "req-b",
                "depends_on": ["req-a"],
                "search_queries": [
                    "minecraft trade source",
                    "minecraft trade implementation",
                ],
            },
        ]
    }

    with pytest.raises(ValueError, match="cycle"):
        retrieval._normalize_retrieval_plan("request", requirements, payload)


def test_semantic_schema_keeps_host_clause_index_bounded() -> None:
    schema = semantic._semantic_schema(7)
    source_index = schema["properties"]["requirements"]["items"]["properties"][
        "source_clause_index"
    ]

    assert source_index == {
        "type": "integer",
        "minimum": 0,
        "maximum": 7,
    }


def test_exact_logged_qwen_parser_failure_uses_bounded_argument_fallback() -> None:
    outer = RuntimeError("model backend failed")
    outer.cause = ToolCallValidationError(
        "Qwen tool 'plan_requirement_retrieval' emitted invalid array value "
        "for parameter 'requirements'"
    )

    assert _native_protocol_failure(outer) is True
