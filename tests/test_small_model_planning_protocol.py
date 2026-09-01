from __future__ import annotations

from minecraft_mod_ai import authored_scope_research_contract as retrieval
from minecraft_mod_ai import planning_authority
from minecraft_mod_ai.forced_tool_execution_contract import _native_protocol_failure
from minecraft_mod_ai.model_adapters.qwen_tool_parser import ToolCallValidationError


class _SmallTextRouter:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []
        self.native_attempts = 0

    def generate_tool_decision(self, *_args: object, **_kwargs: object) -> object:
        self.native_attempts += 1
        raise ToolCallValidationError(
            "Qwen tool 'plan_requirement_retrieval' emitted invalid array value "
            "for parameter 'requirements'"
        )

    def generate_text(
        self,
        role: str,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.calls.append({"role": role, "messages": messages, "kwargs": kwargs})
        if not self.outputs:
            raise AssertionError("unexpected extra small-model turn")
        return self.outputs.pop(0)


def _semantic_leaf(capability: str, anchor: str, statement: str) -> str:
    return "\n".join(
        (
            "### requirement",
            f"capability_id: {capability}",
            f"source_anchor: {anchor}",
            f"semantic_statement: {statement}",
            "given: the authored state exists",
            "when: the player performs the authored action",
            "then: the authored outcome is observable",
        )
    )


def _requirement(
    requirement_id: str,
    capability: str,
    statement: str,
    source_text: str,
) -> dict[str, object]:
    return {
        "requirement_id": requirement_id,
        "capability": capability,
        "semantic_statement": statement,
        "source_span": {"text": source_text},
        "observable_behavior": {
            "given": "the authored state exists",
            "when": "the player performs the authored action",
            "then": "the authored outcome is observable",
        },
    }


def test_semantic_compilation_is_one_text_turn_per_host_clause() -> None:
    router = _SmallTextRouter(
        [
            _semantic_leaf("resource.gathering", "gather crystals", "gather crystals"),
            _semantic_leaf("economy.trade", "trade crystals", "trade crystals"),
        ]
    )
    clauses = [
        {"clause_index": 4, "text": "Players gather crystals."},
        {"clause_index": 9, "text": "Players trade crystals."},
    ]

    payload = planning_authority._call_semantic_compiler(router, clauses)

    assert [item["source_clause_index"] for item in payload["requirements"]] == [4, 9]
    assert router.native_attempts == 0
    assert len(router.calls) == 2
    first_user = router.calls[0]["messages"][1]["content"]
    second_user = router.calls[1]["messages"][1]["content"]
    assert "gather crystals" in first_user
    assert "trade crystals" not in first_user
    assert "trade crystals" in second_user
    assert "gather crystals" not in second_user
    for call in router.calls:
        kwargs = call["kwargs"]
        assert kwargs["response_format"] == "text"
        assert kwargs["response_schema"] is None
        assert kwargs["enable_tools"] is False


def test_host_owns_clause_index_even_if_model_attempts_to_emit_one() -> None:
    text = (
        "### Requirement 1\n"
        "source_clause_index: 999\n"
        "- capability: vehicle.launch\n"
        "- anchor: launch the ship\n"
        "- statement: launch the assembled ship\n"
        "- precondition: the ship is assembled\n"
        "- action: the player launches the ship\n"
        "- outcome: the ship enters flight"
    )

    payload = planning_authority._parse_semantic_markdown(
        text,
        source_clause_index=2,
    )

    assert payload["requirements"] == [
        {
            "capability_id": "vehicle.launch",
            "source_anchor": "launch the ship",
            "semantic_statement": "launch the assembled ship",
            "given": "the ship is assembled",
            "when": "the player launches the ship",
            "then": "the ship enters flight",
            "source_clause_index": 2,
        }
    ]


def test_retrieval_planning_uses_ordinals_and_one_requirement_query_turns() -> None:
    requirements = [
        _requirement(
            "req_host_owned_build",
            "vehicle.build",
            "build a spacecraft",
            "build the spacecraft",
        ),
        _requirement(
            "req_host_owned_launch",
            "vehicle.launch",
            "launch the completed spacecraft",
            "launch the spacecraft",
        ),
    ]
    router = _SmallTextRouter(
        [
            "edge: 1 -> 2",
            (
                "query: minecraft mod spacecraft assembly source\n"
                "query: fabric vehicle construction implementation\n"
                "query: github minecraft modular vehicle build"
            ),
            (
                "query: minecraft mod spacecraft launch source\n"
                "query: fabric vehicle launch mechanic implementation\n"
                "query: github minecraft vehicle flight transition"
            ),
        ]
    )

    payload = planning_authority._call_retrieval_planner(
        router,
        "Build a spacecraft, then launch it.",
        requirements,
    )
    normalized = retrieval._normalize_retrieval_plan(
        "Build a spacecraft, then launch it.",
        requirements,
        payload,
    )

    assert router.native_attempts == 0
    assert payload["_host_model_turns"] == 3
    assert normalized["req_host_owned_build"]["depends_on"] == []
    assert normalized["req_host_owned_launch"]["depends_on"] == ["req_host_owned_build"]
    assert all(len(item["search_queries"]) == 3 for item in normalized.values())
    rendered_messages = "\n".join(
        message["content"]
        for call in router.calls
        for message in call["messages"]
        if message["role"] == "user"
    )
    assert "req_host_owned_build" not in rendered_messages
    assert "req_host_owned_launch" not in rendered_messages
    assert "### 1" in router.calls[0]["messages"][1]["content"]
    assert "### 2" in router.calls[0]["messages"][1]["content"]
    for call in router.calls:
        kwargs = call["kwargs"]
        assert kwargs["response_format"] == "text"
        assert kwargs["response_schema"] is None
        assert kwargs["enable_tools"] is False


def test_full_authority_builds_structured_catalog_without_model_json() -> None:
    router = _SmallTextRouter(
        [
            _semantic_leaf(
                "resource.gathering",
                "gather crystals",
                "the player gathers crystals",
            ),
            (
                "query: minecraft crystal gathering mod source\n"
                "query: fabric collectible resource implementation\n"
                "query: github minecraft gathering mechanic"
            ),
        ]
    )

    catalog = planning_authority.build_authoritative_request_catalog(
        "Players gather crystals.",
        router,
    )

    assert router.native_attempts == 0
    assert len(router.calls) == 2
    assert len(catalog["requirements"]) == 1
    requirement = catalog["requirements"][0]
    assert requirement["capability"] == "resource.gathering"
    assert len(requirement["search_queries"]) == 3
    audit = catalog["semantic_audit"]
    assert audit["normal_model_turns"] == 2
    assert audit["retrieval_model_turns"] == 1
    assert audit["max_clauses_per_model_turn"] == 1
    assert audit["max_requirements_per_query_turn"] == 1
    assert audit["model_owned_requirement_ids"] is False
    assert audit["model_generated_planning_json"] is False


def test_query_parser_accepts_simple_numbered_or_bulleted_lines() -> None:
    queries = planning_authority._parse_query_lines(
        "1. minecraft mod crystal source\n- fabric collectible implementation"
    )

    assert queries == [
        "minecraft mod crystal source",
        "fabric collectible implementation",
    ]


def test_exact_logged_qwen_parser_failure_uses_bounded_argument_fallback() -> None:
    outer = RuntimeError("model backend failed")
    outer.cause = ToolCallValidationError(
        "Qwen tool 'plan_requirement_retrieval' emitted invalid array value "
        "for parameter 'requirements'"
    )

    assert _native_protocol_failure(outer) is True
