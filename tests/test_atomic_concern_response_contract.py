from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.atomic_concern_source import (
    AtomicConcernExecutor,
    END_MARKER,
    INITIALIZE_MARKER,
    MEMBERS_MARKER,
    parse_concern_content,
)
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import (
    _atomic_concern_output_token_ceiling,
    _call_coder,
)


def _response(members: str = "", initialize: str = "") -> str:
    return (
        f"{MEMBERS_MARKER}\n{members}\n"
        f"{INITIALIZE_MARKER}\n{initialize}\n"
        f"{END_MARKER}"
    )


def test_legacy_marker_parser_still_accepts_well_formed_response() -> None:
    members, initialize = parse_concern_content(
        _response("private static final int COST = 10;", ""),
        section="behavior_contract",
    )
    assert members == "private static final int COST = 10;"
    assert initialize == ""


def _executor(
    outputs: list[str],
    *,
    section: str = "behavior_contract",
    require_initialize: bool = False,
) -> AtomicConcernExecutor:
    remaining = list(outputs)

    def call_coder(_messages):
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    return AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section=section,
        concerns=(
            {
                "sequence": 0,
                "identifier": "id",
                "concern": "transitions",
                "task": "implement transitions",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=require_initialize,
        call_coder=call_coder,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )


@pytest.mark.parametrize(
    "output",
    [
        "private static final int COST = 10;",
        "```java\nprivate static final int COST = 10;\n```",
        f"{MEMBERS_MARKER}\nprivate static final int COST = 10;\n{END_MARKER}",
        "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_START\n"
        "private static final int COST = 10;\n"
        "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_END",
    ],
)
def test_executor_members_region_does_not_require_response_markers(output: str) -> None:
    result = _executor([output]).run()
    assert "private static final int COST = 10;" in result["source"]


def test_integration_members_and_initialize_are_generated_as_separate_regions() -> None:
    executor = _executor(
        [
            "private static void registerThing() {}",
            "registerThing();",
        ],
        section="integration",
        require_initialize=True,
    )
    result = executor.run()
    assert "private static void registerThing() {}" in result["source"]
    assert "registerThing();" in result["source"]


def test_inert_initialize_answer_becomes_empty_region() -> None:
    executor = _executor(
        [
            "private static void helper() {}",
            "// no initialization needed",
        ],
        section="integration",
        require_initialize=True,
    )
    result = executor.run()
    assert "private static void helper() {}" in result["source"]


@pytest.mark.parametrize(
    "bad",
    [
        "package example;",
        "import net.minecraft.Foo;",
        "public class Escape {}",
        "public static void initialize() {}",
    ],
)
def test_real_member_scope_escape_is_rejected_and_regenerated(bad: str) -> None:
    executor = _executor(
        [
            bad,
            "private static final int COST = 10;",
        ]
    )
    result = executor.run()
    assert "private static final int COST = 10;" in result["source"]


def test_identical_invalid_region_output_stops_on_semantic_no_progress() -> None:
    executor = _executor(
        [
            "package example;",
            "package example;",
        ]
    )
    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_RESPONSE_NO_PROGRESS",
    ):
        executor.run()


def test_marker_mentions_are_not_required_by_executor_contract() -> None:
    executor = _executor(
        ["private static final String NOTE = \"no response markers required\";"]
    )
    result = executor.run()
    assert "no response markers required" in result["source"]


def test_distinct_invalid_region_outputs_stop_at_bounded_attempt_limit(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "3")
    executor = _executor(
        [
            "package first;",
            "package second;",
            "package third;",
            "private static final int SHOULD_NOT_BE_REACHED = 1;",
        ]
    )
    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_RESPONSE_RETRY_EXHAUSTED",
    ):
        executor.run()


def test_atomic_concern_output_budget_defaults_to_bounded_page(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ATOMIC_CONCERN_OUTPUT_TOKENS", raising=False)
    assert _atomic_concern_output_token_ceiling() == 2048


def test_direct_coder_forwards_atomic_output_token_ceiling() -> None:
    captured: dict[str, object] = {}

    class _Router:
        def generate_text(self, role, messages, **kwargs):
            captured["role"] = role
            captured["messages"] = messages
            captured.update(kwargs)
            return "private static final int COST = 10;"

    result = _call_coder(
        _Router(),
        ({"role": "user", "content": "bounded concern"},),
        output_token_ceiling=1536,
    )

    assert result == "private static final int COST = 10;\n"
    assert captured["role"] == "coder"
    assert captured["enable_tools"] is False
    assert captured["tool_stage"] == "generation"
    assert captured["output_token_ceiling"] == 1536


def test_atomic_concern_output_budget_honors_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_OUTPUT_TOKENS", "1536")
    assert _atomic_concern_output_token_ceiling() == 1536


def test_private_nested_helper_types_are_valid_class_body_members() -> None:
    output = (
        "private enum ShipState { INITIAL, READY_TO_LAUNCH }\n"
        "private record CreditState(long credits) {}\n"
        "private static final class Snapshot {}"
    )
    result = _executor([output]).run()
    assert "private enum ShipState" in result["source"]
    assert "private record CreditState" in result["source"]
    assert "private static final class Snapshot" in result["source"]


@pytest.mark.parametrize(
    "bad",
    [
        "class PackageVisibleEscape {}",
        "protected static class ProtectedEscape {}",
        "} private static final int ESCAPED = 1; {",
    ],
)
def test_non_private_or_brace_escape_member_structure_is_rejected(bad: str) -> None:
    executor = _executor([bad, "private static final int COST = 10;"])
    result = executor.run()
    assert "private static final int COST = 10;" in result["source"]


def test_initialize_region_rejects_even_private_local_type_declarations() -> None:
    executor = _executor(
        [
            "private static void helper() {}",
            "private class LocalEscape {}",
            "helper();",
        ],
        section="integration",
        require_initialize=True,
    )
    result = executor.run()
    assert "helper();" in result["source"]
    assert "LocalEscape" not in result["source"]



@pytest.mark.parametrize(
    "bad",
    [
        (
            'The user wants me to implement the "variables" concern in the '
            '`AuthoredStateModel` class.\nprivate static final int COST = 10;'
        ),
        "I need to add members that define the state variables.\nprivate static final int COST = 10;",
        "- **credits**: owner: PlayerData, type: long\nprivate static final int COST = 10;",
        "1. Define the transition schema\nprivate static final int COST = 10;",
    ],
)
def test_reasoning_or_markdown_is_rejected_before_compile(bad: str) -> None:
    executor = _executor([bad, "private static final int COST = 10;"])
    result = executor.run()
    assert "private static final int COST = 10;" in result["source"]
    assert "The user" not in result["source"]
    assert "I need" not in result["source"]


def test_atomic_coder_can_force_non_thinking_transport() -> None:
    captured: dict[str, object] = {}

    class _Router:
        def generate_text(self, role, messages, **kwargs):
            captured["role"] = role
            captured["messages"] = messages
            captured.update(kwargs)
            return "private static final int COST = 10;"

    result = _call_coder(
        _Router(),
        ({"role": "user", "content": "atomic Java only"},),
        output_token_ceiling=1536,
        force_non_thinking=True,
    )

    assert result == "private static final int COST = 10;\n"
    assert captured["force_non_thinking"] is True
    assert captured["output_token_ceiling"] == 1536



def test_atomic_region_uses_required_structured_tool_not_free_text() -> None:
    captured: dict[str, object] = {}

    class _Router:
        def generate_text(self, *args, **kwargs):
            raise AssertionError("atomic region must never use free-text generation")

        def generate_tool_decision(
            self,
            role,
            messages,
            *,
            tool_name,
            parameters,
            description="",
            output_token_ceiling=None,
        ):
            captured["role"] = role
            captured["messages"] = messages
            captured["tool_name"] = tool_name
            captured["parameters"] = parameters
            captured["description"] = description
            captured["output_token_ceiling"] = output_token_ceiling
            return {"java": "private static final int COST = 10;"}

    result = _call_coder(
        _Router(),
        ({"role": "user", "content": "one atomic region"},),
        output_token_ceiling=1536,
        structured_java_region=True,
    )

    assert result == "private static final int COST = 10;"
    assert captured["role"] == "coder"
    assert captured["tool_name"] == "emit_java_region"
    assert captured["output_token_ceiling"] == 1536
    assert captured["parameters"] == {
        "type": "object",
        "properties": {
            "java": {
                "type": "string",
                "description": (
                    "Only the Java source text for the host-selected atomic region. "
                    "No prose, reasoning, Markdown, response markers, package/import lines, "
                    "or outer lifecycle/type declarations."
                ),
            }
        },
        "required": ["java"],
        "additionalProperties": False,
    }


def test_atomic_structured_tool_allows_intentional_empty_region() -> None:
    class _Router:
        def generate_tool_decision(self, role, messages, **kwargs):
            return {"java": ""}

    assert _call_coder(
        _Router(),
        ({"role": "user", "content": "no initialization required"},),
        structured_java_region=True,
    ) == ""


def test_atomic_structured_tool_rejects_extra_fields() -> None:
    class _Router:
        def generate_tool_decision(self, role, messages, **kwargs):
            return {"java": "private int x;", "reasoning": "I decided..."}

    with pytest.raises(
        CustomModuleGenerationError,
        match="emit_java_region must return exactly",
    ):
        _call_coder(
            _Router(),
            ({"role": "user", "content": "one region"},),
            structured_java_region=True,
        )



def _multi_executor(
    outputs: list[str],
    *,
    concerns: tuple[dict[str, object], ...],
    captured_messages: list[list[dict[str, str]]] | None = None,
) -> tuple[AtomicConcernExecutor, dict[str, int]]:
    remaining = list(outputs)
    compile_calls = {"count": 0}

    def call_coder(messages):
        if captured_messages is not None:
            captured_messages.append(list(messages))
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def compile_java(_root):
        compile_calls["count"] += 1
        return SimpleNamespace(status="PASS")

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="state_model",
        concerns=concerns,
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )
    return executor, compile_calls



def test_sibling_symbol_collision_is_rehomed_before_compile() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "transactions", "task": "transaction state", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "concurrency", "task": "concurrency guard", "rules": []},
    )
    executor, compile_calls = _multi_executor(
        [
            (
                "private static boolean isTransactionInFlight;\n"
                "private static void beginTransaction() {}\n"
                "private static void endTransaction() {}"
            ),
            (
                "private static boolean isTransactionInFlight;\n"
                "private static void beginTransaction() {}\n"
                "private static void endTransaction() {}"
            ),
        ],
        concerns=concerns,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert result["repair_count"] == 0
    assert result["source"].count("isTransactionInFlight") == 1
    assert result["source"].count("beginTransaction()") == 1
    assert result["source"].count("endTransaction()") == 1


def test_concern_authority_slices_out_sibling_requirements() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _concern_authority

    obligation = json.dumps(
        {
            "instruction": json.dumps(
                {
                    "concern": "variables",
                    "section_instruction": (
                        "Implement domain state containers, invariants, and transition helpers only."
                    ),
                }
            ),
            "source_requirements": {
                "R18": "## state_model",
                "R19": "- variables: name owner type unit default domain",
                "R20": "    - player_credits: owner_player double default 0.0",
                "R21": "    - ship_hull_integrity: owner_server float default 100.0",
                "R25": "- transitions: from_state trigger guard mutation to_state",
                "R26": "    - locked_to_unlocked resource_qualified level_threshold",
                "R29": "- invariants: condition enforcement player_credits < min_cost",
            },
        }
    )
    authority = _concern_authority(
        {
            "task_id": "ir_authoredstatemodel",
            "semantic_outcome": "broad state model responsibility",
            "implementation_obligations": [obligation],
        },
        {"concern": "variables"},
    )

    assert authority == {
        "task_id": "ir_authoredstatemodel",
        "concern": "variables",
        "source_requirements": {
            "R18": "## state_model",
            "R19": "- variables: name owner type unit default domain",
            "R20": "    - player_credits: owner_player double default 0.0",
            "R21": "    - ship_hull_integrity: owner_server float default 100.0",
        },
    }


def test_atomic_prompt_does_not_delegate_symbol_bookkeeping_to_model() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "variables", "task": "variables", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "invariants", "task": "invariants", "rules": []},
    )
    captured: list[list[dict[str, str]]] = []
    executor, _compile_calls = _multi_executor(
        [
            "private static int playerCredits;",
            "private static final int INVARIANT_PLAYER_CREDITS_MIN_COST = 0;",
        ],
        concerns=concerns,
        captured_messages=captured,
    )

    executor.run()

    payload = __import__("json").loads(captured[1][-1]["content"])
    assert "existing_symbol_owners" not in payload["scope"]
    assert payload["scope"]["sibling_concerns_out_of_scope"] == ["variables"]
    assert "bookkeeping" in payload["scope"]["scope_rule"]

def test_method_overloads_with_different_parameter_types_do_not_collide() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "first", "task": "first overload", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "second", "task": "second overload", "rules": []},
    )
    executor, compile_calls = _multi_executor(
        [
            "private static void update(int value) {}",
            "private static void update(String value) {}",
        ],
        concerns=concerns,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert "update(int value)" in result["source"]
    assert "update(String value)" in result["source"]



def test_static_initializer_does_not_create_fake_symbol_owner() -> None:
    from minecraft_mod_ai.atomic_concern_source import _member_declaration_symbols

    assert _member_declaration_symbols("static { initializeSomething(); }") == {}


def test_generic_method_erasure_collision_is_rehomed_by_host() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "first", "task": "first generic method", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "second", "task": "second generic method", "rules": []},
    )
    executor, compile_calls = _multi_executor(
        [
            "private static void update(java.util.List<String> value) {}",
            "private static void update(java.util.List<Integer> value) {}",
            "",
        ],
        concerns=concerns,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert result["source"].count("private static void update(") == 1



def test_state_model_later_concerns_rehome_variables_overreach() -> None:
    concerns = (
        {"sequence": 0, "identifier": "v", "concern": "variables", "task": "variables only", "rules": []},
        {"sequence": 1, "identifier": "t", "concern": "transitions", "task": "transitions only", "rules": []},
        {"sequence": 2, "identifier": "i", "concern": "invariants", "task": "invariants only", "rules": []},
    )
    executor, compile_calls = _multi_executor(
        [
            (
                "private static double playerCredits;\n"
                "private static final String FROM_STATE_LOCKED = \"locked\";\n"
                "private static final int INVARIANT_PLAYER_CREDITS_MIN_COST = 0;"
            ),
            "private static final String FROM_STATE_LOCKED = \"locked\";",
            "private static final int INVARIANT_PLAYER_CREDITS_MIN_COST = 0;",
        ],
        concerns=concerns,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert result["repair_count"] == 0
    assert result["source"].count("FROM_STATE_LOCKED") == 1
    assert result["source"].count("INVARIANT_PLAYER_CREDITS_MIN_COST") == 1
    assert "playerCredits" in result["source"]
