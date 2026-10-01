from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.atomic_concern_source import (
    END_MARKER,
    INITIALIZE_MARKER,
    MEMBERS_MARKER,
    AtomicConcernExecutor,
    _messages,
    _strip_host_orchestrated_dependency_lifecycle_calls,
    parse_concern_content,
)
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import (
    _call_coder,
)
from minecraft_mod_ai.implementation_ir import OutputBudgetExhausted


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
    section: str = "algorithm",
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
        ("// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_START\n"
         "private static final int COST = 10;\n"
         "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_END"),
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
def test_real_member_scope_escape_fails_first_pass_without_regeneration(bad: str) -> None:
    executor = _executor([bad])
    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID",
    ):
        executor.run()


def test_invalid_region_output_is_terminal_on_the_first_production_decode() -> None:
    executor = _executor(["package example;"])
    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID",
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
def test_non_private_or_brace_escape_member_structure_is_rejected(bad: str, monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "2")
    executor = _executor([bad, "private static final int COST = 10;"])
    result = executor.run()
    assert "private static final int COST = 10;" in result["source"]


def test_initialize_region_rejects_even_private_local_type_declarations(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "2")
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
    executor = _executor([bad])
    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID",
    ):
        executor.run()


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
            return next(responses)

    responses = iter([
        {"part": "fields"},
        {"type": "int", "name": "COST", "initializer": "10"},
        {"part": "done"},
        {"part": "done"},
    ])

    result = _call_coder(
        _Router(),
        (
            {"role": "system", "content": "structured only"},
            {"role": "user", "content": '{"response_region":"members"}'},
        ),
        output_token_ceiling=1536,
        structured_java_region=True,
    )

    assert result == "private static int COST = 10;"
    assert captured["role"] == "coder"
    assert captured["tool_name"] == "emit_java_part"
    assert captured["output_token_ceiling"] == 1536


def test_atomic_structured_tool_allows_intentional_empty_region() -> None:
    class _Router:
        def generate_tool_decision(self, role, messages, **kwargs):
            return {"part": "done"}

    assert _call_coder(
        _Router(),
        (
            {"role": "system", "content": "structured only"},
            {"role": "user", "content": '{"response_region":"members"}'},
        ),
        structured_java_region=True,
    ) == ""


def test_atomic_structured_tool_rejects_extra_fields() -> None:
    class _Router:
        def generate_tool_decision(self, role, messages, **kwargs):
            return {"part": "done", "reasoning": "I decided..."}

    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_JAVA_ASSEMBLY_INVALID",
    ):
        _call_coder(
            _Router(),
            (
                {"role": "system", "content": "structured only"},
                {"role": "user", "content": '{"response_region":"members"}'},
            ),
            structured_java_region=True,
        )

def _multi_executor(
    outputs: list[str],
    *,
    concerns: tuple[dict[str, object], ...],
    captured_messages: list[list[dict[str, str]]] | None = None,
    section: str = "algorithm",
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
        section=section,
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


def test_sibling_symbol_collision_cannot_replace_accepted_sibling() -> None:
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
    assert result["source"].count("private static boolean isTransactionInFlight;") == 1
    assert result["source"].count("private static void beginTransaction() {}") == 1
    assert result["source"].count("private static void endTransaction() {}") == 1
    assert executor.state["concurrency"][0] == ""

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
        "structured_records": [],
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


def test_generic_method_erasure_collision_preserves_original_owner() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "first", "task": "first generic method", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "second", "task": "second generic method", "rules": []},
    )
    executor, compile_calls = _multi_executor(
        [
            "private static void update(java.util.List<String> value) {}",
            "private static void update(java.util.List<Integer> value) {}",
        ],
        concerns=concerns,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert result["source"].count("private static void update(") == 1
    assert "java.util.List<String> value" in result["source"]
    assert "java.util.List<Integer> value" not in result["source"]
    assert executor.state["second"][0] == ""

def test_state_model_later_concerns_cannot_rehome_variables() -> None:
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
    assert result["source"].count("private static double playerCredits;") == 1
    assert result["source"].count("private static final String FROM_STATE_LOCKED = \"locked\";") == 1
    assert result["source"].count("private static final int INVARIANT_PLAYER_CREDITS_MIN_COST = 0;") == 1
    assert executor.state["transitions"][0] == ""
    assert executor.state["invariants"][0] == ""

def test_private_static_initializer_is_normalized_before_compile(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "2")
    executor = _executor(
        [
            (
                "private static {\n"
                "    java.lang.System.setProperty(\"mmm.test\", \"1\");\n"
                "}"
            ),
            "private static final int COST = 10;",
        ]
    )

    result = executor.run()

    assert "private static {" not in result["source"]
    assert 'java.lang.System.setProperty("mmm.test", "1");' not in result["source"]
    assert "private static final int COST = 10;" in result["source"]
    assert result["repair_count"] == 0


def test_visibility_instance_initializer_is_rejected_before_compile(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "2")
    executor = _executor(
        [
            "private { initializeSomething(); }",
            "private static final int COST = 10;",
        ]
    )

    result = executor.run()

    assert "private {" not in result["source"]
    assert "private static final int COST = 10;" in result["source"]


def test_structured_tool_declares_domain_type_and_field_together() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "records": [
                {
                    "modifiers": ["private"],
                    "name": "StateVariable",
                    "components": [{"type": "String", "name": "name"}],
                    "methods": [],
                }
            ],
            "enums": [],
            "classes": [],
            "fields": [
                {
                    "modifiers": ["private", "static", "final"],
                    "type": "StateVariable",
                    "name": "CREDITS",
                    "initializer": 'new StateVariable("credits")',
                }
            ],
            "methods": [],
            "static_initializers": [],
        },
        response_region="members",
    )

    assert "private record StateVariable(String name) {}" in rendered
    assert "private static final StateVariable CREDITS" in rendered


def test_structured_renderer_canonicalizes_lock_types_before_compilation() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "records": [],
            "enums": [],
            "classes": [],
            "fields": [
                {
                    "modifiers": ["private", "static", "final"],
                    "type": "Lock",
                    "name": "LOCK",
                    "initializer": "new ReentrantLock()",
                },
                {
                    "modifiers": ["private", "static", "final"],
                    "type": "java.util.concurrent.Lock",
                    "name": "LOCK2",
                    "initializer": "new java.util.concurrent.ReentrantLock()",
                },
            ],
            "methods": [],
            "static_initializers": [],
        },
        response_region="members",
    )

    assert "java.util.concurrent.locks.Lock LOCK" in rendered
    assert "new java.util.concurrent.locks.ReentrantLock()" in rendered
    assert "java.util.concurrent.Lock" not in rendered
    assert "java.util.concurrent.ReentrantLock" not in rendered


def test_structured_generation_rejects_blank_final_before_compilation() -> None:
    from minecraft_mod_ai.custom_module_generator import _validate_atomic_java_decision

    with pytest.raises(CustomModuleGenerationError, match="must be initialized at its declaration"):
        _validate_atomic_java_decision(
            {
                "fields": [
                    {
                        "modifiers": ["private", "static", "final"],
                        "type": "boolean",
                        "name": "NO_FLUID_STORAGE_IN_ZERO_G",
                    }
                ],
                "methods": [],
            },
            payload={},
            response_region="members",
        )


def test_structured_generation_rejects_final_rebinding_before_compilation() -> None:
    from minecraft_mod_ai.custom_module_generator import _validate_atomic_java_decision

    with pytest.raises(CustomModuleGenerationError, match="is reassigned"):
        _validate_atomic_java_decision(
            {
                "fields": [
                    {
                        "modifiers": ["private", "static", "final"],
                        "type": "Map<String, Object>",
                        "name": "TRANSFER_CACHE",
                        "initializer": "new HashMap<>()",
                    }
                ],
                "methods": [
                    {
                        "return_type": "void",
                        "name": "resetTransferCache",
                        "body": ["TRANSFER_CACHE = Collections.emptyMap();"],
                    }
                ],
            },
            payload={},
            response_region="members",
        )


def test_structured_generation_rejects_object_return_into_narrow_map_before_compilation() -> None:
    from minecraft_mod_ai.custom_module_generator import _validate_atomic_java_decision

    payload = {
        "dependency_api": [
            {
                "symbol": "AuthoredStateModel",
                "public_api": [
                    "public static Object getState(String key)"
                ],
            }
        ]
    }

    with pytest.raises(CustomModuleGenerationError, match="authoritative Object-valued call"):
        _validate_atomic_java_decision(
            {
                "fields": [],
                "methods": [
                    {
                        "return_type": "Map<String, Object>",
                        "name": "shipConfig",
                        "body": ['return AuthoredStateModel.getState("shipConfig");'],
                    }
                ],
            },
            payload=payload,
            response_region="members",
        )


def test_structured_tool_host_qualifies_common_jdk_types() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "records": [],
            "enums": [],
            "classes": [],
            "fields": [
                {
                    "modifiers": ["private", "static", "final"],
                    "type": "List<String>",
                    "name": "VALUES",
                    "initializer": "List.of()",
                }
            ],
            "methods": [],
            "static_initializers": [],
        },
        response_region="members",
    )

    assert "java.util.List<String>" in rendered
    assert "java.util.List.of()" in rendered

def test_previous_concern_nested_type_is_available_to_next_concern() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "schema", "task": "schema", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "values", "task": "values", "rules": []},
    )
    executor, compile_calls = _multi_executor(
        [
            "private record SharedValue(int value) {}",
            "private static final SharedValue VALUE = new SharedValue(1);",
        ],
        concerns=concerns,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert "SharedValue VALUE" in result["source"]


def test_equal_error_count_with_changed_diagnostics_can_keep_repairing(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "2")
    remaining = [
        "private static int VALUE = missingA();",
        "private static int VALUE = missingB();",
        "private static int VALUE = 1;",
    ]
    state: dict[str, str] = {"source": ""}
    compile_calls = {"count": 0}

    def call_coder(_messages):
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def write_source(_path, source):
        state["source"] = source

    def compile_java(_root):
        compile_calls["count"] += 1
        if compile_calls["count"] >= 3:
            return SimpleNamespace(status="PASS", log="")
        rows = state["source"].splitlines()
        marker = next(
            index for index, row in enumerate(rows)
            if "MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_START" in row
        )
        line_number = marker + 2
        missing = "missingA" if compile_calls["count"] == 1 else "missingB"
        return SimpleNamespace(
            status="FAIL",
            log=(
                f"/tmp/Test.java:{line_number}: error: cannot find symbol\n"
                f"  symbol: method {missing}()\n"
            ),
        )

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("/tmp/Test.java"),
        relative="/tmp/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
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
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda report: getattr(report, "log", ""),
        write_source=write_source,
    )

    result = executor.run()

    assert compile_calls["count"] == 3
    assert result["repair_count"] == 2
    assert "VALUE = 1" in result["source"]


def test_same_compiler_diagnostic_retries_when_concern_source_changed(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "2")
    captured: list[list[dict[str, str]]] = []
    remaining = [
        "private static int VALUE = missingA();",
        "private static int VALUE = missingB();",
        "private static int VALUE = 1;",
    ]
    state: dict[str, str] = {"source": ""}
    compile_calls = {"count": 0}

    def call_coder(messages):
        captured.append(list(messages))
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def write_source(_path, source):
        state["source"] = source

    def compile_java(_root):
        compile_calls["count"] += 1
        if compile_calls["count"] >= 3:
            return SimpleNamespace(status="PASS", log="")
        rows = state["source"].splitlines()
        line_number = next(
            index for index, row in enumerate(rows, start=1)
            if "private static int VALUE" in row
        )
        failing_source = rows[line_number - 1]
        return SimpleNamespace(
            status="FAIL",
            log=(
                f"/tmp/Test.java:{line_number}: error: cannot find symbol\n"
                f"{failing_source}\n"
                "                           ^\n"
                "  symbol: method missing()\n"
                "  location: class Test\n"
            ),
        )

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("/tmp/Test.java"),
        relative="/tmp/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
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
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda report: getattr(report, "log", ""),
        write_source=write_source,
    )

    result = executor.run()

    assert result["repair_count"] == 2
    assert compile_calls["count"] == 3
    first_repair = __import__("json").loads(captured[1][-1]["content"])
    second_repair = __import__("json").loads(captured[2][-1]["content"])
    assert "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE" in first_repair["repair_failure"]
    assert ">>" in first_repair["repair_failure"]
    assert "private static int VALUE = missingA();" in first_repair["repair_failure"]
    assert "private static int VALUE = missingB();" in second_repair["repair_failure"]
    assert first_repair["current_selected_region_source"] == "private static int VALUE = missingA();"
    assert second_repair["current_selected_region_source"] == "private static int VALUE = missingB();"


def test_long_compiler_log_keeps_repair_checklist_and_contract() -> None:
    from minecraft_mod_ai.atomic_concern_source import _compiler_repair_context

    source = (
        "package example;\n"
        "public final class Test {\n"
        "// MMM_ATOMIC_CONCERN_CONCURRENCY_HAZARDS_MEMBERS_START\n"
        "private static final java.util.concurrent.Lock shipConfigLock = "
        "new java.util.concurrent.ReentrantLock();\n"
        "// MMM_ATOMIC_CONCERN_CONCURRENCY_HAZARDS_MEMBERS_END\n"
        "}\n"
    )
    block = (
        "/tmp/Test.java:4: error: cannot find symbol\n"
        "private static final java.util.concurrent.Lock shipConfigLock = "
        "new java.util.concurrent.ReentrantLock();\n"
        "^\n"
        "  symbol:   class Lock\n"
        "  location: package java.util.concurrent\n"
        "/tmp/Test.java:4: error: cannot find symbol\n"
        "private static final java.util.concurrent.Lock shipConfigLock = "
        "new java.util.concurrent.ReentrantLock();\n"
        "^\n"
        "  symbol:   class ReentrantLock\n"
        "  location: package java.util.concurrent\n"
        "Note: Test.java uses unchecked or unsafe operations.\n"
    )
    log = block * 80

    repair = _compiler_repair_context(
        log,
        source=source,
        relative="Test.java",
        concern="concurrency_hazards",
    )

    assert len(repair) <= 12064
    assert "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE" in repair
    assert "CURRENT COMPILED SOURCE AROUND THE REPORTED LINES" in repair
    assert "COMPILER-DERIVED REPAIR CHECKLIST" in repair
    assert "JDK_LOCK_PACKAGE" in repair
    assert "UNCHECKED_TYPES" in repair
    assert "REPAIR CONTRACT:" in repair
    assert "Return only the complete corrected selected Java region" in repair


def test_logged_java_failure_families_repair_through_real_compiler_feedback(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "4")
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "4")
    captured: list[list[dict[str, str]]] = []
    remaining = [
        "private static final boolean NO_FLUID_STORAGE_IN_ZERO_G;",
        (
            "private static final java.util.Map<String, Object> TRANSFER_CACHE = "
            "new java.util.HashMap<>();\n"
            "private static void resetTransferCache() { "
            "TRANSFER_CACHE = java.util.Collections.emptyMap(); }"
        ),
        (
            "private static Object readState() { return null; }\n"
            "private static java.util.Map<String, Object> shipConfig() { "
            "return readState(); }"
        ),
        (
            "private static final java.util.concurrent.Lock shipConfigLock = "
            "new java.util.concurrent.ReentrantLock();"
        ),
        (
            "private static final java.util.concurrent.locks.Lock shipConfigLock = "
            "new java.util.concurrent.locks.ReentrantLock();"
        ),
    ]
    state: dict[str, str] = {"source": ""}
    compile_calls = {"count": 0}

    def call_coder(messages):
        captured.append(list(messages))
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def write_source(_path, source):
        state["source"] = source

    def error_for(source: str):
        rows = source.splitlines()

        def at(fragment: str) -> tuple[int, str]:
            number = next(
                index for index, row in enumerate(rows, start=1)
                if fragment in row
            )
            return number, rows[number - 1]

        if "private static final boolean NO_FLUID_STORAGE_IN_ZERO_G;" in source:
            line, row = at("NO_FLUID_STORAGE_IN_ZERO_G")
            return (
                line,
                row,
                "variable NO_FLUID_STORAGE_IN_ZERO_G might not have been initialized",
                "",
            )
        if "TRANSFER_CACHE = java.util.Collections.emptyMap();" in source:
            line, row = at("TRANSFER_CACHE = java.util.Collections.emptyMap()")
            return (
                line,
                row,
                "cannot assign a value to static final variable TRANSFER_CACHE",
                "",
            )
        if "return readState();" in source:
            line, row = at("return readState();")
            return (
                line,
                row,
                "incompatible types: Object cannot be converted to Map<String,Object>",
                "",
            )
        if "java.util.concurrent.Lock" in source:
            line, row = at("java.util.concurrent.Lock")
            return (
                line,
                row,
                "cannot find symbol",
                "  symbol:   class Lock\n"
                "  location: package java.util.concurrent\n"
                f"/tmp/Test.java:{line}: error: cannot find symbol\n"
                f"{row}\n"
                "^\n"
                "  symbol:   class ReentrantLock\n"
                "  location: package java.util.concurrent\n",
            )
        return None

    def compile_java(_root):
        compile_calls["count"] += 1
        error = error_for(state["source"])
        if error is None:
            return SimpleNamespace(status="PASS", log="")
        line, row, message, tail = error
        return SimpleNamespace(
            status="FAIL",
            log=(
                f"/tmp/Test.java:{line}: error: {message}\n"
                f"{row}\n"
                "^\n"
                f"{tail}"
            ),
        )

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("/tmp/Test.java"),
        relative="/tmp/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
        concerns=(
            {
                "sequence": 0,
                "identifier": "id",
                "concern": "concurrency_hazards",
                "task": "implement compiler-safe concurrency state",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda report: getattr(report, "log", ""),
        write_source=write_source,
    )

    result = executor.run()

    assert compile_calls["count"] == 1
    assert result["repair_count"] == 0
    assert "java.util.concurrent.locks.Lock" in result["source"]
    assert "java.util.concurrent.locks.ReentrantLock" in result["source"]
    assert len(captured) == 4

    validation_failures = [
        __import__("json").loads(messages[-1]["content"])["repair_failure"]
        for messages in captured[1:]
    ]
    assert "blank final field" in validation_failures[0]
    assert "reassigns final field" in validation_failures[1]
    assert "Object-valued local call readState" in validation_failures[2]
    assert all(
        "HOST REGION VALIDATION FAILED BEFORE COMPILATION" in item
        for item in validation_failures
    )
    assert all(
        "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE" not in item
        for item in validation_failures
    )
    assert remaining == [
        (
            "private static final java.util.concurrent.locks.Lock shipConfigLock = "
            "new java.util.concurrent.locks.ReentrantLock();"
        )
    ]

def test_logged_java_failure_families_repair_with_actual_javac(tmp_path, monkeypatch) -> None:
    import shutil
    import subprocess

    if shutil.which("javac") is None:
        pytest.skip("javac is required for the compiler-feedback integration regression")

    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "4")
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "4")
    captured: list[list[dict[str, str]]] = []
    remaining = [
        "private static final boolean NO_FLUID_STORAGE_IN_ZERO_G;",
        (
            "private static final java.util.Map<String, Object> TRANSFER_CACHE = "
            "new java.util.HashMap<>();\n"
            "private static void resetTransferCache() { "
            "TRANSFER_CACHE = java.util.Collections.emptyMap(); }"
        ),
        (
            "private static Object readState() { return null; }\n"
            "private static java.util.Map<String, Object> shipConfig() { "
            "return readState(); }"
        ),
        (
            "private static final java.util.concurrent.Lock shipConfigLock = "
            "new java.util.concurrent.ReentrantLock();"
        ),
        (
            "private static final java.util.concurrent.locks.Lock shipConfigLock = "
            "new java.util.concurrent.locks.ReentrantLock();"
        ),
    ]
    target = tmp_path / "Test.java"
    classes = tmp_path / "classes"
    classes.mkdir()

    def call_coder(messages):
        captured.append(list(messages))
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def write_source(path, source):
        path.write_text(source, encoding="utf-8")

    def compile_java(_root):
        completed = subprocess.run(
            ["javac", "-d", str(classes), str(target)],
            text=True,
            capture_output=True,
            check=False,
        )
        return SimpleNamespace(
            status="PASS" if completed.returncode == 0 else "FAIL",
            log=(completed.stdout + completed.stderr),
        )

    executor = AtomicConcernExecutor(
        root=tmp_path,
        target=target,
        relative="Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
        concerns=(
            {
                "sequence": 0,
                "identifier": "id",
                "concern": "concurrency_hazards",
                "task": "implement compiler-safe concurrency state",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda report: getattr(report, "log", ""),
        write_source=write_source,
    )

    result = executor.run()

    assert result["repair_count"] == 0
    assert (classes / "example" / "Test.class").is_file()
    assert "java.util.concurrent.locks.Lock" in result["source"]
    assert "java.util.concurrent.locks.ReentrantLock" in result["source"]
    assert len(captured) == 4

    validation_failures = [
        __import__("json").loads(messages[-1]["content"])["repair_failure"]
        for messages in captured[1:]
    ]
    assert "blank final field" in validation_failures[0]
    assert "reassigns final field" in validation_failures[1]
    assert "Object-valued local call readState" in validation_failures[2]
    assert all(
        "HOST REGION VALIDATION FAILED BEFORE COMPILATION" in item
        for item in validation_failures
    )
    assert all(
        "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE" not in item
        for item in validation_failures
    )
    assert remaining == [
        (
            "private static final java.util.concurrent.locks.Lock shipConfigLock = "
            "new java.util.concurrent.locks.ReentrantLock();"
        )
    ]

def test_atomic_prompt_prevents_logged_java_failure_families_on_first_pass() -> None:
    concerns = (
        {
            "sequence": 0,
            "identifier": "id",
            "concern": "concurrency_hazards",
            "task": "concurrency",
            "rules": [],
        },
    )
    captured: list[list[dict[str, str]]] = []
    executor, _compile_calls = _multi_executor(
        [
            (
                "private static final java.util.concurrent.locks.ReentrantLock LOCK = "
                "new java.util.concurrent.locks.ReentrantLock();"
            )
        ],
        concerns=concerns,
        captured_messages=captured,
    )

    executor.run()

    payload = __import__("json").loads(captured[0][-1]["content"])
    rules = "\n".join(payload["generation_recipe"]["compiler_first_rules"])
    anchors = payload["generation_recipe"]["jdk_package_anchors"]

    assert "definitely assigned before any read" in rules
    assert "Never reassign a final field" in rules
    assert "When an API returns Object" in rules
    assert "raw collections and unchecked operations" in rules
    assert "Lock and ReentrantLock are in java.util.concurrent.locks" in rules
    assert anchors["lock_interface"] == "java.util.concurrent.locks.Lock"
    assert anchors["reentrant_lock"] == "java.util.concurrent.locks.ReentrantLock"


def test_repeated_compiler_diagnostics_ignore_line_number_churn() -> None:
    from minecraft_mod_ai.atomic_concern_source import _compiler_diagnostic_fingerprint

    first = (
        "/tmp/Test.java:10: error: cannot find symbol\n"
        "  symbol: class MissingType\n"
        "  location: class Test\n"
    )
    second = (
        "/tmp/Test.java:27: error: cannot find symbol\n"
        "  symbol: class MissingType\n"
        "  location: class Test\n"
    )

    assert _compiler_diagnostic_fingerprint(first) == _compiler_diagnostic_fingerprint(second)


def test_compile_repairs_have_hard_per_concern_bound(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "2")
    remaining = [
        "private static int VALUE = missingA();",
        "private static int VALUE = missingB();",
        "private static int VALUE = missingC();",
    ]
    state: dict[str, str] = {"source": ""}
    compile_calls = {"count": 0}

    def call_coder(_messages):
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def write_source(_path, source):
        state["source"] = source

    def compile_java(_root):
        compile_calls["count"] += 1
        rows = state["source"].splitlines()
        marker = next(
            index for index, row in enumerate(rows)
            if "MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_START" in row
        )
        line_number = marker + 2
        missing = chr(ord("A") + compile_calls["count"] - 1)
        return SimpleNamespace(
            status="FAIL",
            log=(
                f"/tmp/Test.java:{line_number}: error: cannot find symbol\n"
                f"  symbol: method missing{missing}()\n"
            ),
        )

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("/tmp/Test.java"),
        relative="/tmp/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
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
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda report: getattr(report, "log", ""),
        write_source=write_source,
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_COMPILE_REPAIR_EXHAUSTED",
    ):
        executor.run()

    assert compile_calls["count"] == 3


def test_atomic_prompt_uses_selected_region_not_whole_host_source() -> None:
    concerns = (
        {"sequence": 0, "identifier": "a", "concern": "variables", "task": "variables", "rules": []},
        {"sequence": 1, "identifier": "b", "concern": "invariants", "task": "invariants", "rules": []},
    )
    captured: list[list[dict[str, str]]] = []
    executor, _compile_calls = _multi_executor(
        [
            "private static int playerCredits;",
            "private static boolean creditsValid() { return playerCredits >= 0; }",
        ],
        concerns=concerns,
        captured_messages=captured,
    )

    executor.run()

    payload = __import__("json").loads(captured[1][-1]["content"])
    assert "current_host_owned_source" not in payload
    assert payload["current_selected_region_source"] == ""
    sibling = payload["available_sibling_api"]
    assert len(sibling) == 1
    assert sibling[0]["owner_concern"] == "variables"
    assert sibling[0]["kind"] == "field"
    assert sibling[0]["symbol"] == "playerCredits"
    assert sibling[0]["declared_type"] == "int"
    assert sibling[0]["mutable"] is True
    assert sibling[0]["static"] is True
    assert sibling[0]["typed_api_source"] == "tree_sitter_java"


def test_compiler_failure_sent_to_model_is_concern_local_and_bounded() -> None:
    from minecraft_mod_ai.atomic_concern_source import _compact_compiler_failure

    source = (
        "package example;\n"
        "public final class Test {\n"
        "// MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_START\n"
        "private static int value = missing();\n"
        "// MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_END\n"
        "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_START\n"
        "private static int other = missingOther();\n"
        "// MMM_ATOMIC_CONCERN_TRANSITIONS_MEMBERS_END\n"
        "}\n"
    )
    noisy = (
        "/tmp/Test.java:4: error: cannot find symbol\n"
        "private static int value = missing();\n"
        "^\n"
        "  symbol: method missing()\n"
        "  location: class Test\n\n"
        "/tmp/Test.java:7: error: cannot find symbol\n"
        "private static int other = missingOther();\n"
        "^\n"
        "  symbol: method missingOther()\n"
        "  location: class Test\n\n"
        + ("gradle stack noise\n" * 1000)
    )

    compact = _compact_compiler_failure(
        noisy,
        source=source,
        relative="/tmp/Test.java",
        concern="variables",
    )

    assert "missing()" in compact
    assert "missingOther" not in compact
    assert "gradle stack noise" not in compact
    assert len(compact) < 6000


def test_dependency_context_exposes_api_without_source_body() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import (
        _dependency_api_context,
        _dependency_declared_identifiers,
    )

    raw = json.dumps(
        {
            "symbol": "AuthoredBehaviorContract",
            "path": "src/main/java/example/AuthoredBehaviorContract.java",
            "responsibility": "behavior contract",
            "public_api": ["public static boolean canLaunch()"],
            "source": (
                "package example; public final class AuthoredBehaviorContract { "
                "private static final String SECRET = \"do-not-send\"; }"
            ),
        }
    )

    compact = _dependency_api_context(raw)

    assert len(compact) == 1
    assert compact[0]["symbol"] == "AuthoredBehaviorContract"
    assert compact[0]["path"] == "src/main/java/example/AuthoredBehaviorContract.java"
    assert compact[0]["responsibility"] == "behavior contract"
    assert compact[0]["public_api"] == ["public static boolean canLaunch()"]
    assert compact[0]["typed_public_api"] == []
    assert compact[0]["typed_api_source"] == "unavailable"
    assert "source" not in compact[0]
    assert "SECRET" not in json.dumps(compact)
    assert _dependency_declared_identifiers(raw) == ("AuthoredBehaviorContract",)


def test_structured_tool_cannot_emit_visibility_on_static_initializer() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "records": [],
            "enums": [],
            "classes": [],
            "fields": [],
            "methods": [],
            "static_initializers": [{"body": ["initializeSomething()"]}],
        },
        response_region="members",
    )

    assert "private static {" not in rendered
    assert rendered == "static {\n    initializeSomething();\n}"


def test_structured_tool_renders_initialize_statements_without_lifecycle_declaration() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {"statements": ["registerDefaults()"]},
        response_region="initialize",
    )

    assert rendered == "registerDefaults();"
    assert "initialize()" not in rendered


def test_structured_members_accept_omitted_empty_categories() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "fields": [
                {
                    "type": "int",
                    "name": "techLevel",
                    "initializer": "1",
                }
            ]
        },
        response_region="members",
    )

    assert rendered == "static int techLevel = 1;"


def test_nested_type_visibility_is_host_owned() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "records": [
                {
                    "modifiers": ["public", "static"],
                    "name": "StateVariable",
                    "components": [],
                    "methods": [],
                }
            ],
            "classes": [
                {
                    "modifiers": ["public"],
                    "name": "Holder",
                    "fields": [],
                    "constructors": [],
                    "methods": [],
                }
            ],
            "enums": [
                {
                    "modifiers": ["public"],
                    "name": "Mode",
                    "constants": ["GROUND"],
                }
            ],
        },
        response_region="members",
    )

    assert "public record" not in rendered
    assert "public class" not in rendered
    assert "public enum" not in rendered
    assert "private record StateVariable()" in rendered
    assert "private static class Holder" in rendered
    assert "private enum Mode" in rendered


def test_parameter_schema_tolerates_small_model_metadata_noise() -> None:
    from jsonschema import Draft202012Validator

    from minecraft_mod_ai.custom_module_generator import _ATOMIC_MEMBERS_PARAMETERS

    decision = {
        "classes": [
            {
                "name": "Holder",
                "constructors": [
                    {
                        "parameters": [
                            {
                                "type": "int",
                                "name": "value",
                                "description": "constructor value",
                                "modifiers": ["final"],
                            }
                        ]
                    }
                ],
            }
        ]
    }

    Draft202012Validator(_ATOMIC_MEMBERS_PARAMETERS).validate(decision)


def test_member_tool_schema_has_no_arbitrary_cardinality_or_length_caps() -> None:
    from minecraft_mod_ai.custom_module_generator import _ATOMIC_MEMBERS_PARAMETERS

    def walk(value):
        if isinstance(value, dict):
            assert "maxItems" not in value
            assert "maxLength" not in value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(_ATOMIC_MEMBERS_PARAMETERS)


def test_preferred_shape_is_guidance_not_schema_restriction() -> None:
    from minecraft_mod_ai.custom_module_generator import (
        _ATOMIC_MEMBERS_PARAMETERS,
        _atomic_parameters_for_request,
    )

    for preferred in ("fields_and_local_types", "methods_and_constants"):
        parameters, shape = _atomic_parameters_for_request(
            {"generation_recipe": {"preferred_shape": preferred}},
            response_region="members",
        )
        assert shape == preferred
        assert parameters is _ATOMIC_MEMBERS_PARAMETERS
        assert set(parameters["properties"]) == {
            "records",
            "enums",
            "classes",
            "fields",
            "methods",
        }

def test_structured_output_exhaustion_becomes_bounded_concern_failure() -> None:
    from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
    from minecraft_mod_ai.llama_finish_reason_contract import (
        LlamaCompletionBoundaryError,
    )
    from minecraft_mod_ai.model_adapters.base import ModelBackendError

    class _Router:
        def generate_tool_decision(self, role, messages, **kwargs):
            del role, messages, kwargs
            raise ModelBackendError(
                role="coder",
                model_id="test",
                cause=LlamaCompletionBoundaryError(
                    "native llama-server exhausted the bounded output allowance before "
                    "the assistant action completed; prompt_tokens=100 completion_tokens=2048 "
                    "max_tokens=2048",
                    kind="output_exhausted",
                    prompt_tokens=100,
                    completion_tokens=2048,
                    max_tokens=2048,
                ),
            )

    with pytest.raises(
        CustomModuleGenerationError,
        match="OUTPUT_BUDGET_EXHAUSTED",
    ):
        _call_atomic_java_region(
            _Router(),
            (
                {"role": "system", "content": "structured only"},
                {
                    "role": "user",
                    "content": (
                        '{"response_region":"members","generation_recipe":'
                        '{"preferred_shape":"methods_and_constants"}}'
                    ),
                },
            ),
            output_token_ceiling=2048,
        )


def test_java_reserved_identifiers_are_canonicalized_consistently() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "records": [
                {
                    "name": "VariablesRecord",
                    "components": [
                        {"type": "String", "name": "name"},
                        {"type": "String", "name": "default"},
                    ],
                    "methods": [
                        {
                            "return_type": "String",
                            "name": "getDefault",
                            "body": ["return default"],
                        }
                    ],
                }
            ]
        },
        response_region="members",
    )

    assert "String default" not in rendered
    assert "String $mmm$default" in rendered
    assert "return $mmm$default;" in rendered


def test_java_keyword_rewrite_does_not_break_switch_default_label() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "fields": [{"type": "int", "name": "default", "initializer": "1"}],
            "methods": [
                {
                    "return_type": "int",
                    "name": "pick",
                    "parameters": [{"type": "int", "name": "value"}],
                    "body": [
                        "switch (value) {",
                        "default:",
                        "return default",
                        "}",
                    ],
                }
            ],
        },
        response_region="members",
    )

    assert "int $mmm$default = 1;" in rendered
    assert "default:" in rendered
    assert "return $mmm$default;" in rendered


def test_atomic_native_tool_call_does_not_force_legacy_2048_ceiling() -> None:
    from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region

    captured = {}
    responses = iter([{ "part": "fields"}, {"type": "int", "name": "value"}, {"part": "done"}, {"part": "done"}])

    class _Router:
        def generate_tool_decision(self, role, messages, **kwargs):
            del role, messages
            captured.update(kwargs)
            return next(responses)

    rendered = _call_atomic_java_region(
        _Router(),
        (
            {"role": "system", "content": "structured only"},
            {"role": "user", "content": '{"response_region":"members"}'},
        ),
        output_token_ceiling=None,
    )

    assert rendered == "private static int value;"
    assert "output_token_ceiling" not in captured


def test_record_methods_are_not_arbitrarily_capped_by_tool_schema() -> None:
    from jsonschema import Draft202012Validator

    from minecraft_mod_ai.custom_module_generator import _ATOMIC_MEMBERS_PARAMETERS

    decision = {
        "records": [
            {
                "name": "VariablesRecord",
                "components": [
                    {"type": "String", "name": "name"},
                    {"type": "String", "name": "owner"},
                    {"type": "String", "name": "type"},
                    {"type": "String", "name": "unit"},
                    {"type": "String", "name": "default"},
                    {"type": "String", "name": "domain"},
                ],
                "methods": [
                    {
                        "return_type": "String",
                        "name": f"getter{index}",
                        "body": ["return name"],
                    }
                    for index in range(8)
                ],
            }
        ]
    }

    Draft202012Validator(_ATOMIC_MEMBERS_PARAMETERS).validate(decision)


def test_atomic_member_schema_accepts_model_modifier_noise_for_host_filtering() -> None:
    from jsonschema import Draft202012Validator

    from minecraft_mod_ai.custom_module_generator import _ATOMIC_MEMBERS_PARAMETERS

    Draft202012Validator(_ATOMIC_MEMBERS_PARAMETERS).validate(
        {
            "fields": [
                {
                    "modifiers": ["private", "static", "model_extra_modifier"],
                    "type": "int",
                    "name": "value",
                    "initializer": "1",
                }
            ]
        }
    )


def test_state_model_variables_are_host_lowered_to_runtime_fields() -> None:
    import json

    obligation = json.dumps(
        {
            "instruction": json.dumps(
                {
                    "concern": "variables",
                    "section": "state_model",
                }
            ),
            "source_requirements": {
                "R12": (
                    "- variables: name(player_credits), owner(Player) type(double) "
                    "unit(currency) default(0.0) domain(Unbounded Positive)"
                )
            },
        }
    )
    model_calls = {"count": 0}

    def call_coder(_messages):
        model_calls["count"] += 1
        raise AssertionError("explicit state variables must be host-lowered")

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={
            "task_id": "state",
            "semantic_outcome": "state",
            "implementation_obligations": [obligation],
        },
        section="state_model",
        concerns=(
            {
                "sequence": 0,
                "identifier": "v",
                "concern": "variables",
                "task": "variables",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )

    result = executor.run()

    assert model_calls["count"] == 0
    assert "private static double player_credits = 0.0;" in result["source"]
    assert "record Variables" not in result["source"]


def test_sibling_api_exposes_exact_type_and_mutability() -> None:
    from minecraft_mod_ai.atomic_concern_source import _sibling_symbol_inventory

    source = (
        "package example;\n"
        "public final class Test {\n"
        "// MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_START\n"
        "private static double player_credits = 0.0;\n"
        "private static final String MODE = \"ground\";\n"
        "private record Snapshot(double credits) {}\n"
        "// MMM_ATOMIC_CONCERN_VARIABLES_MEMBERS_END\n"
        "}\n"
    )

    api = _sibling_symbol_inventory(
        source,
        sibling_concerns=("variables",),
    )

    credits = next(row for row in api if row["symbol"] == "player_credits")
    mode = next(row for row in api if row["symbol"] == "MODE")
    snapshot = next(row for row in api if row["symbol"] == "Snapshot")
    assert credits["mutable"] is True
    assert "double player_credits" in credits["declaration"]
    assert mode["mutable"] is False
    assert "final String MODE" in mode["declaration"]
    assert "record Snapshot(double credits)" in snapshot["declaration"]


def test_nested_init_method_is_normalized_to_record_constructor() -> None:
    from jsonschema import Draft202012Validator

    from minecraft_mod_ai.custom_module_generator import (
        _ATOMIC_MEMBERS_PARAMETERS,
        _render_atomic_java_structure,
    )

    decision = {
        "records": [
            {
                "name": "Snapshot",
                "components": [{"type": "double", "name": "credits"}],
                "methods": [
                    {
                        "return_type": "void",
                        "name": "<init>",
                        "parameters": [{"type": "double", "name": "credits"}],
                        "body": ["this.credits = credits"],
                    }
                ],
            }
        ]
    }

    Draft202012Validator(_ATOMIC_MEMBERS_PARAMETERS).validate(decision)
    rendered = _render_atomic_java_structure(
        decision,
        response_region="members",
    )

    assert "<init>" not in rendered
    assert "private Snapshot {" in rendered
    assert "credits = credits;" in rendered


def test_record_explicit_constructor_slot_is_supported() -> None:
    from jsonschema import Draft202012Validator

    from minecraft_mod_ai.custom_module_generator import (
        _ATOMIC_MEMBERS_PARAMETERS,
        _render_atomic_java_structure,
    )

    decision = {
        "records": [
            {
                "name": "Snapshot",
                "components": [{"type": "double", "name": "credits"}],
                "constructors": [
                    {
                        "parameters": [{"type": "double", "name": "credits"}],
                        "body": ["if (credits < 0) credits = 0"],
                    }
                ],
            }
        ]
    }

    Draft202012Validator(_ATOMIC_MEMBERS_PARAMETERS).validate(decision)
    rendered = _render_atomic_java_structure(
        decision,
        response_region="members",
    )

    assert "private Snapshot {" in rendered
    assert "if (credits < 0) credits = 0;" in rendered


def test_state_model_followup_is_fully_host_compiled_from_structured_records() -> None:
    import json

    obligation_variables = json.dumps(
        {
            "instruction": json.dumps({"concern": "variables"}),
            "source_requirements": {
                "R12": "- variables: name(player_credits) type(double) default(0.0)"
            },
            "structured_records": [
                {"name": "player_credits", "type": "double", "default": "0.0"}
            ],
        }
    )
    obligation_invariants = json.dumps(
        {
            "instruction": json.dumps({"concern": "invariants"}),
            "source_requirements": {
                "R14": "- invariants: player_credits cannot be negative"
            },
            "structured_records": [
                {"condition": "player_credits >= 0.0", "enforcement": "reject"}
            ],
        }
    )

    def call_coder(_messages):
        raise AssertionError("complete structured state must never enter model Java generation")

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={
            "task_id": "state",
            "semantic_outcome": "state",
            "implementation_obligations": [
                obligation_variables,
                obligation_invariants,
            ],
        },
        section="state_model",
        concerns=(
            {"sequence": 0, "identifier": "v", "concern": "variables", "task": "variables", "rules": []},
            {"sequence": 1, "identifier": "i", "concern": "invariants", "task": "invariants", "rules": []},
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )

    result = executor.run()

    assert result["repair_count"] == 0
    assert '"player_credits"' in result["source"]
    assert "$mmmState.put" in result["source"]
    assert "$mmmInvariants.add" in result["source"]

def test_concern_authority_keeps_repeated_same_concern_rows() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _concern_authority

    obligation = json.dumps(
        {
            "instruction": json.dumps({"concern": "transitions"}),
            "source_requirements": {
                "R12": "## state_model",
                "R13": "- variables: name(player_credits) type(double) default(0.0)",
                "R14": "- transitions: from_state(shipping_lock) trigger(purchase_action) guard(sufficient_funds) to_state(ship_ready)",
                "R15": "- transitions: from_state(space_travel) trigger(dock_event) guard(fuel_level_zero) to_state(portal_entry)",
                "R16": "- invariants: player_credits cannot be negative",
            },
        }
    )

    authority = _concern_authority(
        {
            "task_id": "state",
            "implementation_obligations": [obligation],
        },
        {"concern": "transitions"},
    )

    assert authority["source_requirements"] == {
        "R12": "## state_model",
        "R14": "- transitions: from_state(shipping_lock) trigger(purchase_action) guard(sufficient_funds) to_state(ship_ready)",
        "R15": "- transitions: from_state(space_travel) trigger(dock_event) guard(fuel_level_zero) to_state(portal_entry)",
    }


def test_state_variable_contract_keeps_repeated_variable_rows() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _state_variable_contract

    obligation = json.dumps(
        {
            "instruction": json.dumps({"concern": "variables"}),
            "source_requirements": {
                "R12": "## state_model",
                "R13": "- variables: name(player_credits) type(double) default(0.0)",
                "R14": "- variables: name(fuel_reserve) type(int) default(100)",
                "R15": "- transitions: from_state(ground) trigger(launch) to_state(space)",
            },
        }
    )

    contract = _state_variable_contract(
        {
            "task_id": "state",
            "implementation_obligations": [obligation],
        },
        {"concern": "variables"},
    )

    assert [(row["name"], row["java_type"], row["default_literal"]) for row in contract] == [
        ("player_credits", "double", "0.0"),
        ("fuel_reserve", "int", "100"),
    ]


def test_atomic_prompt_hides_planning_record_schema_from_coder() -> None:
    captured = []
    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
        concerns=(
            {
                "sequence": 0,
                "identifier": "id",
                "concern": "inputs",
                "task": "Resolve exactly one inputs record for the supplied feature.",
                "rules": ["Author the requested gameplay record."],
                "record_schema": {
                    "name": "string",
                    "type": "string",
                    "default": "string",
                },
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=lambda messages: (
            captured.append(messages)
            or "private static final String INPUT = \"construct_ship\";"
        ),
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )

    executor.run()

    payload = __import__("json").loads(captured[0][-1]["content"])
    assert {
        "sequence",
        "identifier",
        "name",
        "implementation_goal",
        "task",
        "rules",
        "java_shape",
        "semantic_fields",
    } <= set(payload["concern"])
    assert "record_schema" not in payload["concern"]
    assert "record_schema" not in captured[0][-1]["content"]
    assert payload["concern"]["task"]
    assert payload["concern"]["rules"]


def test_logic_concerns_cannot_emit_nested_types() -> None:
    from jsonschema import Draft202012Validator, ValidationError

    from minecraft_mod_ai.custom_module_generator import (
        _ATOMIC_LOGIC_MEMBERS_PARAMETERS,
        _atomic_parameters_for_request,
    )

    parameters, _shape = _atomic_parameters_for_request(
        {
            "concern": {"name": "initialization"},
            "generation_recipe": {"preferred_shape": "methods_and_constants"},
        },
        response_region="members",
    )

    assert parameters is _ATOMIC_LOGIC_MEMBERS_PARAMETERS
    assert set(parameters["properties"]) == {
        "fields",
        "methods",
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(parameters).validate(
            {"classes": [{"name": "Initializer"}]}
        )


def test_type_owning_concern_keeps_nested_type_schema() -> None:
    from minecraft_mod_ai.custom_module_generator import (
        _ATOMIC_MEMBERS_PARAMETERS,
        _atomic_parameters_for_request,
    )

    parameters, _shape = _atomic_parameters_for_request(
        {"concern": {"name": "variables"}},
        response_region="members",
    )

    assert parameters is _ATOMIC_MEMBERS_PARAMETERS
    assert {"records", "enums", "classes"} <= set(parameters["properties"])


def test_duplicate_constructor_shapes_are_deduplicated_by_host() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "classes": [
                {
                    "name": "PlayerData",
                    "constructors": [{"parameters": [], "body": []}],
                    "methods": [
                        {
                            "return_type": "void",
                            "name": "<init>",
                            "parameters": [],
                            "body": [],
                        }
                    ],
                }
            ]
        },
        response_region="members",
    )

    assert rendered.count("PlayerData() {") == 1


def test_nested_type_cannot_shadow_host_selected_outer_class() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    with pytest.raises(
        CustomModuleGenerationError,
        match="collides with the host-selected outer class",
    ):
        _render_atomic_java_structure(
            {"classes": [{"name": "AuthoredStateModel"}]},
            response_region="members",
            host_symbol="AuthoredStateModel",
        )


def test_outer_atomic_fields_and_methods_are_forced_static() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    rendered = _render_atomic_java_structure(
        {
            "fields": [{"type": "int", "name": "credits"}],
            "methods": [
                {
                    "return_type": "int",
                    "name": "credits",
                    "body": ["return credits"],
                }
            ],
        },
        response_region="members",
    )

    assert "static int credits;" in rendered
    assert "static int credits()" in rendered


def test_compiler_repair_cannot_expand_nested_type_structure(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "2")
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "1")
    compile_calls = {"count": 0}
    outputs = iter(
        [
            "private static int value = missing();",
            (
                "private static int value = 1;\n"
                "private static class AuthoredStateModel {}"
            ),
            "private static int value = 1;",
        ]
    )

    def call_coder(_messages):
        return next(outputs)

    def compile_java(_root):
        compile_calls["count"] += 1
        if compile_calls["count"] == 1:
            return SimpleNamespace(status="FAIL")
        return SimpleNamespace(status="PASS")

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
        concerns=(
            {
                "sequence": 0,
                "identifier": "id",
                "concern": "transitions",
                "task": "transitions",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda _report: (
            f"/tmp/Test.java:{next(i for i, line in enumerate(executor.source.splitlines(), 1) if 'missing()' in line)}: error: cannot find symbol\n"
            "private static int value = missing();\n"
            "                           ^\n"
            "  symbol: method missing()\n"
        ),
        write_source=lambda _path, _source: None,
    )

    result = executor.run()

    assert compile_calls["count"] == 2
    assert "class AuthoredStateModel" not in result["source"]
    assert "private static int value = 1;" in result["source"]


def test_noncanonical_record_constructor_must_delegate() -> None:
    from minecraft_mod_ai.custom_module_generator import _render_atomic_java_structure

    with pytest.raises(
        CustomModuleGenerationError,
        match="must delegate to the canonical constructor",
    ):
        _render_atomic_java_structure(
            {
                "records": [
                    {
                        "name": "PlayerData",
                        "components": [{"type": "int", "name": "credits"}],
                        "constructors": [
                            {
                                "parameters": [],
                                "body": ["credits = 0"],
                            }
                        ],
                    }
                ]
            },
            response_region="members",
        )


def test_stored_state_invalid_shape_fails_first_pass_without_regeneration() -> None:
    calls = {"count": 0}

    def call_coder(_messages):
        calls["count"] += 1
        return "private static void registerState(String key, String value) {}"

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="persistence",
        concerns=(
            {
                "sequence": 0,
                "identifier": "persistence.stored_state",
                "concern": "stored_state",
                "task": "store persistent state declarations",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID",
    ):
        executor.run()

    assert calls["count"] == 1


def test_multiline_behavior_actor_source_is_host_lowered_without_structured_plan() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _behavior_actor_records
    from minecraft_mod_ai.authored_execution_schema import concern_contracts
    from minecraft_mod_ai.implementation_graph_execution import (
        _canonical_atomic_obligations,
    )

    requirements = {
        "R2": "## behavior_contract",
        "R3": "- actors:",
        "R4": "    - player: 자원 채굴, 제작, 거래를 실행하는 주체",
        "R5": "    - dockyard_ai: 건설 요청을 검증하고 배치하는 자동화 로직",
        "R6": "    - trade_npc: 자원 및 통화를 교환하는 상점 NPC",
        "R7": "- entry_conditions:",
        "R8": "    - trigger_owner: player",
    }
    concerns = list(concern_contracts("behavior_contract"))
    obligations, drifted, active = _canonical_atomic_obligations(
        section="behavior_contract",
        concerns=concerns,
        requirements=requirements,
        raw_obligations=[],
        structured_sections={},
    )

    assert drifted == []
    assert [item["concern"] for item in active][:2] == [
        "actors",
        "entry_conditions",
    ]

    task = {"implementation_obligations": obligations}
    actors = next(item for item in active if item["concern"] == "actors")
    rows = _behavior_actor_records(task, actors)

    assert rows == (
        {
            "name": "player",
            "role": "자원 채굴, 제작, 거래를 실행하는 주체",
            "authority": "",
        },
        {
            "name": "dockyard_ai",
            "role": "건설 요청을 검증하고 배치하는 자동화 로직",
            "authority": "",
        },
        {
            "name": "trade_npc",
            "role": "자원 및 통화를 교환하는 상점 NPC",
            "authority": "",
        },
    )

    actor_payload = next(
        json.loads(raw)
        for raw in obligations
        if json.loads(json.loads(raw)["instruction"])["concern"] == "actors"
    )
    assert list(actor_payload["source_requirements"]) == ["R2", "R3", "R4", "R5", "R6"]


def test_bold_inline_behavior_actor_source_is_host_lowered_without_structured_plan() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _behavior_actor_records
    from minecraft_mod_ai.authored_execution_schema import concern_contracts
    from minecraft_mod_ai.implementation_graph_execution import (
        _canonical_atomic_obligations,
    )

    requirements = {
        "R2": "## behavior_contract",
        "R3": (
            "- **actors**: 플레이어 (User), "
            "우주선 컨트롤러 (ShipController, 서버 권위), "
            "행성 관리자 (PlanetManager, 서버 권위), "
            "시장 중재자 (MarketArbitrator, 클라이언트 예측 + 서버 검증)."
        ),
        "R4": "- **entry_conditions**: 플레이어가 우주 도크에 접근한 경우.",
    }
    concerns = list(concern_contracts("behavior_contract"))
    obligations, drifted, active = _canonical_atomic_obligations(
        section="behavior_contract",
        concerns=concerns,
        requirements=requirements,
        raw_obligations=[],
        structured_sections={},
    )

    assert drifted == []
    assert [item["concern"] for item in active][:2] == [
        "actors",
        "entry_conditions",
    ]

    task = {"implementation_obligations": obligations}
    actors = next(item for item in active if item["concern"] == "actors")
    rows = _behavior_actor_records(task, actors)

    assert rows == (
        {"name": "플레이어", "role": "플레이어", "authority": "User"},
        {
            "name": "우주선 컨트롤러",
            "role": "우주선 컨트롤러",
            "authority": "ShipController, 서버 권위",
        },
        {
            "name": "행성 관리자",
            "role": "행성 관리자",
            "authority": "PlanetManager, 서버 권위",
        },
        {
            "name": "시장 중재자",
            "role": "시장 중재자",
            "authority": "MarketArbitrator, 클라이언트 예측 + 서버 검증",
        },
    )

    actor_payload = next(
        json.loads(raw)
        for raw in obligations
        if json.loads(json.loads(raw)["instruction"])["concern"] == "actors"
    )
    assert list(actor_payload["source_requirements"]) == ["R2", "R3"]


def test_sibling_inventory_exposes_exact_generic_field_type() -> None:
    from minecraft_mod_ai.atomic_concern_source import _sibling_symbol_inventory

    source = (
        "final class Probe {\n"
        "// MMM_ATOMIC_CONCERN_STEPS_MEMBERS_START\n"
        "private static final java.util.Map<String, "
        "java.util.Map<String, java.lang.Object>> FAILURE_RULES = "
        "new java.util.HashMap<>();\n"
        "// MMM_ATOMIC_CONCERN_STEPS_MEMBERS_END\n"
        "// MMM_ATOMIC_CONCERN_STEPS_INIT_START\n"
        "// MMM_ATOMIC_CONCERN_STEPS_INIT_END\n"
        "}\n"
    )

    rows = _sibling_symbol_inventory(
        source,
        sibling_concerns=("steps",),
    )

    field = next(row for row in rows if row["symbol"] == "FAILURE_RULES")
    assert field["declared_type"] == (
        "java.util.Map<java.lang.String, "
        "java.util.Map<java.lang.String, java.lang.Object>>"
    )
    assert field["generic_type_is_authoritative"] is True


def test_map_entry_value_generic_narrowing_is_canonicalized() -> None:
    from minecraft_mod_ai.atomic_concern_source import (
        _canonicalize_generated_jdk_semantics,
    )

    candidate = (
        "private static boolean failClosed() {\n"
        "    for (java.util.Map.Entry<String, "
        "java.util.Map<String, Object>> entry : "
        "FAILURE_RULES.entrySet()) {\n"
        "        java.util.Map<String, String> condition = entry.getValue();\n"
        "        if (condition.isEmpty()) { return false; }\n"
        "    }\n"
        "    return true;\n"
        "}"
    )

    normalized, changes = _canonicalize_generated_jdk_semantics(
        candidate,
        authoritative_field_types={
            "FAILURE_RULES": (
                "java.util.Map<java.lang.String, "
                "java.util.Map<java.lang.String, java.lang.Object>>"
            )
        },
    )

    assert (
        "java.util.Map<java.lang.String, java.lang.Object> condition = "
        "entry.getValue();"
    ) in normalized
    assert "java.util.Map<String, String> condition" not in normalized
    assert any("entry.getValue" in change for change in changes)


def test_dependency_api_context_uses_tree_sitter_typed_contracts() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _dependency_api_context

    source = """
package example;

public final class Dependency {
    public static Object getState(String key) { return null; }
    public static java.util.Map<String, Object> snapshot(String key) {
        return java.util.Map.of();
    }
    private static void hidden() {}
}
"""
    raw = json.dumps(
        {
            "symbol": "Dependency",
            "path": "src/main/java/example/Dependency.java",
            "responsibility": "state dependency",
            "public_api": [
                "public static Object getState(String key)",
                "public static java.util.Map<String, Object> snapshot(String key)",
            ],
            "source": source,
        }
    )

    rows = _dependency_api_context(raw)

    assert len(rows) == 1
    row = rows[0]
    assert row["typed_api_source"] == "tree_sitter_java"
    assert [(item["kind"], item["symbol"]) for item in row["typed_public_api"]] == [
        ("method", "getState"),
        ("method", "snapshot"),
    ]
    assert row["typed_public_api"][0]["return_type"] == "Object"
    assert row["typed_public_api"][1]["return_type"] == "java.util.Map<String, Object>"


def test_authority_return_types_prefer_typed_contracts_and_drop_ambiguous_overloads() -> None:
    from minecraft_mod_ai.custom_module_generator import _authority_method_return_types

    payload = {
        "available_sibling_api": [
            {
                "kind": "method",
                "symbol": "localValue",
                "return_type": "java.util.Map<String, Object>",
                "declaration": "public static Object localValue() { ... }",
            }
        ],
        "dependency_api": [
            {
                "typed_public_api": [
                    {
                        "kind": "method",
                        "symbol": "getState",
                        "return_type": "Object",
                        "parameters": [{"type": "String", "name": "key"}],
                    },
                    {
                        "kind": "method",
                        "symbol": "lookup",
                        "return_type": "String",
                        "parameters": [{"type": "String", "name": "key"}],
                    },
                    {
                        "kind": "method",
                        "symbol": "lookup",
                        "return_type": "Object",
                        "parameters": [{"type": "int", "name": "id"}],
                    },
                ],
                "public_api": [
                    "public static String getState(String key)"
                ],
            }
        ],
    }

    returns = _authority_method_return_types(payload)

    assert returns["localValue"] == "java.util.Map<String, Object>"
    assert returns["getState"] == "Object"
    assert "lookup" not in returns


def test_first_pass_tree_sitter_gate_rejects_blank_final_before_compile() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    with pytest.raises(
        CustomModuleGenerationError,
        match="blank final field.*NO_FLUID_STORAGE_IN_ZERO_G",
    ):
        _validate_first_pass_java_semantics(
            "private static final boolean NO_FLUID_STORAGE_IN_ZERO_G;",
            dependency_source="",
            sibling_api=(),
        )


def test_first_pass_tree_sitter_gate_rejects_final_rebinding_before_compile() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    source = (
        "private static void reset() { "
        "TRANSFER_CACHE = java.util.Collections.emptyMap(); }"
    )
    with pytest.raises(
        CustomModuleGenerationError,
        match="reassigns final field.*TRANSFER_CACHE",
    ):
        _validate_first_pass_java_semantics(
            source,
            dependency_source="",
            sibling_api=(
                {
                    "kind": "field",
                    "symbol": "TRANSFER_CACHE",
                    "declared_type": "java.util.Map<String, Object>",
                    "mutable": False,
                },
            ),
        )


def test_first_pass_tree_sitter_gate_rejects_unknown_dependency_method_and_arity() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "path": "AuthoredStateModel.java",
            "public_api": ["public static Object getState(String key)"],
            "source": (
                "public final class AuthoredStateModel { "
                "public static Object getState(String key) { return null; } "
                "}"
            ),
        }
    )

    with pytest.raises(CustomModuleGenerationError, match="does not exist"):
        _validate_first_pass_java_semantics(
            "private static Object read() { "
            "return AuthoredStateModel.getUnknown(\"x\"); }",
            dependency_source=dependency_source,
            sibling_api=(),
        )

    with pytest.raises(CustomModuleGenerationError, match="called with 2 argument"):
        _validate_first_pass_java_semantics(
            "private static Object read() { "
            "return AuthoredStateModel.getState(\"x\", 1); }",
            dependency_source=dependency_source,
            sibling_api=(),
        )


def test_first_pass_tree_sitter_gate_rejects_object_direct_return_to_map() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "path": "AuthoredStateModel.java",
            "public_api": ["public static Object getState(String key)"],
            "source": (
                "public final class AuthoredStateModel { "
                "public static Object getState(String key) { return null; } "
                "}"
            ),
        }
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match="directly returns Object-valued dependency call",
    ):
        _validate_first_pass_java_semantics(
            "private static java.util.Map<String, Object> shipConfig() { "
            "return AuthoredStateModel.getState(\"ship\"); }",
            dependency_source=dependency_source,
            sibling_api=(),
        )


def test_first_pass_tree_sitter_gate_accepts_explicit_object_narrowing() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "path": "AuthoredStateModel.java",
            "public_api": ["public static Object getState(String key)"],
            "source": (
                "public final class AuthoredStateModel { "
                "public static Object getState(String key) { return null; } "
                "}"
            ),
        }
    )

    _validate_first_pass_java_semantics(
        (
            "private static java.util.Map<String, Object> shipConfig() { "
            "Object raw = AuthoredStateModel.getState(\"ship\"); "
            "if (!(raw instanceof java.util.Map<?, ?> map)) { "
            "return java.util.Map.of(); } "
            "java.util.Map<String, Object> result = new java.util.HashMap<>(); "
            "for (java.util.Map.Entry<?, ?> entry : map.entrySet()) { "
            "if (entry.getKey() instanceof String key) { "
            "result.put(key, entry.getValue()); } } "
            "return result; }"
        ),
        dependency_source=dependency_source,
        sibling_api=(),
    )


def test_first_pass_tree_sitter_gate_rejects_ungrounded_simple_type_name() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*ReentrantLokk",
    ):
        _validate_first_pass_java_semantics(
            "private static ReentrantLokk lock;",
            dependency_source="",
            sibling_api=(),
        )


def test_first_pass_tree_sitter_gate_rejects_ungrounded_generic_leaf_type() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*HashMapp",
    ):
        _validate_first_pass_java_semantics(
            "private static HashMapp<String, Object> cache;",
            dependency_source="",
            sibling_api=(),
        )


def test_first_pass_type_authority_accepts_dependency_declared_simple_type() -> None:
    import json

    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "path": "AuthoredStateModel.java",
            "public_api": [
                "public static State currentState()",
            ],
            "source": (
                "public final class AuthoredStateModel { "
                "public static record State(String name) {} "
                "public static State currentState() { return new State(\"ok\"); } "
                "}"
            ),
        }
    )

    _validate_first_pass_java_semantics(
        "private static State copy(State input) { return input; }",
        dependency_source=dependency_source,
        sibling_api=(),
    )


def test_first_pass_type_authority_defers_fqcn_binding_to_jdt_javac() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    _validate_first_pass_java_semantics(
        "private static com.example.ExternalType value;",
        dependency_source="",
        sibling_api=(),
    )


def test_first_pass_canonicalizes_unique_installed_jdk_simple_types() -> None:
    from minecraft_mod_ai.atomic_concern_source import _canonicalize_generated_jdk_semantics

    source, changes = _canonicalize_generated_jdk_semantics(
        (
            "private static Optional<String> maybe;\n"
            "private static UUID id;"
        )
    )

    assert "java.util.Optional<String> maybe;" in source
    assert "java.util.UUID id;" in source
    assert any("Optional->java.util.Optional" in item for item in changes)
    assert any("UUID->java.util.UUID" in item for item in changes)


def test_ambiguous_installed_jdk_simple_type_requires_explicit_fqcn() -> None:
    from minecraft_mod_ai.atomic_concern_source import (
        _canonicalize_generated_jdk_semantics,
        _validate_first_pass_java_semantics,
    )

    source, changes = _canonicalize_generated_jdk_semantics(
        "private static Duration timeout;"
    )

    assert source == "private static Duration timeout;"
    assert not changes
    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*Duration",
    ):
        _validate_first_pass_java_semantics(
            source,
            dependency_source="",
            sibling_api=(),
        )


def test_first_pass_canonicalizes_wrong_unique_jdk_fqcn_from_installed_jdk() -> None:
    from minecraft_mod_ai.atomic_concern_source import _canonicalize_generated_jdk_semantics

    source, _changes = _canonicalize_generated_jdk_semantics(
        (
            "private static final java.util.concurrent.ReentrantLock LOCK = "
            "new java.util.concurrent.ReentrantLock();"
        )
    )

    assert "java.util.concurrent.ReentrantLock" not in source
    assert "java.util.concurrent.locks.ReentrantLock" in source


def test_first_pass_type_authority_rejects_nonexistent_java_fqcn() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    with pytest.raises(
        CustomModuleGenerationError,
        match="ungrounded simple Java type.*java.util.DoesNotExist",
    ):
        _validate_first_pass_java_semantics(
            "private static java.util.DoesNotExist value;",
            dependency_source="",
            sibling_api=(),
        )


def test_jdk_type_index_prefers_project_toolchain(tmp_path, monkeypatch) -> None:
    from minecraft_mod_ai.jdk_type_index import _java_home

    project_home = tmp_path / "project-jdk"
    system_home = tmp_path / "system-jdk"
    (project_home / "lib").mkdir(parents=True)
    (system_home / "lib").mkdir(parents=True)
    (project_home / "lib" / "modules").write_bytes(b"project")
    (system_home / "lib" / "modules").write_bytes(b"system")

    monkeypatch.setenv("MMM_PROJECT_JAVA_HOME", str(project_home))
    monkeypatch.setenv("JAVA_HOME", str(system_home))

    assert _java_home() == project_home.resolve()


def test_jdk_image_parser_indexes_only_top_level_java_types() -> None:
    from minecraft_mod_ai.jdk_type_index import _parse_jimage_type_index

    index = _parse_jimage_type_index(
        """
Module: java.base
    java/lang/String.class
    java/util/Optional.class
    java/util/Map$Entry.class
Module: java.sql
    java/sql/Date.class
Module: other
    com/example/Hidden.class
"""
    )

    assert index["String"] == ("java.lang.String",)
    assert index["Optional"] == ("java.util.Optional",)
    assert index["Date"] == ("java.sql.Date",)
    assert "Entry" not in index
    assert "Hidden" not in index


def test_first_pass_rejects_invalid_installed_jdk_constructor_arity() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    with pytest.raises(
        CustomModuleGenerationError,
        match="installed-JDK constructor java.lang.Object called with 1 argument",
    ):
        _validate_first_pass_java_semantics(
            "private static Object value = new Object(1);",
            dependency_source="",
            sibling_api=(),
        )


def test_first_pass_accepts_valid_installed_jdk_constructor_arity() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    _validate_first_pass_java_semantics(
        "private static Object value = new Object();",
        dependency_source="",
        sibling_api=(),
    )


def test_javap_constructor_shape_parser_handles_varargs() -> None:
    from minecraft_mod_ai.jdk_type_index import _constructor_shapes_from_javap

    shapes = _constructor_shapes_from_javap(
        (
            "public final class java.example.Sample {\n"
            "  public java.example.Sample();\n"
            "  public java.example.Sample(java.lang.String, java.lang.Object...);\n"
            "}\n"
        ),
        "java.example.Sample",
    )

    assert shapes == (
        {"arity": 0, "varargs": False},
        {"arity": 2, "varargs": True},
    )


def test_small_model_prompt_exposes_exact_dependency_calls_and_drops_irrelevant_grounding() -> None:
    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "path": "src/main/java/example/AuthoredStateModel.java",
            "responsibility": "host-compiled state runtime",
            "public_api": [
                "public static void initialize()",
                "public static synchronized Object getState(String name)",
                "public static synchronized Object getState(String name, java.util.Map<String, Object> context)",
                "public static synchronized void setState(String name, Object value)",
                "public static synchronized void setState(String name, Object value, java.util.Map<String, Object> context)",
            ],
            "source": (
                "package example; public final class AuthoredStateModel { "
                "public static void initialize() {} "
                "public static synchronized Object getState(String name) { return null; } "
                "public static synchronized Object getState(String name, java.util.Map<String, Object> context) { return null; } "
                "public static synchronized void setState(String name, Object value) {} "
                "public static synchronized void setState(String name, Object value, java.util.Map<String, Object> context) {} "
                "}"
            ),
        },
        ensure_ascii=False,
    )
    concern = {
        "sequence": 0,
        "identifier": "feature/failure_and_limits/invalid_inputs",
        "concern": "invalid_inputs",
        "task": "Reject invalid input without corrupting state.",
        "rules": [],
        "record_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }
    messages = _messages(
        section="failure_and_limits",
        concern=concern,
        task={"task_id": "t", "implementation_obligations": []},
        grounding={
            "schema_version": "mmm/host-owned-coder-grounding",
            "facts": [{"huge_irrelevant_platform_fact": "x" * 5000}],
            "policy": {},
            "direct_host_context": {
                "platform": {"java_version": 25},
                "host_version_facts": {"api_symbols": {"noise": {"owner": "net.minecraft.Noise"}}},
            },
        },
        dependency_source=dependency_source,
        current_source="public final class Test {}",
        response_region="members",
        sibling_concerns=(),
        host_symbol="Test",
    )

    payload = json.loads(messages[-1]["content"])
    calls = {
        (row["owner"], row["method"], row["arity"])
        for row in payload["dependency_call_contract"]
    }

    assert ("AuthoredStateModel", "initialize", 0) in calls
    assert ("AuthoredStateModel", "getState", 1) in calls
    assert ("AuthoredStateModel", "getState", 2) in calls
    assert ("AuthoredStateModel", "setState", 2) in calls
    assert ("AuthoredStateModel", "setState", 3) in calls
    assert payload["implementation_authority"] == ""
    assert payload["host_grounding"]["facts"] == []
    assert payload["host_grounding"]["host_version_facts"] == {}
    assert "Never add arguments to a zero-arity method" in messages[0]["content"]
    assert "Do not think aloud" in messages[0]["content"]


def test_first_pass_accepts_context_aware_state_access_contract() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "source": (
                "public final class AuthoredStateModel { "
                "public static Object getState(String name) { return null; } "
                "public static Object getState(String name, java.util.Map<String, Object> context) { return null; } "
                "public static void setState(String name, Object value) {} "
                "public static void setState(String name, Object value, java.util.Map<String, Object> context) {} "
                "}"
            ),
        }
    )
    source = (
        "private static void handle(java.util.Map<String, Object> context) { "
        'Object value = AuthoredStateModel.getState("inventory", context); '
        'AuthoredStateModel.setState("inventory", value, context); '
        "}"
    )

    _validate_first_pass_java_semantics(
        source,
        dependency_source=dependency_source,
        sibling_api=(),
    )


def test_first_pass_rejects_argument_added_to_zero_arity_dependency_hook() -> None:
    from minecraft_mod_ai.atomic_concern_source import _validate_first_pass_java_semantics

    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "path": "src/main/java/example/AuthoredStateModel.java",
            "responsibility": "host-compiled state runtime",
            "public_api": ["public static void initialize()"],
            "source": (
                "package example; public final class AuthoredStateModel { "
                "public static void initialize() {} "
                "public static Object getState(String name) { return null; } "
                "public static void setState(String name, Object value) {} "
                "}"
            ),
        }
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match=r"AuthoredStateModel\.initialize called with 1 argument.*arity is \[0\]",
    ):
        _validate_first_pass_java_semantics(
            (
                "private static void rejectInvalidInput() { "
                'AuthoredStateModel.initialize("invalid_inputs_initialization"); '
                "}"
            ),
            dependency_source=dependency_source,
            sibling_api=(),
        )


def test_host_removes_only_standalone_dependency_lifecycle_calls() -> None:
    dependency_source = json.dumps(
        {
            "symbol": "AuthoredStateModel",
            "source": (
                "public final class AuthoredStateModel { "
                "public static void initialize() {} "
                "public static Object getState(String name) { return null; } "
                "}"
            ),
        }
    )
    source = (
        "private static void run() {\n"
        "    AuthoredStateModel.initialize(\"wrong-extra-argument\");\n"
        "    Object value = AuthoredStateModel.getState(\"credits\");\n"
        "}"
    )

    repaired, changes = _strip_host_orchestrated_dependency_lifecycle_calls(
        source,
        dependency_source=dependency_source,
    )

    assert changes == ("AuthoredStateModel.initialize",)
    assert "wrong-extra-argument" not in repaired
    assert "AuthoredStateModel.getState(\"credits\")" in repaired


def test_output_budget_exhaustion_escapes_atomic_executor_for_graph_decomposition() -> None:
    def exhausted(_messages):
        raise OutputBudgetExhausted(
            "OUTPUT_BUDGET_EXHAUSTED: return to implementation decomposition"
        )

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="algorithm",
        concerns=(
            {
                "sequence": 0,
                "identifier": "feature/algorithm/steps",
                "concern": "steps",
                "task": "implement one bounded step",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=exhausted,
        compile_java=lambda _root: (_ for _ in ()).throw(
            AssertionError("compile must not run after output exhaustion")
        ),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
        region_attempt_limit=1,
        compile_repair_limit=0,
    )

    with pytest.raises(OutputBudgetExhausted):
        executor.run()
