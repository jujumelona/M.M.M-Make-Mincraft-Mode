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
