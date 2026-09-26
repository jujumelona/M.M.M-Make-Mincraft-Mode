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
