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


@pytest.mark.parametrize(
    "wrapped",
    [
        lambda body: "Here is the implementation:\n" + body + "\nDone.",
        lambda body: "```java\n" + body + "\n```",
        lambda body: "\n\n" + body + "\n\n",
    ],
)
def test_inert_outer_wrapper_is_ignored(wrapped) -> None:
    members, initialize = parse_concern_content(
        wrapped(_response("private static final int COST = 10;", "")),
        section="behavior_contract",
    )
    assert members == "private static final int COST = 10;"
    assert initialize == ""


def test_inert_inner_wrappers_and_host_marker_echo_are_normalized() -> None:
    members, initialize = parse_concern_content(
        _response(
            "\n".join(
                [
                    "```java",
                    "Members:",
                    "// MMM_ATOMIC_CONCERN_COST_MEMBERS_START",
                    "private static final int COST = 10;",
                    "// MMM_ATOMIC_CONCERN_COST_MEMBERS_END",
                    "```",
                ]
            ),
            "// no initialization needed",
        ),
        section="behavior_contract",
    )
    assert members == "private static final int COST = 10;"
    assert initialize == ""


def test_marker_mentions_in_prose_do_not_break_exact_marker_line_protocol() -> None:
    text = (
        f"Use {MEMBERS_MARKER} then {END_MARKER} exactly.\n"
        + _response("private static final int COST = 10;", "")
    )
    members, initialize = parse_concern_content(text, section="behavior_contract")
    assert members == "private static final int COST = 10;"
    assert initialize == ""


def test_duplicate_exact_marker_lines_and_wrong_order_remain_fail_closed() -> None:
    duplicated = _response("private static final int COST = 10;", "") + "\n" + END_MARKER
    with pytest.raises(CustomModuleGenerationError, match="exactly once as a marker line"):
        parse_concern_content(duplicated, section="behavior_contract")

    out_of_order = f"{INITIALIZE_MARKER}\n{MEMBERS_MARKER}\n{END_MARKER}"
    with pytest.raises(CustomModuleGenerationError, match="out of order"):
        parse_concern_content(out_of_order, section="behavior_contract")


@pytest.mark.parametrize(
    "members",
    [
        "package example;",
        "import net.minecraft.Foo;",
        "public class Escape {}",
        "public static void initialize() {}",
    ],
)
def test_real_scope_escape_is_still_rejected(members: str) -> None:
    with pytest.raises(CustomModuleGenerationError):
        parse_concern_content(_response(members, ""), section="behavior_contract")


def test_scope_words_inside_comments_and_literals_are_not_false_positives() -> None:
    members, _ = parse_concern_content(
        _response(
            '// class import package are words only\n'
            'private static final String NOTE = "class import package";',
            "",
        ),
        section="behavior_contract",
    )
    assert "NOTE" in members


def test_non_integration_real_initialize_code_is_still_rejected() -> None:
    with pytest.raises(CustomModuleGenerationError, match="only integration concerns"):
        parse_concern_content(
            _response("", "register();"),
            section="behavior_contract",
        )


def _executor(outputs: list[str]) -> AtomicConcernExecutor:
    remaining = list(outputs)

    def call_coder(_messages):
        return remaining.pop(0)

    return AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "t", "semantic_outcome": "x"},
        section="behavior_contract",
        concerns=(
            {
                "sequence": 0,
                "identifier": "id",
                "concern": "cost",
                "task": "implement cost",
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


def test_response_contract_failure_is_repaired_locally_before_pipeline_abort() -> None:
    executor = _executor(
        [
            _response("package bad;", ""),
            _response("private static final int COST = 10;", ""),
        ]
    )
    result = executor.run()
    assert "private static final int COST = 10;" in result["source"]


def test_repeated_same_response_violation_stops_on_no_progress() -> None:
    executor = _executor([_response("package bad;", ""), _response("package worse;", "")])
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_CONCERN_RESPONSE_NO_PROGRESS"):
        executor.run()
