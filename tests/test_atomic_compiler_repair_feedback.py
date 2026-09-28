from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor


def _executor(outputs: list[str], captured: list[list[dict[str, str]]]) -> AtomicConcernExecutor:
    remaining = list(outputs)
    written = {"source": ""}

    def call_coder(messages):
        captured.append(list(messages))
        if not remaining:
            raise AssertionError("unexpected extra model call")
        return remaining.pop(0)

    def write_source(_path, source):
        written["source"] = source

    def compile_java(_root):
        source = written["source"]
        if "private static int broken = 1;" in source:
            return SimpleNamespace(status="PASS", error="")
        failing_line = next(
            index
            for index, line in enumerate(source.splitlines(), start=1)
            if "private static int broken" in line
        )
        failing_source = source.splitlines()[failing_line - 1]
        return SimpleNamespace(
            status="FAIL",
            error=(
                f"/tmp/Test.java:{failing_line}: error: cannot find symbol\n"
                f"{failing_source}\n"
                "                              ^\n"
                "  symbol:   variable Missing\n"
                "  location: class Test\n"
            ),
        )

    return AtomicConcernExecutor(
        root=Path("."),
        target=Path("src/main/java/example/Test.java"),
        relative="src/main/java/example/Test.java",
        symbol="Test",
        original="package example;\n// MMM_AUTHORED_FEATURE_BODY\n",
        task={"task_id": "repair", "semantic_outcome": "compile"},
        section="behavior_contract",
        concerns=(
            {
                "sequence": 0,
                "identifier": "repair",
                "concern": "concurrency",
                "task": "implement concurrency behavior",
                "rules": [],
            },
        ),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=call_coder,
        compile_java=compile_java,
        compile_log=lambda report: report.error,
        write_source=write_source,
    )


def test_compiler_feedback_is_sent_with_exact_failing_source_and_retried_after_progress() -> None:
    captured: list[list[dict[str, str]]] = []
    executor = _executor(
        [
            "private static int broken = Missing.one();",
            "private static int broken = Missing.two();",
            "private static int broken = 1;",
        ],
        captured,
    )

    result = executor.run()

    assert result["repair_count"] == 2
    assert "private static int broken = 1;" in result["source"]
    assert len(captured) == 3

    first_repair = json.loads(captured[1][-1]["content"])
    second_repair = json.loads(captured[2][-1]["content"])

    first_failure = first_repair["repair_failure"]
    second_failure = second_repair["repair_failure"]
    assert "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE" in first_failure
    assert "error: cannot find symbol" in first_failure
    assert "symbol:   variable Missing" in first_failure
    assert ">>" in first_failure
    assert "private static int broken = Missing.one();" in first_failure
    assert "private static int broken = Missing.two();" in second_failure

    assert (
        first_repair["current_selected_region_source"]
        == "private static int broken = Missing.one();"
    )
    assert (
        second_repair["current_selected_region_source"]
        == "private static int broken = Missing.two();"
    )


def test_first_pass_prompt_prioritizes_compilation_and_canonical_jdk_packages() -> None:
    captured: list[list[dict[str, str]]] = []
    executor = _executor(["private static int broken = 1;"], captured)

    executor.run()

    payload = json.loads(captured[0][-1]["content"])
    recipe = payload["generation_recipe"]
    rules = "\n".join(recipe["compiler_first_rules"])

    assert "first answer must compile" in rules
    assert "Never guess a package or fully-qualified class name" in rules
    assert recipe["jdk_package_anchors"]["locks"] == "java.util.concurrent.locks"
    assert recipe["jdk_package_anchors"]["atomics"] == "java.util.concurrent.atomic"
