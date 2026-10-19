"""Compiler-sourced member-only repair must preserve successful Java bodies."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.construction_host_member_repair import (
    _failing_member, repair_compiler_member,
)


SOURCE = """package example;
public class Demo {
    public int ready() {
        return 7;
    }
    public int broken() {
        return missingSymbol;
    }
}
"""
ERROR = "/tmp/sources/example/Demo.java:7: error: cannot find symbol"


class Router:
    def __init__(self):
        self.calls = []

    def generate_java_source(self, role, messages, *, data_name, source_frame):
        self.calls.append((role, messages, data_name, source_frame))
        return source_frame["prefix"] + "    return 9;" + source_frame["suffix"]


def test_repair_changes_only_failing_member():
    router = Router()
    new = repair_compiler_member(
        router, SOURCE, ERROR, filename="Demo.java",
        target_path="src/main/java/example/Demo.java",
        symbols={}, owner="example.Demo", api_evidence={}, attempt=1)
    assert "return 7;" in new
    assert "return 9;" in new
    assert "missingSymbol" not in new
    assert new.startswith("package example;")
    assert len(router.calls) == 1
    assert router.calls[0][3]["kind"] == "method"
    assert router.calls[0][3]["prefix"].startswith("public int broken()")


def test_non_member_compiler_error_does_not_trigger_regeneration():
    router = Router()
    with pytest.raises(ValueError, match="JAVA_HOST_REPAIR_NOT_A_MEMBER_DIAGNOSTIC"):
        repair_compiler_member(
            router, SOURCE, "Demo.java:2: error: wrong superclass",
            filename="Demo.java", target_path="src/main/java/example/Demo.java",
            symbols={}, owner="example.Demo", api_evidence={}, attempt=1)
    assert router.calls == []


def test_without_real_compiler_line_does_not_guess_failing_method():
    with pytest.raises(ValueError, match="JAVA_HOST_REPAIR_NO_SOURCE_LINE"):
        _failing_member(SOURCE, "generic backend failed", filename="Demo.java")
