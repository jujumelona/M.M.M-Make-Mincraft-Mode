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


def test_javac_parent_constructor_is_fixed_from_required_types_without_model():
    source = """package example;
public class AlienCombatHostileEntity extends net.minecraft.world.entity.monster.Monster {
    public AlienCombatHostileEntity(net.minecraft.world.entity.EntityType type,
                                   net.minecraft.world.level.Level level) {
        this.type = type;
    }
    public int ready() { return 7; }
}
"""
    diagnostic = """/tmp/src/AlienCombatHostileEntity.java:3: error: constructor Monster in class Monster cannot be applied to given types;
  required: EntityType<? extends Monster>,Level
  found: no arguments
/tmp/src/AlienCombatHostileEntity.java:5: error: type has private access in Entity
this.type = type;"""
    router = Router()
    revised = repair_compiler_member(
        router, source, diagnostic, filename="AlienCombatHostileEntity.java",
        target_path="src/main/java/example/AlienCombatHostileEntity.java",
        symbols={}, owner="example.AlienCombatHostileEntity", api_evidence={}, attempt=1)
    assert "super(type, level);" in revised
    assert "this.type = type;" not in revised
    assert "public int ready() { return 7; }" in revised
    assert router.calls == []


def test_constructor_repair_refuses_unproved_argument_type_or_parent():
    from minecraft_mod_ai.construction_host_member_repair import _javac_constructor_repair
    source = """package example;
public class Demo extends Base {
    public Demo(java.lang.String name) { }
}
"""
    diagnostic = """Demo.java:3: error: constructor Base in class Base cannot be applied to given types;
  required: int
  found: no arguments"""
    assert _javac_constructor_repair(
        source, diagnostic, filename="Demo.java") is None
    assert _javac_constructor_repair(
        source.replace("extends Base", "extends Other"),
        diagnostic, filename="Demo.java") is None


def test_multidiagnostic_targets_first_error_not_shortest_method():
    # Regression from production-logs (13): the constructor failure came
    # before a private-field failure inside a tiny unrelated method.
    source = """package example;
public class Alien extends net.minecraft.world.entity.monster.Monster {
    public Alien(net.minecraft.world.entity.EntityType type,
                 net.minecraft.world.level.Level level) {
        this.type = type;
    }
    public void expose() { this.type = type; }
}
"""
    diagnostic = """Alien.java:3: error: constructor Monster in class Monster cannot be applied to given types;
  required: EntityType<? extends Monster>,Level
  found: no arguments
Alien.java:7: error: type has private access in Entity"""
    _raw, node, original, _frame = _failing_member(
        source, diagnostic, filename="Alien.java")
    assert node.type == "constructor_declaration"
    assert "public Alien(" in original
    router = Router()
    updated = repair_compiler_member(
        router, source, diagnostic, filename="Alien.java",
        target_path="src/main/java/example/Alien.java",
        symbols={}, owner="example.Alien", api_evidence={}, attempt=1)
    assert "super(type, level);" in updated
    assert "this.type = type;" not in updated.split("public void expose")[0]
    # Still invalid in a different method: the next compile should report
    # it rather than silently claiming this class was fully fixed.
    assert "public void expose() { this.type = type; }" in updated
    assert router.calls == []


def test_coder_receives_only_error_for_selected_member():
    import json
    diagnostic = """/tmp/example/Demo.java:7: error: cannot find symbol
        return missingSymbol;
               ^
  symbol: variable missingSymbol
/tmp/example/Demo.java:4: error: incompatible types
        return 7;
               ^"""
    router = Router()
    result = repair_compiler_member(
        router, SOURCE, diagnostic, filename="Demo.java",
        target_path="src/main/java/example/Demo.java",
        symbols={}, owner="example.Demo", api_evidence={}, attempt=1)
    assert "return 9;" in result
    payload = json.loads(router.calls[0][1][-1]["content"])
    assert "missingSymbol" in payload["javac_diagnostic"]
    assert "incompatible types" not in payload["javac_diagnostic"]
    assert "return missingSymbol;" in payload["current_member"]
    assert any("public int ready()" in sig for sig in payload["host_declared_neighbor_signatures"])
    assert not any("public int broken()" in sig for sig in payload["host_declared_neighbor_signatures"])


def test_identical_first_compiler_repair_gets_different_focused_retry():
    import json

    class RepeatedFirstRouter:
        def __init__(self):
            self.calls = []

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            self.calls.append((messages, data_name, source_frame))
            if len(self.calls) == 1:
                return _failing_member(SOURCE, ERROR, filename="Demo.java")[2]
            return source_frame["prefix"] + "return 9;" + source_frame["suffix"]

    router = RepeatedFirstRouter()
    revised = repair_compiler_member(
        router, SOURCE, ERROR, filename="Demo.java",
        target_path="src/main/java/example/Demo.java",
        symbols={}, owner="example.Demo", api_evidence={}, attempt=1,
    )
    assert "return 9;" in revised
    assert "missingSymbol" not in revised
    assert len(router.calls) == 2
    assert "no_progress" in router.calls[1][1]
    selected = json.loads(router.calls[1][0][-1]["content"])
    assert selected["javac_diagnostic"]
    assert "return missingSymbol;" in selected["rejected_member"]
    assert selected["immutable_source_frame"] == router.calls[1][2]
    assert any("public int ready()" in sig
               for sig in selected["host_declared_neighbor_signatures"])


def test_identical_second_compiler_repair_still_fails():
    class NoProgress:
        calls = 0

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            self.calls += 1
            return _failing_member(SOURCE, ERROR, filename="Demo.java")[2]

    router = NoProgress()
    with pytest.raises(ValueError, match="JAVA_HOST_REPAIR_NO_MEMBER_PROGRESS"):
        repair_compiler_member(
            router, SOURCE, ERROR, filename="Demo.java",
            target_path="src/main/java/example/Demo.java",
            symbols={}, owner="example.Demo", api_evidence={}, attempt=1,
        )
    assert router.calls == 2
