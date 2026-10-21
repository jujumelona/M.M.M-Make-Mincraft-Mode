"""Compiler repair cannot undo the host-owned callback compiler boundary."""
import hashlib
import json

import pytest

from minecraft_mod_ai.construction_host_member_repair import repair_compiler_member

BEHAVIOR = {"behavior_id": "generic", "intent": "Increment target effects",
            "trigger": "target.active", "effects": "Increment target.effects",
            "state_reads": [], "state_writes": []}
HASH = hashlib.sha256(b"generic").hexdigest()[:20]


class RepairRouter:
    def __init__(self):
        self.calls = []

    def generate_java_source(self, *args, **kwargs):
        raise AssertionError("Callback repair must never return to free-form Java")

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((data_name, messages))
        if "_predicate_" in data_name:
            return {"expression": "target.active"}
        if "_operation_0_assign_" in data_name:
            return {"target": "target.effects", "operator": "+=", "value": "1"}
        if "_operation_0_" in data_name:
            return {"action": "assign"}
        if "_operation_1_" in data_name:
            return {"action": "complete"}
        raise AssertionError(data_name)


@pytest.mark.parametrize("role,return_type,original,expected", [
    ("applyBehavior", "boolean", "target.missing = 1;", "target.effects += 1;"),
    ("matchesBehavior", "boolean", "return target.missing;", "return target.active;"),
])
def test_failing_callback_member_is_repaired_as_operations_with_original_behavior(
    role, return_type, original, expected,
):
    source = ("package example;\npublic class Bridge {\n"
              "private int untouched() { return 17; }\n"
              f"private {return_type} {role}{HASH}(Target target) {{\n"
              f"{original}\n}}\n}}\n")
    router = RepairRouter()
    repaired = repair_compiler_member(
        router, source, "Bridge.java:5: error: cannot find symbol: missing", filename="Bridge.java",
        target_path="src/main/java/example/Bridge.java", owner="example.Bridge",
        symbols={}, api_evidence={}, attempt=1, behaviors=[BEHAVIOR])
    assert expected in repaired
    assert "private int untouched() { return 17; }" in repaired
    prompt = json.loads(router.calls[0][1][1]["content"])
    assert prompt["assigned_behavior"] == BEHAVIOR
    assert "cannot find symbol" in prompt["diagnostic"]


def test_host_dispatch_cannot_be_rewritten_as_return_only_stub():
    source = ("package example;\npublic class Bridge {\n"
              f"private boolean handleBehavior{HASH}(Target target) {{\n"
              "return target.missing;\n}\n}")
    with pytest.raises(ValueError, match="HOST_OWNED_CALLBACK"):
        repair_compiler_member(
            RepairRouter(), source, "Bridge.java:4: error: cannot find symbol", filename="Bridge.java",
            target_path="src/main/java/example/Bridge.java", owner="example.Bridge",
            symbols={}, api_evidence={}, attempt=1, behaviors=[BEHAVIOR])
