from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.generation_compile_state import verify_compile_backed_java


class _Runtime:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = []

    def call(self, stage, name, arguments):
        self.calls.append((stage, name, arguments))
        return self.receipt


class _State:
    def __init__(self, target_path="src/main/java/demo/Test.java"):
        self.mutation_context = SimpleNamespace(target_path=target_path)
        self.records = []

    def record_verification(self, name, payload, status):
        self.records.append((name, payload, status))
        return True


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("PASS", "PASS"),
        ("FAIL", "FAIL"),
        ("UNAVAILABLE", "UNAVAILABLE"),
        ("DEFERRED", "UNAVAILABLE"),
        ("TIMEOUT", "UNAVAILABLE"),
    ],
)
def test_compile_state_maps_target_compile_receipt_status(raw_status, expected) -> None:
    receipt = {
        "status": raw_status,
        "reason": raw_status.lower(),
        "diagnostics": [],
    }
    runtime = _Runtime(receipt)
    state = _State()

    status, returned = verify_compile_backed_java(
        runtime,
        state,
        stage="generation",
    )

    assert status == expected
    assert returned is receipt
    assert runtime.calls == [
        (
            "generation",
            "target_compile",
            {"target_path": "src/main/java/demo/Test.java"},
        )
    ]
    assert state.records[-1][0] == "target_compile"
    assert state.records[-1][2] == expected


def test_compile_state_rejects_missing_target_path() -> None:
    with pytest.raises(RuntimeError, match="no pinned target path"):
        verify_compile_backed_java(
            _Runtime({"status": "PASS"}),
            _State(target_path=""),
            stage="generation",
        )


def test_compile_state_rejects_non_mapping_receipt() -> None:
    with pytest.raises(RuntimeError, match="non-mapping receipt"):
        verify_compile_backed_java(
            _Runtime("PASS"),
            _State(),
            stage="generation",
        )
