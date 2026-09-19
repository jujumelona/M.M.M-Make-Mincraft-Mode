from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_orchestrator_services as services
from minecraft_mod_ai.complete_orchestrator_support import CompleteProductionError


class _Bridge:
    ACTIONS = frozenset(
        {
            "connect",
            "disconnect",
            "walk_to",
            "wait_for",
            "inventory",
            "status",
        }
    )

    def call(self, action, **params):
        if action == "wait_for":
            return {"matched": True, "observed": params}
        return {"ok": True, "action": action, "params": params}

    def close(self):
        return None


def test_playtest_wait_for_must_cover_every_approved_acceptance_test(monkeypatch) -> None:
    monkeypatch.setattr(services, "MineflayerBridge", _Bridge)
    tests = (
        "Player receives the debug token",
        "Token remains usable after interaction",
    )
    actions = (
        {"action": "walk_to", "params": {"x": 1, "y": 64, "z": 1}},
        {
            "action": "wait_for",
            "acceptance_test": tests[0],
            "params": {"condition": "inventory_has_token"},
        },
        {
            "action": "wait_for",
            "acceptance_test": tests[1],
            "params": {"condition": "token_use_succeeds"},
        },
    )

    receipt = services.run_playtest(actions, tests)

    assert receipt["status"] == "PASS"
    assert receipt["acceptance_tests"] == list(tests)
    assert receipt["covered_acceptance_tests"] == list(tests)
    assert {item["test"] for item in receipt["acceptance_test_results"]} == set(tests)


def test_playtest_rejects_missing_acceptance_test_coverage(monkeypatch) -> None:
    monkeypatch.setattr(services, "MineflayerBridge", _Bridge)
    tests = ("first", "second")
    actions = (
        {"action": "walk_to", "params": {"x": 1, "y": 64, "z": 1}},
        {
            "action": "wait_for",
            "acceptance_test": "first",
            "params": {"condition": "ok"},
        },
    )

    with pytest.raises(CompleteProductionError, match="does not cover every approved"):
        services.run_playtest(actions, tests)


def test_playtest_rejects_unapproved_acceptance_test_reference(monkeypatch) -> None:
    monkeypatch.setattr(services, "MineflayerBridge", _Bridge)
    actions = (
        {"action": "walk_to", "params": {"x": 1, "y": 64, "z": 1}},
        {
            "action": "wait_for",
            "acceptance_test": "not-approved",
            "params": {"condition": "ok"},
        },
    )

    with pytest.raises(CompleteProductionError, match="bind to one approved"):
        services.run_playtest(actions, ("approved",))
