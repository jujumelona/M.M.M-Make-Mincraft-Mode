from __future__ import annotations

import json
from typing import Any

import pytest

from minecraft_mod_ai.planner_hole_filling import (
    PlanningHoleFillError,
    fill_evidence_pages,
)


def _make_skeleton(module_id: str, hole_ids: list[str]) -> dict[str, Any]:
    return {
        "modules": [
            {
                "module_id": module_id,
                "config": {
                    "scope": f"Scope for {module_id}",
                    "implementation_template": {
                        "schema_version": "mmm/planner-template-v1",
                        "semantic_outcome": f"Outcome for {module_id}",
                        "holes": [
                            {
                                "hole_id": hid,
                                "summary": f"Summary for {hid}",
                                "required": True,
                            }
                            for hid in hole_ids
                        ],
                        "completion_policy": {
                            "required_hole_ids": list(hole_ids),
                        },
                    },
                },
            }
        ],
        "assets": [],
        "acceptance_tests": [],
        "completed_deliverables": [],
    }


def _page_block(page_id: str, hole_id: str, decision: str) -> str:
    return (
        f"BEGIN PAGE {page_id}\n"
        f"BEGIN {hole_id}\n"
        f"Decision: {decision}\n"
        f"Steps:\n"
        f"- Execute step for {hole_id}\n"
        f"Verification: Verify {hole_id}\n"
        f"END {hole_id}\n"
        f"END PAGE {page_id}\n"
    )


def test_multi_page_bundling_concurrent_expansion():
    class MultiPageRouter:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            self.calls += 1
            self.prompts.append(messages)
            assert kwargs == {"response_format": "text", "enable_tools": False}
            return (
                _page_block("mod_alpha", "hole_a1", "Alpha choice 1")
                + "\n"
                + _page_block("mod_beta", "hole_b1", "Beta choice 1")
            )

    router = MultiPageRouter()
    skel_a = _make_skeleton("mod_alpha", ["hole_a1"])
    skel_b = _make_skeleton("mod_beta", ["hole_b1"])

    results = fill_evidence_pages(
        router,
        [skel_a, skel_b],
        valid_module_catalog={"mod_alpha", "mod_beta"},
    )

    assert len(results) == 2
    assert router.calls == 1

    mod_a_config = results[0]["modules"][0]["config"]
    fills_a = mod_a_config["hole_fills"]
    assert len(fills_a) == 1
    assert fills_a[0]["hole_id"] == "hole_a1"
    assert fills_a[0]["implementation_decision"] == "Alpha choice 1"

    mod_b_config = results[1]["modules"][0]["config"]
    fills_b = mod_b_config["hole_fills"]
    assert len(fills_b) == 1
    assert fills_b[0]["hole_id"] == "hole_b1"
    assert fills_b[0]["implementation_decision"] == "Beta choice 1"


def test_isolated_sibling_page_failure_handling():
    class IsolatedRetryRouter:
        def __init__(self):
            self.calls = 0
            self.requested_pages_per_call = []

        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            self.calls += 1
            content = messages[-1]["content"]
            payload = json.loads(content.split("\n", 1)[1])
            page_ids = [p["page_id"] for p in payload["pages"]]
            self.requested_pages_per_call.append(page_ids)

            if self.calls == 1:
                # Attempt 1: mod_alpha succeeds, mod_beta is malformed/incomplete
                return (
                    _page_block("mod_alpha", "hole_a1", "Alpha choice 1")
                    + "\n"
                    + "BEGIN PAGE mod_beta\nBEGIN hole_b1\nDecision: Incomplete Beta\nEND hole_b1\nEND PAGE mod_beta"
                )
            elif self.calls == 2:
                # Attempt 2: mod_beta is repaired
                return _page_block("mod_beta", "hole_b1", "Beta choice repaired")
            raise RuntimeError("Unexpected 3rd call")

    router = IsolatedRetryRouter()
    skel_a = _make_skeleton("mod_alpha", ["hole_a1"])
    skel_b = _make_skeleton("mod_beta", ["hole_b1"])

    results = fill_evidence_pages(
        router,
        [skel_a, skel_b],
        valid_module_catalog={"mod_alpha", "mod_beta"},
    )

    assert router.calls == 2
    assert router.requested_pages_per_call[0] == ["mod_alpha", "mod_beta"]
    # Sibling isolation: Second call ONLY requested the failed page (mod_beta)
    assert router.requested_pages_per_call[1] == ["mod_beta"]

    fills_a = results[0]["modules"][0]["config"]["hole_fills"]
    assert fills_a[0]["hole_id"] == "hole_a1"
    assert fills_a[0]["implementation_decision"] == "Alpha choice 1"

    fills_b = results[1]["modules"][0]["config"]["hole_fills"]
    assert fills_b[0]["hole_id"] == "hole_b1"
    assert fills_b[0]["implementation_decision"] == "Beta choice repaired"


def test_host_owned_id_enforcement_and_unfilled_mandatory_error():
    class HallucinatingRouter:
        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            return (
                "BEGIN PAGE mod_alpha\n"
                "BEGIN invented_hole_999\n"
                "Decision: Hallucinated choice\n"
                "Steps:\n- fake step\n"
                "Verification: fake verify\n"
                "END invented_hole_999\n"
                "END PAGE mod_alpha"
            )

    router = HallucinatingRouter()
    skel_a = _make_skeleton("mod_alpha", ["hole_a1"])
    skel_b = _make_skeleton("mod_beta", ["hole_b1"])

    with pytest.raises(PlanningHoleFillError, match="Host template references unknown or unfilled mandatory holes"):
        fill_evidence_pages(
            router,
            [skel_a, skel_b],
            valid_module_catalog={"mod_alpha", "mod_beta"},
        )


def test_budget_partitioning_by_hole_count():
    class CountCheckingRouter:
        def __init__(self):
            self.calls = 0
            self.page_counts = []

        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            self.calls += 1
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            self.page_counts.append(len(payload["pages"]))
            response = []
            for page in payload["pages"]:
                pid = page["page_id"]
                for h in page["holes"]:
                    hid = h["hole_id"]
                    response.append(_page_block(pid, hid, f"Decision for {hid}"))
            return "\n".join(response)

    # 3 skeletons, each with 15 holes -> 45 holes total
    # _MAX_PAGE_HOLES is 32, so 15 + 15 = 30 fits in bundle 1, 3rd skeleton (15 holes) goes to bundle 2
    skel1 = _make_skeleton("mod_1", [f"h1_{i}" for i in range(15)])
    skel2 = _make_skeleton("mod_2", [f"h2_{i}" for i in range(15)])
    skel3 = _make_skeleton("mod_3", [f"h3_{i}" for i in range(15)])

    router = CountCheckingRouter()
    results = fill_evidence_pages(
        router,
        [skel1, skel2, skel3],
        valid_module_catalog={"mod_1", "mod_2", "mod_3"},
    )

    assert len(results) == 3
    assert router.calls == 2
    assert router.page_counts == [2, 1]

    assert len(results[0]["modules"][0]["config"]["hole_fills"]) == 15
    assert len(results[1]["modules"][0]["config"]["hole_fills"]) == 15
    assert len(results[2]["modules"][0]["config"]["hole_fills"]) == 15
