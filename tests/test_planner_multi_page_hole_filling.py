from __future__ import annotations

import json
import math
from typing import Any

import pytest

from minecraft_mod_ai.planner_hole_filling import (
    PlanningHoleFillError,
    _MAX_BUNDLE_CHARS,
    _MAX_PAGE_HOLES,
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
                                "hole_id": hole_id,
                                "summary": f"Summary for {hole_id}",
                                "required": True,
                            }
                            for hole_id in hole_ids
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
        "Steps:\n"
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
            page_ids = [page["page_id"] for page in payload["pages"]]
            self.requested_pages_per_call.append(page_ids)

            if self.calls == 1:
                return (
                    _page_block("mod_alpha", "hole_a1", "Alpha choice 1")
                    + "\n"
                    + "BEGIN PAGE mod_beta\nBEGIN hole_b1\nDecision: Incomplete Beta\nEND hole_b1\nEND PAGE mod_beta"
                )
            if self.calls == 2:
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

    with pytest.raises(
        PlanningHoleFillError,
        match="Host template references unknown or unfilled mandatory holes",
    ):
        fill_evidence_pages(
            router,
            [skel_a, skel_b],
            valid_module_catalog={"mod_alpha", "mod_beta"},
        )


def test_budget_partitioning_by_hole_count_and_prompt_size():
    class CountCheckingRouter:
        def __init__(self):
            self.calls = 0
            self.hole_counts = []
            self.prompt_chars = []

        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            self.calls += 1
            payload_text = messages[-1]["content"].split("\n", 1)[1]
            payload = json.loads(payload_text)
            pages = payload["pages"]
            holes = [hole for page in pages for hole in page["holes"]]
            self.hole_counts.append(len(holes))
            self.prompt_chars.append(len(payload_text))
            response = []
            for page in pages:
                page_id = page["page_id"]
                for hole in page["holes"]:
                    hole_id = hole["hole_id"]
                    response.append(
                        _page_block(page_id, hole_id, f"Decision for {hole_id}")
                    )
            return "\n".join(response)

    total_holes = 45
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
    assert router.calls == math.ceil(total_holes / _MAX_PAGE_HOLES)
    assert sum(router.hole_counts) == total_holes
    assert all(0 < count <= _MAX_PAGE_HOLES for count in router.hole_counts)
    assert all(chars <= _MAX_BUNDLE_CHARS for chars in router.prompt_chars)

    assert len(results[0]["modules"][0]["config"]["hole_fills"]) == 15
    assert len(results[1]["modules"][0]["config"]["hole_fills"]) == 15
    assert len(results[2]["modules"][0]["config"]["hole_fills"]) == 15
