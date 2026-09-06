from __future__ import annotations

import json
import math
from typing import Any

from minecraft_mod_ai.planner_hole_filling import (
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
                                "kind": "implementation",
                                "subject": f"Summary for {hole_id}",
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


def test_multi_page_bundling_refines_complete_host_templates_once():
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
    results = fill_evidence_pages(
        router,
        [_make_skeleton("mod_alpha", ["hole_a1"]), _make_skeleton("mod_beta", ["hole_b1"])],
        valid_module_catalog={"mod_alpha", "mod_beta"},
    )

    assert len(results) == 2
    assert router.calls == 1
    fills_a = results[0]["modules"][0]["config"]["hole_fills"]
    fills_b = results[1]["modules"][0]["config"]["hole_fills"]
    assert fills_a[0]["implementation_decision"] == "Alpha choice 1"
    assert fills_b[0]["implementation_decision"] == "Beta choice 1"
    assert fills_a[0]["fill_source"] == "model_refined_host_template"
    assert fills_b[0]["fill_source"] == "model_refined_host_template"


def test_incomplete_sibling_page_keeps_host_default_without_retry():
    class IncompleteRouter:
        def __init__(self):
            self.calls = 0
            self.requested_pages_per_call = []

        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            self.calls += 1
            payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
            self.requested_pages_per_call.append(
                [page["page_id"] for page in payload["pages"]]
            )
            return (
                _page_block("mod_alpha", "hole_a1", "Alpha choice 1")
                + "\nBEGIN PAGE mod_beta\nBEGIN hole_b1\n"
                "Decision: Incomplete Beta\nEND hole_b1\nEND PAGE mod_beta"
            )

    router = IncompleteRouter()
    results = fill_evidence_pages(
        router,
        [_make_skeleton("mod_alpha", ["hole_a1"]), _make_skeleton("mod_beta", ["hole_b1"])],
        valid_module_catalog={"mod_alpha", "mod_beta"},
    )

    assert router.calls == 1
    assert router.requested_pages_per_call == [["mod_alpha", "mod_beta"]]
    fills_a = results[0]["modules"][0]["config"]["hole_fills"]
    fills_b = results[1]["modules"][0]["config"]["hole_fills"]
    assert fills_a[0]["implementation_decision"] == "Alpha choice 1"
    assert fills_a[0]["fill_source"] == "model_refined_host_template"
    assert fills_b[0]["hole_id"] == "hole_b1"
    assert fills_b[0]["fill_source"] == "host_template_default"
    assert fills_b[0]["implementation_decision"]
    assert fills_b[0]["local_steps"]
    assert fills_b[0]["verification_intent"]


def test_hallucinated_ids_are_ignored_and_required_host_holes_still_complete():
    class HallucinatingRouter:
        def __init__(self):
            self.calls = 0

        def generate_text(self, role: str, messages: list[dict[str, str]], **kwargs):
            self.calls += 1
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
    results = fill_evidence_pages(
        router,
        [_make_skeleton("mod_alpha", ["hole_a1"]), _make_skeleton("mod_beta", ["hole_b1"])],
        valid_module_catalog={"mod_alpha", "mod_beta"},
    )

    assert router.calls == 1
    all_fills = [
        fill
        for result in results
        for fill in result["modules"][0]["config"]["hole_fills"]
    ]
    assert {fill["hole_id"] for fill in all_fills} == {"hole_a1", "hole_b1"}
    assert all(fill["fill_source"] == "host_template_default" for fill in all_fills)
    assert not any(fill["hole_id"] == "invented_hole_999" for fill in all_fills)


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
            return "\n".join(
                _page_block(page["page_id"], hole["hole_id"], f"Decision for {hole['hole_id']}")
                for page in pages
                for hole in page["holes"]
            )

    total_holes = 45
    router = CountCheckingRouter()
    results = fill_evidence_pages(
        router,
        [
            _make_skeleton("mod_1", [f"h1_{i}" for i in range(15)]),
            _make_skeleton("mod_2", [f"h2_{i}" for i in range(15)]),
            _make_skeleton("mod_3", [f"h3_{i}" for i in range(15)]),
        ],
        valid_module_catalog={"mod_1", "mod_2", "mod_3"},
    )

    assert len(results) == 3
    assert router.calls == math.ceil(total_holes / _MAX_PAGE_HOLES)
    assert sum(router.hole_counts) == total_holes
    assert all(0 < count <= _MAX_PAGE_HOLES for count in router.hole_counts)
    assert all(chars <= _MAX_BUNDLE_CHARS for chars in router.prompt_chars)
    assert [
        len(result["modules"][0]["config"]["hole_fills"]) for result in results
    ] == [15, 15, 15]
