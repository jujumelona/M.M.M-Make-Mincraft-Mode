"""Prevent host-generated metadata from imposing unrelated client review gates."""
from __future__ import annotations

from minecraft_mod_ai.production_contract import compile_production_contract


def _compile(prompt: str, accessibility=()):
    return compile_production_contract(
        requested_prompt=prompt,
        game_design={
            "authored_plan": {
                "text": "Generate item and block models and localization resources.",
                "structured_sections": {
                    "resources_and_ui": {
                        "specification": {"accessibility": list(accessibility)}
                    }
                },
            },
            "_research_brief": {"summary": "3D models, images, localization"},
        },
        modules=[
            {"module_id": "crystal_fragment", "kind": "item", "config": {},
             "depends_on": [], "required_gates": []},
            {"module_id": "crystal_block", "kind": "block", "config": {},
             "depends_on": [], "required_gates": []},
        ],
        assets=[],
        acceptance_tests=["Generated item and block are present."],
    )


def test_default_item_block_resources_do_not_require_client_visual_review():
    contract = _compile("Add a crystal_fragment item and crystal_block with resources.").contract
    dimensions = {x["dimension_id"] for x in contract["quality_dimension_catalog"]}
    assert "visual_3d" not in dimensions
    assert "accessibility" not in dimensions
    assert {"correctness", "build", "runtime"} <= dimensions


def test_explicit_visual_and_accessibility_request_remains_fail_closed():
    contract = _compile(
        "Add a crystal item; review its visual 3D rendering and accessibility."
    ).contract
    dimensions = {x["dimension_id"] for x in contract["quality_dimension_catalog"]}
    assert "visual_3d" in dimensions
    assert "accessibility" in dimensions


def test_explicit_authored_accessibility_path_remains_required():
    contract = _compile(
        "Add a crystal item.",
        accessibility=[{"path": "inventory", "check": "keyboard"}],
    ).contract
    dimensions = {x["dimension_id"] for x in contract["quality_dimension_catalog"]}
    assert "accessibility" in dimensions


def test_unbound_research_does_not_block_basic_content_build() -> None:
    contract = _compile("Add a crystal fragment and block.").contract
    dimensions = {x["dimension_id"] for x in contract["quality_dimension_catalog"]}
    assert "research" not in dimensions


def test_explicit_research_remains_mandatory() -> None:
    contract = _compile("Research prior art, then add a crystal item.").contract
    dimensions = {x["dimension_id"] for x in contract["quality_dimension_catalog"]}
    assert "research" in dimensions
