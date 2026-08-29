from pathlib import Path

from minecraft_mod_ai.planner_template_schema import (
    PRODUCTION_PAGE_TEMPLATE,
    build_batch_skeleton,
    merge_model_output_into_skeleton,
)


def test_host_owned_schema_drops_unknown_fields():
    skeleton = build_batch_skeleton(
        "custom_features", "custom", ["feature"], ["custom_features"]
    )
    merged = merge_model_output_into_skeleton(
        skeleton,
        {
            "modules": [
                {
                    "module_id": "custom_features",
                    "kind": "made_up",
                    "config": {},
                    "depends_on": [],
                    "required_gates": [],
                    "unknown": 1,
                }
            ],
            "unknown_top": True,
        },
        set(),
    )
    assert set(merged) == {
        "modules",
        "assets",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
    assert set(merged["modules"][0]) == {
        "module_id",
        "kind",
        "config",
        "depends_on",
        "required_gates",
    }
    assert merged["modules"][0]["kind"] == "custom_java"


def test_template_surface_is_closed_and_host_owned():
    assert set(PRODUCTION_PAGE_TEMPLATE) == {
        "modules",
        "assets",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
    assert set(PRODUCTION_PAGE_TEMPLATE["modules"][0]) == {
        "module_id",
        "kind",
        "config",
        "depends_on",
        "required_gates",
    }


def test_planner_has_only_the_production_batch_outline_path():
    source = (
        Path(__file__).resolve().parents[1]
        / "minecraft_mod_ai"
        / "complete_planner.py"
    ).read_text(encoding="utf-8")
    assert "build_batch_skeleton" in source
    assert "merge_model_output_into_skeleton" not in source
    assert "page = skeleton" in source
    assert "def _expand_" + "batches" in source
    assert "def _expand_" + "one_batch" not in source
    assert "module_" + "batches" not in source
