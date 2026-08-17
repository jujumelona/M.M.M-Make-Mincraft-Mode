from minecraft_mod_ai.planner_template_schema import build_batch_skeleton, merge_model_output_into_skeleton

def test_host_owned_schema_drops_unknown_fields():
    skeleton = build_batch_skeleton("custom_features", "custom", ["feature"], ["custom_features"])
    merged = merge_model_output_into_skeleton(skeleton, {"modules": [{"module_id": "custom_features", "kind": "made_up", "config": {}, "depends_on": [], "required_gates": [], "unknown": 1}], "unknown_top": True}, set())
    assert set(merged) == {"modules", "assets", "acceptance_tests", "completed_deliverables", "complete", "next_cursor"}
    assert set(merged["modules"][0]) == {"module_id", "kind", "config", "depends_on", "required_gates"}
    assert merged["modules"][0]["kind"] == "custom_java"
