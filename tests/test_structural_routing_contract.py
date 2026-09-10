from minecraft_mod_ai.structural_routing_contract import build_artifact_plan


def test_feature_labels_cannot_change_artifact_selection():
    structural = {"explicit_artifacts": ["entity"], "networking": True, "persistent_state": True}
    left = build_artifact_plan({**structural, "feature_id": "alpha", "description": "boss economy dungeon"})
    right = build_artifact_plan({**structural, "feature_id": "renamed", "description": "skill trade unrelated"})
    assert left == right
