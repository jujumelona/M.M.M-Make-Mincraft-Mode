"""Feature-level unit of work must preserve target and required gameplay."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.feature_work_order import FeatureWorkOrder


def _target(version="26.2", loader="fabric"):
    return {"minecraft_version": version, "loader": loader, "java_version": 25}


def _behaviors():
    return [
        {"behavior_id": "alien.spawn", "requirement_id": "req:aliens",
         "intent": "spawn hostile alien entity", "effects": "spawn real enemy",
         "state_reads": ["alien_population"], "state_writes": ["alien_population"]},
        {"behavior_id": "alien.combat", "requirement_id": "req:aliens",
         "intent": "attack nearest player", "effects": "deal damage after cooldown",
         "state_reads": ["alien_population", "cooldown"], "state_writes": ["cooldown"]},
    ]


def _order(**kwargs):
    return FeatureWorkOrder.for_feature(
        requirement_id="req:aliens", target=_target(),
        target_path="src/main/java/test/AlienMod.java", behaviors=_behaviors(),
        **kwargs,
    )


def test_work_order_preserves_all_behaviors_state_and_acceptance():
    order = _order()
    data = order.to_dict()
    assert data["target"]["minecraft_version"] == "26.2"
    assert data["target"]["loader"] == "fabric"
    assert data["behavior_ids"] == ["alien.spawn", "alien.combat"]
    assert data["state_reads"] == ["alien_population", "cooldown"]
    assert data["state_writes"] == ["alien_population", "cooldown"]
    assert data["acceptance"]["compile"] == "real_gradle_compileJava"
    assert data["acceptance"]["max_compiler_repair_attempts"] == 2
    assert order.fingerprint() == _order().fingerprint()


def test_target_binding_invalidates_old_work_order():
    baseline = _order()
    changed_version = FeatureWorkOrder.for_feature(
        requirement_id="req:aliens", target=_target("1.21.8"),
        target_path=baseline.target_path, behaviors=_behaviors())
    changed_loader = FeatureWorkOrder.for_feature(
        requirement_id="req:aliens", target=_target("26.2", "neoforge"),
        target_path=baseline.target_path, behaviors=_behaviors())
    assert baseline.fingerprint() != changed_version.fingerprint()
    assert baseline.fingerprint() != changed_loader.fingerprint()


def test_external_donor_routes_are_not_accepted():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _order(route="source_reuse", evidence_id="legacy-donor")


def test_duplicate_or_missing_behavior_is_never_dropped():
    behaviors = _behaviors()
    with pytest.raises(ValueError, match="BEHAVIOR_DUPLICATE"):
        FeatureWorkOrder.for_feature(
            requirement_id="req:aliens", target=_target(),
            target_path="src/main/java/test/AlienMod.java",
            behaviors=[behaviors[0], behaviors[0]],
        )
    with pytest.raises(ValueError, match="SCOPE_MISMATCH"):
        FeatureWorkOrder.for_feature(
            requirement_id="req:aliens", target=_target(),
            target_path="src/main/java/test/AlienMod.java",
            behaviors=[{**behaviors[0], "requirement_id": "req:other"}],
        )


def test_rejects_unsafe_output_path():
    with pytest.raises(ValueError, match="PATH_INVALID"):
        FeatureWorkOrder.for_feature(
            requirement_id="req:aliens", target=_target(),
            target_path="../outside.java", behaviors=_behaviors(),
        )
