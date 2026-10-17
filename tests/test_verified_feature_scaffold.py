"""Scaffold must be derived from selected inspected Fabric symbols, not a guessed API."""
import pytest
from minecraft_mod_ai.verified_feature_scaffold import verified_fabric_feature_scaffold
from minecraft_mod_ai.feature_work_order import FeatureWorkOrder


def _work(version="26.2"):
    return FeatureWorkOrder.for_feature(
        requirement_id="req:alien",
        target={"minecraft_version": version, "loader": "fabric", "java_version": 25},
        target_path="src/main/java/demo/AlienMod.java",
        behaviors=[{"requirement_id": "req:alien", "behavior_id": "alien.spawn",
                    "intent": "register alien entity", "effects": "make entity type available"}],
    )


def _symbols():
    return {
        "net.fabricmc.api.ModInitializer": {"kind": "INTERFACE", "source_hash": "hash-interface"},
        "net.fabricmc.api.ModInitializer#onInitialize()V": {
            "kind": "METHOD", "owner": "net.fabricmc.api.ModInitializer",
            "name": "onInitialize", "parameters": [], "return_type": "void",
            "source_hash": "hash-method",
        },
    }


def test_selected_target_scaffold_is_not_claimed_as_implemented():
    row = verified_fabric_feature_scaffold(
        _work(), _symbols(), package_name="demo", class_name="AlienMod")
    assert row["implementation_status"] == "SCAFFOLD_ONLY_UNIMPLEMENTED"
    assert row["target_compile_passed"] is False
    assert "onInitialize()" in row["source"]
    assert row["work_order_sha256"] == _work().fingerprint()


def test_uninspected_target_interface_must_fail_closed():
    with pytest.raises(ValueError, match="NOT_INSPECTED"):
        verified_fabric_feature_scaffold(
            _work(), {}, package_name="demo", class_name="AlienMod")


def test_target_changes_scaffold_identity():
    result_a = verified_fabric_feature_scaffold(
        _work(), _symbols(), package_name="demo", class_name="AlienMod")
    result_b = verified_fabric_feature_scaffold(
        _work("1.21.8"), _symbols(), package_name="demo", class_name="AlienMod")
    assert result_a["work_order_sha256"] != result_b["work_order_sha256"]


def test_unverified_fabric_method_cannot_be_assumed():
    rows = _symbols()
    rows.pop("net.fabricmc.api.ModInitializer#onInitialize()V")
    with pytest.raises(ValueError, match="LIFECYCLE_UNVERIFIED"):
        verified_fabric_feature_scaffold(
            _work(), rows, package_name="demo", class_name="AlienMod")
