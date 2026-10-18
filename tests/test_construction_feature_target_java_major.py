"""Regression for the Colab FEATURE_WORK_ORDER_JAVA_VERSION_INVALID failure.

Target metadata stores Java 25 as a decimal string, whereas an immutable
FeatureWorkOrder requires an integer. Construction must normalize the pinned
version, not change it to a hard-coded default or drop the requirement.
"""
from __future__ import annotations

import pytest

from minecraft_mod_ai.construction_runtime import normalized_construction_target
from minecraft_mod_ai.feature_work_order import FeatureWorkOrder


def test_string_java_major_from_official_fabric_target_reaches_feature_work_order():
    original = {
        "adapter_id": "official-fabric-26.2",
        "minecraft_version": "26.2",
        "loader": "fabric",
        "java_version": "25",
    }
    typed = normalized_construction_target(original)
    assert original["java_version"] == "25"  # no mutation of upstream receipt
    assert type(typed["java_version"]) is int
    assert typed["java_version"] == 25
    order = FeatureWorkOrder.for_feature(
        requirement_id="request_demo",
        target=typed,
        target_path="src/main/java/ai/example/Demo.java",
        behaviors=[{
            "behavior_id": "behavior_demo",
            "requirement_id": "request_demo",
            "intent": "teleport to another planet",
            "effects": "player reaches the requested planet",
            "state_reads": [],
            "state_writes": [],
        }],
    )
    assert order.java_version == 25
    assert order.minecraft_version == "26.2"
    assert order.loader == "fabric"
    assert order.to_dict()["target"]["java_version"] == 25


@pytest.mark.parametrize("raw", [None, "", " 25 ", "25.0", 25.0, True, 0, -1, "0"])
def test_reject_noncanonical_or_missing_java_major(raw):
    with pytest.raises(ValueError, match="CONSTRUCTION_JAVA_VERSION_INVALID"):
        normalized_construction_target({
            "minecraft_version": "26.2", "loader": "fabric", "java_version": raw,
        })
