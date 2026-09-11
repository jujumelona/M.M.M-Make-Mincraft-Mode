from pathlib import Path

import yaml

from minecraft_mod_ai.feature_template_pipeline import ATOMIC_CHECKS, atomic_leaves, evaluate_atomicity


def _checks(failed=()):
    failed = set(failed)
    return [{"check": name, "passed": name not in failed, "reason": "explicit test evidence"} for name in ATOMIC_CHECKS]


def test_host_derives_atomicity_from_every_required_check():
    result = evaluate_atomicity(_checks())
    assert result["atomic"] is True
    assert result["failed_checks"] == []
    result = evaluate_atomicity(_checks({"explicit_trigger", "test_case_writable"}))
    assert result["atomic"] is False
    assert result["failed_checks"] == ["explicit_trigger", "test_case_writable"]


def test_atomicity_rejects_missing_or_duplicate_checks():
    records = _checks()
    try:
        evaluate_atomicity(records[:-1])
    except ValueError as exc:
        assert "missing checks" in str(exc)
    else:
        raise AssertionError("missing atomic check was accepted")
    try:
        evaluate_atomicity(records + [records[0]])
    except ValueError as exc:
        assert "duplicate check" in str(exc)
    else:
        raise AssertionError("duplicate atomic check was accepted")


def test_atomic_leaves_rejects_unresolved_non_atomic_node():
    tree = {"atomicity": {"atomic": False}, "children": []}
    try:
        list(atomic_leaves(tree))
    except ValueError as exc:
        assert "must have children" in str(exc)
    else:
        raise AssertionError("unresolved non-atomic feature was accepted")


def test_atomic_template_never_owns_final_atomic_decision():
    path = Path(__file__).parents[1] / "minecraft_mod_ai" / "templates" / "feature" / "atomic_check.yaml"
    template = yaml.safe_load(path.read_text(encoding="utf-8"))
    props = template["record_schema"]["properties"]
    assert "atomic" not in props
    assert set(props) == {"check", "passed", "reason"}
    assert tuple(props["check"]["enum"]) == ATOMIC_CHECKS


class MockFeatureRouter:
    def __init__(self, non_atomic_ids=(), splits=None):
        self.non_atomic_ids = set(non_atomic_ids)
        self.splits = splits or {}
        self.calls = []

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
        import json

        context = json.loads(messages[1]["content"])
        self.calls.append((tool_name, context))

        if tool_name.startswith("submit_feature_") and tool_name.endswith("_count"):
            target = tool_name.removeprefix("submit_feature_").removesuffix("_count")
            if target == "discover":
                count = 1
            elif target == "decompose":
                fid = context["feature"]["feature_id"]
                count = len(self.splits.get(fid, ()))
            else:
                count = 1
            return {"count": count, "blocked_reason": ""}

        if not tool_name.startswith("submit_one_feature_"):
            raise AssertionError(tool_name)
        target = tool_name.removeprefix("submit_one_feature_")

        if target == "discover":
            return {
                "feature_id": "multi_part_machine",
                "feature_description": "A machine that stores energy and opens a menu",
                "evidence_basis": "prompt",
            }
        if target == "atomic_check":
            check = context["target_check"]
            fid = context["feature_id"]
            passed = not (
                fid in self.non_atomic_ids
                and check in ("single_primary_behavior", "explicit_trigger")
            )
            return {
                "check": check,
                "passed": passed,
                "reason": "verified" if passed else "multi responsibility",
            }
        if target == "decompose":
            fid = context["feature"]["feature_id"]
            rows = self.splits.get(fid, [])
            index = int(context["record_index"])
            return dict(rows[index])

        fid = context["feature"]["feature_id"]
        record_fact = {
            "purpose": {"feature_id": fid, "purpose": f"{fid} purpose"},
            "behavior": {"feature_id": fid, "behavior": f"{fid} behavior"},
            "trigger": {"feature_id": fid, "trigger": f"{fid} trigger"},
            "input": {"feature_id": fid, "input_name": "data", "input_contract": "non-empty"},
            "output": {"feature_id": fid, "output_name": "result", "output_contract": "emits result"},
            "state": {"feature_id": fid, "state_name": "active", "state_contract": "boolean"},
            "transition": {"from_state": "idle", "event": "start", "to_state": "active"},
            "rules": {"feature_id": fid, "rule": "deterministic rule"},
            "constraints": {"feature_id": fid, "constraint": "positive bounds"},
            "dependencies": {"feature_id": fid, "dependency": "core", "reason": "required"},
            "connections": {"feature_id": fid, "target": "network", "connection": "sync"},
            "persistence": {"feature_id": fid, "persistent_value": "energy", "lifetime": "world"},
            "networking": {"feature_id": fid, "network_responsibility": "state_sync", "authority": "server"},
            "ui": {"feature_id": fid, "ui_responsibility": "gauge", "interaction": "read_only"},
            "resources": {"feature_id": fid, "resource_kind": "model", "resource_requirement": "block_model"},
            "assets": {"feature_id": fid, "asset_kind": "texture", "asset_requirement": "machine_png"},
        }.get(target)
        if record_fact is None:
            raise AssertionError(tool_name)
        return record_fact


def test_discover_features_and_decompose_pipeline_integration():
    from minecraft_mod_ai.feature_template_pipeline import (
        discover_features,
        decompose_features_pipeline,
    )

    splits = {
        "multi_part_machine": [
            {
                "feature_id": "energy_storage",
                "behavior": "Accumulate and retain energy units",
                "reason_for_split": "separable responsibility",
            },
            {
                "feature_id": "machine_ui",
                "behavior": "Render gauge and handle interactions",
                "reason_for_split": "separable responsibility",
            },
        ]
    }
    router = MockFeatureRouter(non_atomic_ids={"multi_part_machine"}, splits=splits)
    discovered = discover_features(
        router,
        context={"prompt": "Add machine with energy and screen", "research_context": "evidenced"},
        allowed_refs=set(),
    )
    assert len(discovered) == 1
    assert discovered[0]["feature_id"] == "multi_part_machine"

    leaves = decompose_features_pipeline(router, discovered, allowed_refs=set())
    leaf_ids = [leaf["feature_id"] for leaf in leaves]
    assert leaf_ids == ["energy_storage", "machine_ui"]
    assert all(leaf["atomicity"]["atomic"] for leaf in leaves)
    assert all(not leaf["children"] for leaf in leaves)


def test_feature_decomposition_cycle_and_semantic_progress_protection():
    import pytest
    from minecraft_mod_ai.task_template_runner import TemplateBlocked
    from minecraft_mod_ai.feature_template_pipeline import complete_feature

    router_cycle = MockFeatureRouter(
        non_atomic_ids={"loop_feature"},
        splits={
            "loop_feature": [
                {
                    "feature_id": "loop_feature",
                    "behavior": "Same",
                    "reason_for_split": "Same",
                }
            ]
        },
    )
    with pytest.raises(TemplateBlocked, match="child repeats parent"):
        complete_feature(
            router_cycle,
            {"feature_id": "loop_feature", "feature_description": "Loop"},
            allowed_refs=set(),
        )

    no_progress_router = MockFeatureRouter(
        non_atomic_ids={"d0", "d1"},
        splits={
            "d0": [{"feature_id": "d1", "behavior": "b1", "reason_for_split": "r"}],
        },
    )
    with pytest.raises(TemplateBlocked, match="unresolved atomic checks did not strictly decrease"):
        complete_feature(
            no_progress_router,
            {"feature_id": "d0", "feature_description": "Progress test"},
            allowed_refs=set(),
        )
