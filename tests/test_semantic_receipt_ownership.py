from minecraft_mod_ai.complete_orchestrator import _semantic_execution_observations
from minecraft_mod_ai.complete_spec import ProductionModule


def _module(module_id: str) -> ProductionModule:
    return ProductionModule(module_id=module_id, kind="item", config={"evidence_task": {"task_sha256": "sha256:" + module_id[0] * 64, "requirement_refs": [module_id], "gap_refs": [], "impact_probes": []}})


def _by_task(observations):
    return {item["task_id"]: (item["patch_receipt"], tuple(item["touched_paths"])) for item in observations}


def test_grouped_receipt_uses_explicit_module_ownership_not_position():
    members = [_module("alpha"), _module("beta"), _module("gamma")]
    grouped = {"module_ids": ["beta", "alpha"], "operation_count": 2, "touched_paths": ["src/grouped.java"], "patch_receipt": "grouped"}
    solo = {"module_id": "gamma", "operation_count": 1, "touched_paths": ["src/gamma.java"], "patch_receipt": "solo"}
    expected = {"alpha": ("grouped", ("src/grouped.java",)), "beta": ("grouped", ("src/grouped.java",)), "gamma": ("solo", ("src/gamma.java",))}
    forward = _semantic_execution_observations(members, [grouped, solo], downstream_ids=lambda _module_id: ())
    reverse = _semantic_execution_observations(members, [solo, grouped], downstream_ids=lambda _module_id: ())
    assert _by_task(forward) == expected
    assert _by_task(reverse) == expected


def test_unowned_receipt_is_not_positionally_assigned_in_multi_module_node():
    members = [_module("alpha"), _module("beta")]
    observations = _semantic_execution_observations(members, [{"patch_receipt": "ambiguous", "touched_paths": ["src/unknown.java"]}], downstream_ids=lambda _module_id: ())
    assert observations == []
