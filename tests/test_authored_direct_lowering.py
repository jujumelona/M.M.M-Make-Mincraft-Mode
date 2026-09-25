from __future__ import annotations

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules


_TARGET = {
    "minecraft_version": "1.21.11",
    "loader": "fabric",
    "mappings": "1.21.11+build.1",
}


def test_fresh_authored_design_always_lowers_to_exact_host_owned_java_tasks() -> None:
    plan = AuthoredPlan(
        "space economy",
        "# Design\n"
        "## Wallet\nPersist credits across relog.\n"
        "## Purchase\nDeduct credits exactly once.\n",
    )
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="authored_test",
        package_name="ai.minecraft.generated.authored_test",
        target=_TARGET,
    )

    assert manifest["policy"] == "host_exact_task_queue_no_coder_file_planning"
    assert len(modules) == manifest["unit_count"] == 2
    for module, record in zip(modules, manifest["units"], strict=True):
        task = module.config["evidence_task"]
        anchors = task["owned_anchors"]
        assert len(anchors) == 1
        assert anchors[0]["locator"] == f"{record['path']}#{record['symbol']}"
        assert task["required_gates"] == ["target_compile"]
        assert module.required_gates == ("target_compile",)
        assert "authored_plan" not in module.config


def test_contract_shaped_document_uses_exact_task_queue_not_bounded_coherent() -> None:
    plan = AuthoredPlan(
        "feature",
        "# Design\n"
        "## Trigger\nRight click activates the feature.\n"
        "## State\nPersist the feature state.\n"
        "## Failure\nReject invalid activation.\n",
    )
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id="authored_test",
        package_name="ai.minecraft.generated.authored_test",
        target=_TARGET,
    )

    assert modules
    assert manifest["policy"] == "host_exact_task_queue_no_coder_file_planning"
    assert all("evidence_task" in module.config for module in modules)
    assert all(
        module.config.get("authored_execution_mode") != "bounded_coherent"
        for module in modules
    )
