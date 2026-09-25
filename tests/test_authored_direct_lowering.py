from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import compile_authored_design


def _router() -> SimpleNamespace:
    return SimpleNamespace()


def test_fresh_authored_design_always_lowers_to_exact_host_owned_java_tasks(
    monkeypatch,
) -> None:
    target = {
        "minecraft_version": "1.21.11",
        "loader": "fabric",
        "mappings": "1.21.11+build.1",
    }

    from minecraft_mod_ai import authored_production as authored

    monkeypatch.setattr(
        authored.PlanningPipeline,
        "_bind_existing_project",
        lambda self, design: dict(design),
    )
    monkeypatch.setattr(
        authored.PlanningPipeline,
        "_bind_platform",
        lambda self, prompt, design, base: (
            {**design, **target},
            base,
            (),
            (),
        ),
    )

    plan = AuthoredPlan(
        "space economy",
        "# Design\n"
        "## Wallet\nPersist credits across relog.\n"
        "## Purchase\nDeduct credits exactly once.\n",
    )
    proposal = compile_authored_design(_router(), plan)

    manifest = proposal.game_design["_authored_execution_manifest"]
    assert manifest["policy"] == "host_exact_task_queue_no_coder_file_planning"
    assert len(proposal.modules) == manifest["unit_count"] == 2
    for module, record in zip(proposal.modules, manifest["units"], strict=True):
        task = module.config["evidence_task"]
        anchors = task["owned_anchors"]
        assert len(anchors) == 1
        assert anchors[0]["locator"] == f"{record['path']}#{record['symbol']}"
        assert task["required_gates"] == ["target_compile"]
        assert module.required_gates == ("target_compile",)
        assert "authored_plan" not in module.config


def test_contract_shaped_document_is_not_routed_to_bounded_coherent_coder(
    monkeypatch,
) -> None:
    target = {
        "minecraft_version": "1.21.11",
        "loader": "fabric",
        "mappings": "1.21.11+build.1",
    }

    from minecraft_mod_ai import authored_production as authored

    monkeypatch.setattr(
        authored.PlanningPipeline,
        "_bind_existing_project",
        lambda self, design: dict(design),
    )
    monkeypatch.setattr(
        authored.PlanningPipeline,
        "_bind_platform",
        lambda self, prompt, design, base: (
            {**design, **target},
            base,
            (),
            (),
        ),
    )

    plan = AuthoredPlan(
        "feature",
        "# Design\n"
        "## Trigger\nRight click activates the feature.\n"
        "## State\nPersist the feature state.\n"
        "## Failure\nReject invalid activation.\n",
    )
    proposal = compile_authored_design(_router(), plan)

    assert proposal.modules
    assert all("evidence_task" in module.config for module in proposal.modules)
    assert all(
        module.config.get("authored_execution_mode") != "bounded_coherent"
        for module in proposal.modules
    )
