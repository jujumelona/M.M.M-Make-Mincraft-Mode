from __future__ import annotations

import pytest

from minecraft_mod_ai import planning_pipeline as planning_pipeline_module
from minecraft_mod_ai import planning_state_pipeline
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    _task_local_module_contract,
)
from minecraft_mod_ai.planning_pipeline import PlanningPipeline


def test_top_level_prepare_forwards_exact_detail_applicability_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_prepare_planning_state(
        router: object,
        prompt: str,
        *,
        existing_state: object = None,
        checkpoint: object = None,
        detail_section_applicability_resolver: object = None,
    ) -> dict[str, object]:
        del router, existing_state
        observed["resolver"] = detail_section_applicability_resolver
        state: dict[str, object] = {
            "original_prompt": prompt,
            "state_sha256": "sha256:" + "0" * 64,
        }
        if callable(checkpoint):
            checkpoint(state)
        return state

    monkeypatch.setattr(
        planning_state_pipeline,
        "prepare_planning_state",
        fake_prepare_planning_state,
    )
    monkeypatch.setattr(
        planning_pipeline_module,
        "_host_operation",
        lambda _operation, callback: callback(),
    )
    monkeypatch.setattr(
        PlanningPipeline,
        "_semantic_design",
        lambda self, prompt, *, planning_state, media_paths: (
            {"design": True},
            object(),
        ),
    )
    monkeypatch.setattr(
        PlanningPipeline,
        "_bind_platform",
        lambda self, prompt, design, proposal: (
            design,
            proposal,
            {"brief": True},
            {"schema_version": "test/evidence-v1"},
        ),
    )
    monkeypatch.setattr(
        PlanningPipeline,
        "_validated_evidence",
        staticmethod(lambda value: dict(value)),
    )

    def resolver(requirement_ids: tuple[str, ...]) -> dict[str, dict[str, str]]:
        return {
            requirement_id: {"persistence": "unknown"}
            for requirement_id in requirement_ids
        }

    pipeline = PlanningPipeline(object())  # type: ignore[arg-type]
    artifacts = pipeline.prepare(
        "resolver forwarding probe",
        detail_section_applicability_resolver=resolver,
    )

    assert observed["resolver"] is resolver
    assert artifacts.planning_state["original_prompt"] == "resolver forwarding probe"


@pytest.mark.parametrize(
    "config",
    [
        {
            "original_prompt": "DO NOT LEAK",
            "engineering_worksheet": {"global": "DO NOT LEAK"},
        },
        {"evidence_task": "not-a-structured-task", "secret": "DO NOT LEAK"},
    ],
)
def test_task_local_module_contract_fails_closed_without_structured_evidence_task(
    config: dict[str, object],
) -> None:
    module = ProductionModule(
        module_id="secret_module",
        kind="custom_java",
        config=dict(config),
        depends_on=("global_dependency",),
        required_gates=("global gate",),
    )

    with pytest.raises(
        CustomModuleGenerationError,
        match=r"^TASK_LOCAL_CONTRACT_REQUIRED:",
    ):
        _task_local_module_contract(module)
