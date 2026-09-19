from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minecraft_mod_ai import custom_module_generator as generator
from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from minecraft_mod_ai.complete_spec import CompleteProposal
from minecraft_mod_ai import platform_live_execution_contract as live_platform
from minecraft_mod_ai.platform_catalog import adapter_for_lock_values, provider_for_loader


def _assert_official_scaffold_route(proposal: CompleteProposal) -> None:
    adapter = adapter_for_lock_values(proposal.base_proposal.spec.platform)
    provider = provider_for_loader(adapter.loader)
    assert adapter.loader == "fabric"
    assert tuple(adapter.deterministic_module_kinds) == ()
    assert provider.host_authoritative is True, (
        provider.provider_id,
        provider.host_authoritative,
    )
    assert live_platform._uses_official_scaffold(adapter) is True


class _DeterministicRouter:
    profile = "deterministic-e2e"
    registry = None

    def __init__(self) -> None:
        self.workspace_root: Path | None = None

    def bind_agent_workspace(
        self,
        workspace_root: str | Path,
        *,
        require_fresh_evidence: bool = False,
    ) -> _DeterministicRouter:
        del require_fresh_evidence
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        return self

    def generate_text(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError(
            "deterministic full-pipeline E2E must not invoke a live model"
        )


def _bound_workspace(router: Any) -> Path:
    current = router
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        workspace = getattr(current, "workspace_root", None)
        if workspace is not None:
            return Path(workspace).expanduser().resolve()
        current = getattr(current, "_router", None)
    raise AssertionError("custom coder router was not bound to a staged workspace")


def _deterministic_coder(
    router: Any,
    role: str,
    messages: Any,
    *args: Any,
    **kwargs: Any,
) -> str:
    del args, kwargs
    assert role == "coder"
    request = json.loads(messages[-1]["content"])
    grounding = request["host_grounding"]["evidence_bindings"][
        "implementation_contract"
    ]["grounding"]
    assert grounding["artifact_kind"] == "item"
    fact = grounding["facts"][0]
    imports = list(dict.fromkeys(fact["required_imports"]))
    templates = list(fact["templates"])
    key_template = next(
        item
        for item in templates
        if "resource_key_create" in item.get("symbol_usage", ())
    )
    register_template = next(
        item
        for item in templates
        if "register_item" in item.get("symbol_usage", ())
    )

    key_body = key_template["render_body"]
    register_body = register_template["render_body"]
    replacements = {
        "{{java_constant}}": "DEBUG_TOKEN",
        "{{mod_id}}": "mmm_debug_fixture",
        "{{registry_path}}": "debug_token",
        "ModItemIds.DEBUG_TOKEN_KEY": "DEBUG_TOKEN_KEY",
    }
    for before, after in replacements.items():
        key_body = key_body.replace(before, after)
        register_body = register_body.replace(before, after)

    source = "\n".join(
        [
            "package dev.mmm.debugfixture;",
            "",
            *(f"import {owner};" for owner in imports),
            "",
            "public final class DebugToken {",
            "    private DebugToken() {}",
            "",
            *(
                "    " + line if line else ""
                for line in key_body.strip().splitlines()
            ),
            "",
            *(
                "    " + line if line else ""
                for line in register_body.strip().splitlines()
            ),
            "}",
            "",
        ]
    )

    target = (
        _bound_workspace(router)
        / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return json.dumps(
        {
            "summary": (
                "Created the host-grounded DebugToken fixture in the owned target."
            )
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_debug_fixture_runs_real_build_and_packaging_without_live_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generator, "_generate_coder_text", _deterministic_coder)
    plan_path = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="1.21.8",
        loader="fabric",
    )
    proposal = CompleteProposal.from_dict(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    _assert_official_scaffold_route(proposal)
    orchestrator = CompleteProductionOrchestrator(
        workspace_root=tmp_path / "workspace",
        profile="deterministic-e2e",
        router_factory=_DeterministicRouter,
    )
    _assert_official_scaffold_route(proposal)
    _original_official_scaffold_route = live_platform._uses_official_scaffold

    def _checked_official_scaffold_route(adapter):
        provider = provider_for_loader(adapter.loader)
        result = _original_official_scaffold_route(adapter)
        assert result is True, {
            "loader": adapter.loader,
            "adapter_id": adapter.adapter_id,
            "source_api_family": adapter.source_api_family,
            "deterministic_module_kinds": list(adapter.deterministic_module_kinds),
            "provider_id": provider.provider_id,
            "provider_host_authoritative": provider.host_authoritative,
        }
        return result

    monkeypatch.setattr(
        live_platform,
        "_uses_official_scaffold",
        _checked_official_scaffold_route,
    )
    result = orchestrator.execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="debug-full-pipeline-e2e",
        options=CompleteExecutionOptions(
            source_only=False,
            run_jdt=False,
            run_gametest=True,
            auto_repair=False,
            run_blockbench=False,
            run_runtime=False,
            run_client=False,
            run_mineflayer=False,
            run_visual_review=False,
            cleanup_runtime=True,
            eula_accepted=False,
        ),
    )

    project_root = Path(result.project_root)
    target = project_root / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    assert target.is_file()
    assert result.build_report is not None
    assert result.build_report["status"] == "PASS"
    assert CompleteProductionOrchestrator._full_gradle_build_receipt_passed(
        result.build_report
    )
    assert result.jar_validation is not None
    assert result.jar_validation["status"] == "PASS"
    assert result.release_zip is not None
    assert Path(result.release_zip).is_file()
    assert not any(
        gate.startswith("required-gate:debug_token:target_compile:")
        for gate in result.unresolved_gates
    )
