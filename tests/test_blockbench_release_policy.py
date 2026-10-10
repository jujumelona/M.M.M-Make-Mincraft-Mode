from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _validate_external_execution_preflight,
)
from minecraft_mod_ai.complete_orchestrator_support import file_sha256


def test_full_entity_preflight_requires_blockbench_even_if_plan_omits_gate() -> None:
    proposal = SimpleNamespace(
        external_runtime_required=False,
        acceptance_tests=(),
        modules=(SimpleNamespace(kind="entity"),),
    )
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        run_blockbench=False,
        eula_accepted=False,
        server_launcher=None,
        playtest_actions=(),
        screenshot_paths=(),
    )

    with pytest.raises(Exception, match="requires Blockbench"):
        _validate_external_execution_preflight(proposal, options)


def test_optional_client_cannot_run_without_runtime() -> None:
    proposal = SimpleNamespace(
        external_runtime_required=False,
        acceptance_tests=(),
        modules=(),
    )
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=True,
        run_mineflayer=False,
        run_visual_review=False,
        run_blockbench=True,
        eula_accepted=False,
        server_launcher=None,
        playtest_actions=(),
        screenshot_paths=(),
    )

    with pytest.raises(Exception, match="Client verification requires runtime"):
        _validate_external_execution_preflight(proposal, options)


def test_host_required_blockbench_review_is_independent_of_plan_gate(tmp_path) -> None:
    preview = tmp_path / "entity-preview.png"
    preview.write_bytes(b"preview")
    proposal = SimpleNamespace(
        modules=(
            SimpleNamespace(module_id="dragon", kind="boss", required_gates=()),
        ),
    )
    receipt = {
        "entity": "dragon",
        "uv": {"status": "PASS"},
        "preview": str(preview),
        "preview_sha256": file_sha256(preview),
    }

    assert CompleteProductionOrchestrator._mandatory_blockbench_failures(
        proposal,
        [receipt],
    ) == []
    assert CompleteProductionOrchestrator._mandatory_blockbench_failures(
        proposal,
        [],
    ) == ["blockbench:dragon:missing-host-required-review"]


def test_disabled_mandatory_blockbench_fails_before_gradle_java_preflight(
    monkeypatch,
) -> None:
    from minecraft_mod_ai import complete_preflight_contract as preflight

    expensive_stages: list[str] = []
    monkeypatch.setattr(
        preflight,
        "_validate_java_toolchain_preflight",
        lambda *_args, **_kwargs: expensive_stages.append("toolchain"),
    )
    proposal = SimpleNamespace(
        base_proposal=SimpleNamespace(spec=SimpleNamespace(platform=object())),
        external_runtime_required=False,
        acceptance_tests=(),
        modules=(SimpleNamespace(kind="entity"),),
    )
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        run_blockbench=False,
        eula_accepted=False,
        server_launcher=None,
    )
    with pytest.raises(Exception, match="requires Blockbench"):
        preflight.validate_external_execution_preflight(proposal, options)
    assert expensive_stages == []


def test_enabled_blockbench_reaches_real_toolchain_preflight(monkeypatch) -> None:
    from minecraft_mod_ai import complete_preflight_contract as preflight

    calls: list[str] = []
    monkeypatch.setattr(
        preflight,
        "_validate_java_toolchain_preflight",
        lambda *_args, **_kwargs: calls.append("toolchain"),
    )
    proposal = SimpleNamespace(
        base_proposal=SimpleNamespace(spec=SimpleNamespace(platform=object())),
        external_runtime_required=False,
        acceptance_tests=(),
        modules=(SimpleNamespace(kind="entity"),),
    )
    options = SimpleNamespace(
        source_only=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
        run_blockbench=True,
        eula_accepted=False,
        server_launcher=None,
    )
    preflight.validate_external_execution_preflight(proposal, options)
    assert calls == ["toolchain"]
