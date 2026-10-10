"""Mandatory full-build Blockbench verification must survive stale Colab flags."""

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    _mandatory_blockbench_execution_options,
)
from minecraft_mod_ai.complete_preflight_contract import (
    validate_external_execution_preflight,
)
from minecraft_mod_ai.complete_orchestrator_support import CompleteProductionError


@pytest.mark.parametrize("kind", ("entity", "boss", "npc"))
def test_full_entity_modes_auto_enable_real_review(kind):
    proposal = SimpleNamespace(
        modules=(SimpleNamespace(kind=kind),),
        base_proposal=None,
        external_runtime_required=False,
    )
    old_options = CompleteExecutionOptions(
        run_blockbench=False,
        run_runtime=False,
        run_client=False,
        run_mineflayer=False,
        run_visual_review=False,
    )
    resolved = _mandatory_blockbench_execution_options(proposal, old_options)
    assert resolved.run_blockbench is True
    assert old_options.run_blockbench is False
    validate_external_execution_preflight(proposal, resolved)


def test_preflight_still_blocks_direct_bypass_of_mandatory_review():
    proposal = SimpleNamespace(
        modules=(SimpleNamespace(kind="entity"),),
        base_proposal=None,
        external_runtime_required=False,
    )
    disabled = CompleteExecutionOptions(
        run_blockbench=False, run_runtime=False,
        run_client=False, run_mineflayer=False,
        run_visual_review=False,
    )
    with pytest.raises(CompleteProductionError, match="Blockbench"):
        validate_external_execution_preflight(proposal, disabled)


@pytest.mark.parametrize("kind", ("item", "block", "recipe"))
def test_non_entity_builds_do_not_enable_unneeded_review(kind):
    proposal = SimpleNamespace(modules=(SimpleNamespace(kind=kind),))
    options = CompleteExecutionOptions(run_blockbench=False)
    assert _mandatory_blockbench_execution_options(proposal, options) is options


def test_source_only_does_not_start_external_review():
    proposal = SimpleNamespace(modules=(SimpleNamespace(kind="entity"),))
    options = CompleteExecutionOptions(source_only=True, run_blockbench=False)
    assert _mandatory_blockbench_execution_options(proposal, options) is options


def test_explicit_enabled_review_remains_unchanged():
    proposal = SimpleNamespace(modules=(SimpleNamespace(kind="entity"),))
    options = CompleteExecutionOptions(run_blockbench=True)
    assert _mandatory_blockbench_execution_options(proposal, options) is options
