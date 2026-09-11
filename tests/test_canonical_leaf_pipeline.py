from __future__ import annotations

import json
import pytest

from minecraft_mod_ai.artifact_expansion import (
    expand_facts_to_jobs,
    FACT_TO_CANONICAL_LEAVES,
)
from minecraft_mod_ai.atomic_slot_executor import (
    fill_one_slot,
    SlotDefinition,
    SlotFillError,
    MAX_SLOT_CONTEXT_CHARS,
)
from minecraft_mod_ai.host_version_catalog import audit_host_catalog, host_target, load_host_catalog
from minecraft_mod_ai.minecraft_template_steps import (
    responsibility_ids_for_artifact,
    steps_for_artifact,
    TemplateStep,
    ContextProjection,
)
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
from minecraft_mod_ai.resolved_version_context import ResolvedVersionContext, VersionContextError, _encode
from minecraft_mod_ai.structural_routing_contract import CANONICAL_ARTIFACT_KINDS


def test_every_canonical_leaf_has_complete_binding_in_all_host_bundles():
    """All 43 bundles must have explicit leaf coverage for all 346 canonical leaves."""
    report = audit_host_catalog()
    assert report["schema_version"] == "mmm/host-version-catalog-audit-v1"
    assert len(report["contexts"]) == 43

    all_leaves = {
        leaf
        for kind in CANONICAL_ARTIFACT_KINDS
        for leaf in responsibility_ids_for_artifact(kind)
    }
    assert len(all_leaves) == 346

    _, bundles = load_host_catalog()
    for ctx in bundles:
        bindings = ctx.facts["leaf_bindings"]
        assert len(bindings) == 346
        assert set(bindings) == all_leaves
        for leaf_id, binding in bindings.items():
            assert binding["state"] in {"admitted", "unsupported", "not_reviewed"}
            if binding["state"] == "admitted":
                assert "implementation" in binding


def test_missing_canonical_leaf_causes_audit_failure():
    """Audit must fail closed with HOST_LEAF_COVERAGE_INCOMPLETE if any canonical leaf is unmapped."""
    resolver, bundles = load_host_catalog()
    bundle = bundles[0]
    bundle_dict = bundle.to_dict()

    # Drop one canonical leaf binding and re-encode
    bundle_dict.pop("context_id", None)
    bundle_dict["host_facts"]["leaf_bindings"].pop("minecraft/item/registry")
    corrupted_ctx = ResolvedVersionContext(_encode(bundle_dict))

    from minecraft_mod_ai.host_version_catalog import VersionResolver
    corrupted_resolver = VersionResolver([corrupted_ctx], auto_context_id=corrupted_ctx.context_id)

    from unittest.mock import patch
    with patch("minecraft_mod_ai.host_version_catalog.load_host_catalog", return_value=(corrupted_resolver, [corrupted_ctx])):
        with pytest.raises(VersionContextError) as exc_info:
            audit_host_catalog()
        assert "HOST_LEAF_COVERAGE_INCOMPLETE" in str(exc_info.value)
        assert "minecraft/item/registry" in str(exc_info.value)


def test_canonical_pipeline_lowering_item_and_block():
    """Verify lowering from PromptFact through canonical leaves to concrete ArtifactJobs with context_id."""
    target = host_target("auto")
    ctx = target.version_context

    facts = [
        PromptFact(fact_id="f1", fact_type=FactType.ITEM_EXISTS, subject="ruby"),
        PromptFact(fact_id="f2", fact_type=FactType.ITEM_STACK_LIMIT, subject="ruby", value=32),
        PromptFact(fact_id="f3", fact_type=FactType.BLOCK_EXISTS, subject="ruby_block"),
    ]

    jobs = expand_facts_to_jobs(
        facts,
        mod_id="gemmod",
        package_name="com.gemmod",
        version_context=ctx,
    )

    assert len(jobs) == 13
    assert all(job.context_id == ctx.context_id for job in jobs)
    template_ids = {job.template_id for job in jobs}
    assert "fabric/item/register_basic" in template_ids
    assert "fabric/item/settings_max_stack" in template_ids
    assert "fabric/block/register_basic" in template_ids
    assert "fabric/block/blockstate_basic" in template_ids


def test_unsupported_leaf_rejected_by_version_context():
    """Recipes on legacy Minecraft (< 1.21.2) must fail closed with UNSUPPORTED_LEAF."""
    ctx_1_20_1 = host_target("1.20.1").version_context

    with pytest.raises(VersionContextError) as exc_info:
        ctx_1_20_1.require_leaf_binding("minecraft/recipe/serializer")
    assert "UNSUPPORTED_LEAF" in str(exc_info.value)
    assert "unsupported" in str(exc_info.value)

    # In modern 1.21.4, recipe serializer is admitted
    ctx_1_21_4 = host_target("1.21.4").version_context
    binding = ctx_1_21_4.require_leaf_binding("minecraft/recipe/serializer")
    assert binding["state"] == "admitted"


def test_specialized_leaf_contracts_and_side_separation():
    """Verify that specialized leaf contracts enforce execution mode, side, and context limits."""
    item_steps = steps_for_artifact("item")
    step_by_id = {s.template_id: s for s in item_steps}

    req_step = step_by_id["minecraft/item/requirement"]
    assert req_step.execution_mode == "deterministic"
    assert req_step.side == ("common",)

    reg_step = step_by_id["minecraft/item/registry"]
    assert reg_step.execution_mode == "model"
    assert reg_step.side == ("common",)
    assert reg_step.context_projection.max_bytes <= 4096

    # Entity renderer is strictly client side
    entity_steps = steps_for_artifact("entity")
    renderer_step = next(s for s in entity_steps if s.template_id == "minecraft/entity/renderer")
    assert renderer_step.side == ("client",)
    assert renderer_step.execution_mode == "model"


def test_context_projection_exceeding_4096_bytes_fails_closed():
    """Context projection exceeding 4096 characters must fail-closed with SLOT_CONTEXT_TOO_LARGE, never truncate."""
    slot = SlotDefinition(
        slot_id="test_slot",
        schema={"type": "string", "maxLength": 64},
        description="A test slot",
    )

    oversized_context = {
        "large_payload": "x" * 4100,
    }

    from unittest.mock import MagicMock
    fake_router = MagicMock()

    with pytest.raises(SlotFillError) as exc_info:
        fill_one_slot(fake_router, slot, oversized_context)

    assert "SLOT_CONTEXT_TOO_LARGE" in str(exc_info.value)
    assert str(MAX_SLOT_CONTEXT_CHARS) in str(exc_info.value)
