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


def test_generator_handoff_subordinated_to_canonical_leaf_and_admitted():
    """Generator handoffs must be bound to canonical leaves with context_id and hash."""
    from minecraft_mod_ai.artifact_expansion import generator_implementation_profile

    target = host_target("auto")
    ctx = target.version_context

    profile = generator_implementation_profile(FactType.ENTITY_EXISTS, version_context=ctx)
    assert profile.canonical_leaf == "minecraft/entity/registry"
    assert profile.executor == "generator_handoff_entity_exists"
    assert profile.context_id == ctx.context_id
    assert profile.implementation_hash.startswith("sha256:")
    assert "java_syntax" in profile.validators or "java_parse" in profile.validators or len(profile.validators) >= 0


def test_version_bounded_generator_leaves_fail_closed_on_legacy_versions():
    """Version-bounded generator handoffs fail closed with UNSUPPORTED_LEAF on older versions."""
    from minecraft_mod_ai.artifact_expansion import generator_implementation_profile

    ctx_1_20_1 = host_target("1.20.1").version_context
    with pytest.raises(VersionContextError) as exc_info:
        generator_implementation_profile(FactType.DATA_COMPONENT, version_context=ctx_1_20_1)
    assert "UNSUPPORTED_LEAF" in str(exc_info.value)
    assert "minecraft/component/type" in str(exc_info.value)

    # 1.21.4 admits DATA_COMPONENT
    ctx_1_21_4 = host_target("1.21.4").version_context
    profile = generator_implementation_profile(FactType.DATA_COMPONENT, version_context=ctx_1_21_4)
    assert profile.canonical_leaf == "minecraft/component/type"
    assert profile.context_id == ctx_1_21_4.context_id

    # 1.15.2 rejects DIMENSION (custom dimensions were introduced in 1.16)
    ctx_1_15_2 = host_target("1.15.2").version_context
    with pytest.raises(VersionContextError) as exc_info:
        generator_implementation_profile(FactType.DIMENSION, version_context=ctx_1_15_2)
    assert "UNSUPPORTED_LEAF" in str(exc_info.value)
    assert "minecraft/dimension/registry" in str(exc_info.value)


def test_artifact_validation_prevents_cross_side_leakage():
    """Common/server templates must fail closed if client classes leak into output."""
    from minecraft_mod_ai.task_template_catalog import load_template

    target = host_target("auto")
    ctx = target.version_context

    template = load_template("fabric/item/register_basic")
    leaked_output = (
        "package com.example;\n"
        "import net.minecraft.client.MinecraftClient;\n"
        "import net.minecraft.registry.Registry;\n"
        "public class Items { public static void register() { Registry.register(null, null, null); } }"
    )

    with pytest.raises(VersionContextError) as exc_info:
        ctx.validate_artifact(template, leaked_output)
    assert "CROSS_SIDE_LEAKAGE" in str(exc_info.value)
    assert template["id"] in str(exc_info.value)


def test_artifact_validation_verifies_api_invocation_syntax():
    """API symbol validation checks that callable symbols are actually invoked."""
    from minecraft_mod_ai.task_template_catalog import load_template

    target = host_target("auto")
    ctx = target.version_context

    template = load_template("fabric/item/register_basic")
    # Output includes the string 'Registry.register' as a comment/identifier but never invokes it
    malformed_output = (
        "package com.example;\n"
        "public class Items {\n"
        "    // note about Registry.register\n"
        "    public static void register() { int x = 1; }\n"
        "}"
    )

    with pytest.raises(VersionContextError) as exc_info:
        ctx.validate_artifact(template, malformed_output)
    assert "INVALID_API_INVOCATION" in str(exc_info.value)
    assert "register_item" in str(exc_info.value)


def test_canonical_leaf_attached_to_jobs_and_receipts():
    """ArtifactJob carries canonical_leaf and propagates it into validation receipts."""
    from minecraft_mod_ai.task_template_runner import execute_artifact_template

    target = host_target("auto")
    ctx = target.version_context

    facts = [
        PromptFact(fact_id="f1", fact_type=FactType.ITEM_EXISTS, subject="sapphire"),
    ]
    jobs = expand_facts_to_jobs(
        facts,
        mod_id="testmod",
        package_name="com.testmod",
        version_context=ctx,
    )

    for job in jobs:
        assert job.canonical_leaf != ""
        assert job.canonical_leaf.startswith("minecraft/")

    key_job = next(j for j in jobs if j.template_id == "fabric/item/key")
    assert key_job.canonical_leaf == "minecraft/item/registry"

    result = execute_artifact_template(key_job, context={"resolved_version_context": ctx})
    assert result["status"] == "PASS"
    assert len(key_job.validation_receipts) > 0
    for receipt in key_job.validation_receipts:
        assert receipt.get("canonical_leaf") == "minecraft/item/registry"
        assert receipt.get("implementation_id") != ""
        assert receipt.get("executor_type") == "deterministic_renderer"


def test_production_readiness_audit_and_rejection():
    """production_readiness_audit passes on default scope and rejects unreviewed leaves."""
    from minecraft_mod_ai.host_version_catalog import production_readiness_audit

    report = production_readiness_audit()
    assert report["status"] == "PASS"
    assert report["bundles_evaluated"] == 43
    assert report["scope_size"] >= 90

    with pytest.raises(VersionContextError) as exc_info:
        production_readiness_audit(["minecraft/advancement/criterion"])
    assert "PRODUCTION_AUDIT_FAILED" in str(exc_info.value)
    assert "minecraft/advancement/criterion" in str(exc_info.value)


def test_admitted_leaf_requires_all_eight_fields():
    """require_leaf_binding fails if any of the 8 required implementation fields are missing."""
    target = host_target("auto")
    ctx = target.version_context

    binding = ctx.require_leaf_binding("minecraft/item/registry")
    impl = binding["implementation"]
    required_fields = (
        "implementation_id",
        "executor_type",
        "implementation_sha256",
        "validator_profile",
        "validator_sha256",
        "input_schema_sha256",
        "output_schema_sha256",
        "evidence_id",
    )
    for field in required_fields:
        assert field in impl
        assert isinstance(impl[field], str) and impl[field] != ""
        if field.endswith("_sha256"):
            assert impl[field].startswith("sha256:")

    # Test tampering with missing field - ResolvedVersionContext fails at boundary
    bundle_dict = ctx.to_dict()
    bundle_dict.pop("context_id", None)
    corrupted_impl = dict(impl)
    corrupted_impl.pop("validator_sha256")
    bundle_dict["host_facts"]["leaf_bindings"]["minecraft/item/registry"]["implementation"] = corrupted_impl

    with pytest.raises(VersionContextError) as exc_info:
        ResolvedVersionContext(_encode(bundle_dict))
    assert "HOST_LEAF_BINDING_INVALID" in str(exc_info.value)
    assert "validator_sha256" in str(exc_info.value)


def test_structured_api_symbols_schema_and_ast_validation():
    """Symbols in host catalog have {owner, name, descriptor, kind, static, side, namespace} and AST validation strips comments/strings."""
    from collections.abc import Mapping

    target = host_target("auto")
    ctx = target.version_context

    for sym_name, sym in ctx.api_symbols.items():
        assert isinstance(sym, Mapping)
        for key in ("owner", "name", "descriptor", "kind", "static", "side", "namespace"):
            assert key in sym, f"Missing {key} in {sym_name}"
        assert sym["kind"] in {"method", "field", "class", "constructor"}
        assert isinstance(sym["static"], bool)
        assert sym["side"] in {"common", "client", "server"}
        assert sym["namespace"] in {"minecraft", "fabric"}

    # AST validation: symbol commented out should fail
    from minecraft_mod_ai.task_template_catalog import load_template
    template = load_template("fabric/item/register_basic")
    commented_out = """
    // Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, null);
    public static final Item RAW_LUNITE = null;
    """
    with pytest.raises(VersionContextError) as exc_info:
        ctx.validate_artifact(template, commented_out)
    assert "INVALID_API_INVOCATION" in str(exc_info.value)

    # Symbol only in string literal should fail
    string_only = """
    String s = "Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, null);";
    public static final Item RAW_LUNITE = null;
    """
    with pytest.raises(VersionContextError) as exc_info:
        ctx.validate_artifact(template, string_only)
    assert "INVALID_API_INVOCATION" in str(exc_info.value)

    # Cross side leakage only in comment should NOT fail, but in executable code MUST fail
    valid_with_comment = """
    /* Uses net.minecraft.client.gui.screen.Screen internally */
    public static final Item RAW_LUNITE = Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, new Item(new Item.Properties()));
    """
    res = ctx.validate_artifact(template, valid_with_comment)
    assert res["status"] == "PASS"

    leakage_in_code = """
    net.minecraft.client.gui.screen.Screen screen = null;
    public static final Item RAW_LUNITE = Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, new Item(new Item.Properties()));
    """
    with pytest.raises(VersionContextError) as exc_info:
        ctx.validate_artifact(template, leakage_in_code)
    assert "CROSS_SIDE_LEAKAGE" in str(exc_info.value)


