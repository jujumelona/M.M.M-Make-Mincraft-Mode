from __future__ import annotations

from hashlib import sha256

import pytest

from minecraft_mod_ai.artifact_expansion import (
    expand_facts_to_jobs,
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
)
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
from minecraft_mod_ai.resolved_version_context import ResolvedVersionContext, VersionContextError, _encode
from minecraft_mod_ai.structural_routing_contract import CANONICAL_ARTIFACT_KINDS


def _validation_context(template, *, required_symbols=()):
    """Build a test-only admitted template context without changing production admission."""
    ctx = host_target("auto").version_context
    raw = ctx.to_dict()
    raw.pop("context_id", None)
    facts = raw["host_facts"]
    symbols = dict(facts["api_symbols"])
    symbols.setdefault(
        "register_item",
        {
            "owner": "Registry",
            "name": "register",
            "descriptor": "(registry,key,value)->value",
            "kind": "method",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
    )
    rules = dict(facts["artifact_rules"])
    rules[template["id"]] = {
        "template_sha256": "sha256:" + sha256(_encode(template).encode()).hexdigest(),
        "required_symbols": list(required_symbols),
        "requires_capabilities": [],
    }
    facts["api_symbols"] = symbols
    facts["artifact_rules"] = rules
    return ResolvedVersionContext(_encode(raw))


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
    """Metadata discovery must not authorize untested production execution."""
    ctx = host_target("auto").version_context
    with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
        expand_facts_to_jobs([PromptFact(fact_id="f1", fact_type=FactType.ITEM_EXISTS, subject="ruby")],
                            mod_id="gemmod", package_name="com.gemmod", version_context=ctx)


def test_unsupported_leaf_rejected_by_version_context():
    for version in ("1.20.1", "1.21.4"):
        with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
            host_target(version).version_context.require_leaf_binding("minecraft/recipe/serializer")


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


def test_generator_handoff_subordinated_to_canonical_leaf():
    from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity
    authority = bootstrap_integrity()
    ctx = host_target("auto").version_context
    binding = ctx.facts["leaf_bindings"]["minecraft/entity/registry"]
    impl = binding["implementation"]
    assert impl["executor_type"] == "generator_handoff_entity"
    assert callable(authority.executors[impl["implementation_id"]])
    assert "evidence_id" not in impl


def test_version_bounded_generator_leaves_fail_closed_on_legacy_versions():
    for version, leaf in (("1.20.1", "minecraft/component/type"),
                          ("1.21.4", "minecraft/component/type"),
                          ("1.15.2", "minecraft/dimension/registry")):
        with pytest.raises(VersionContextError, match="UNSUPPORTED_LEAF"):
            host_target(version).version_context.require_leaf_binding(leaf)


def test_artifact_validation_prevents_cross_side_leakage():
    """Common/server templates must fail closed if client classes leak into output."""
    from minecraft_mod_ai.task_template_catalog import load_template

    template = load_template("fabric/item/register_basic")
    ctx = _validation_context(template)
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

    template = load_template("fabric/item/register_basic")
    ctx = _validation_context(template, required_symbols=("register_item",))
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
    from minecraft_mod_ai.task_template_runner import execute_artifact_template
    jobs = expand_facts_to_jobs([PromptFact(fact_id="f1", fact_type=FactType.ITEM_EXISTS, subject="sapphire")],
                               mod_id="testmod", package_name="com.testmod")
    assert all(job.canonical_leaf.startswith("minecraft/") for job in jobs)
    with pytest.raises(VersionContextError, match="VERSION_CONTEXT_REQUIRED"):
        execute_artifact_template(jobs[0])


def test_production_readiness_audit_and_rejection():
    from minecraft_mod_ai.host_version_catalog import production_readiness_audit
    report = production_readiness_audit()
    assert report["status"] == "FAIL"
    assert report["bundles_evaluated"] == 43
    assert report["failures"]
    specific = production_readiness_audit(["minecraft/advancement/criterion"])
    assert specific["status"] == "FAIL"
    assert all(row["leaf"] == "minecraft/advancement/criterion" for row in specific["failures"])


def test_admitted_leaf_requires_all_eight_fields():
    """require_leaf_binding fails if any of the 8 required implementation fields are missing."""
    target = host_target("auto")
    ctx = target.version_context

    binding = ctx.facts["leaf_bindings"]["minecraft/item/registry"]
    impl = binding["implementation"]
    required_fields = (
        "implementation_id",
        "executor_type",
        "implementation_sha256",
        "validator_profile",
        "validator_sha256",
        "input_schema_sha256",
        "output_schema_sha256",
    )
    for field in required_fields:
        assert field in impl
        assert isinstance(impl[field], str) and impl[field] != ""
        if field.endswith("_sha256"):
            assert impl[field].startswith("sha256:")

    bundle_dict = ctx.to_dict()
    bundle_dict.pop("context_id", None)
    corrupted_impl = dict(impl)
    corrupted_impl["evidence_id"] = "sha256:" + "0" * 64
    corrupted_impl.pop("validator_sha256")
    bundle_dict["host_facts"]["leaf_bindings"]["minecraft/item/registry"]["state"] = "admitted"
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

    from minecraft_mod_ai.task_template_catalog import load_template
    template = load_template("fabric/item/register_basic")
    validation_ctx = _validation_context(template, required_symbols=("register_item",))
    commented_out = """
    // Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, null);
    public static final Item RAW_LUNITE = null;
    """
    with pytest.raises(VersionContextError) as exc_info:
        validation_ctx.validate_artifact(template, commented_out)
    assert "INVALID_API_INVOCATION" in str(exc_info.value)

    string_only = """
    String s = "Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, null);";
    public static final Item RAW_LUNITE = null;
    """
    with pytest.raises(VersionContextError) as exc_info:
        validation_ctx.validate_artifact(template, string_only)
    assert "INVALID_API_INVOCATION" in str(exc_info.value)

    valid_with_comment = """
    /* Uses net.minecraft.client.gui.screen.Screen internally */
    public static final Item RAW_LUNITE = Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, new Item(new Item.Properties()));
    """
    res = validation_ctx.validate_artifact(template, valid_with_comment)
    assert res["status"] == "PASS"

    leakage_in_code = """
    net.minecraft.client.gui.screen.Screen screen = null;
    public static final Item RAW_LUNITE = Registry.register(BuiltInRegistries.ITEM, ModItemIds.RAW_LUNITE_KEY, new Item(new Item.Properties()));
    """
    with pytest.raises(VersionContextError) as exc_info:
        validation_ctx.validate_artifact(template, leakage_in_code)
    assert "CROSS_SIDE_LEAKAGE" in str(exc_info.value)
