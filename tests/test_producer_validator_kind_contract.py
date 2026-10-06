from __future__ import annotations

from minecraft_mod_ai.artifact_expansion import FACT_TO_CANONICAL_LEAVES
from minecraft_mod_ai.complete_spec import ASSET_KINDS, MODULE_KINDS, ProductionModule
from minecraft_mod_ai.content_design_contract import CONTENT_KIND_TO_FACT_TYPE
from minecraft_mod_ai.platform_backend_contract import NATIVE_PRODUCTION_MODULE_KINDS
from minecraft_mod_ai.resource_asset_production import _assign_capability_owners
from minecraft_mod_ai.resource_contracts import SUPPORTED_RENDER_KINDS, infer_render_kind
from minecraft_mod_ai.typed_platform_ir import PLATFORM_HOST_KINDS, PLATFORM_KINDS


def test_typed_platform_emitted_kinds_are_final_spec_kinds() -> None:
    emitted = set(PLATFORM_KINDS) - set(PLATFORM_HOST_KINDS)
    assert emitted <= set(MODULE_KINDS)


def test_native_dispatch_kinds_are_final_spec_kinds() -> None:
    assert set(NATIVE_PRODUCTION_MODULE_KINDS) <= set(MODULE_KINDS)


def test_every_content_semantic_fact_has_artifact_expansion() -> None:
    assert set(CONTENT_KIND_TO_FACT_TYPE.values()) <= set(FACT_TO_CANONICAL_LEAVES)


def test_every_asset_kind_has_a_supported_default_render_kind() -> None:
    for kind in ASSET_KINDS:
        assert infer_render_kind(kind) in SUPPORTED_RENDER_KINDS


def test_reuse_fallback_uses_current_typed_host_kind() -> None:
    modules = (
        ProductionModule("content_item", "item", {}),
        ProductionModule(
            "authored_typed_plan",
            "typed_host",
            {"typed_plan_ir": {}},
        ),
    )
    decisions = ({"capability": "no semantic token matches"},)
    owners = _assign_capability_owners(modules, decisions)
    assert tuple(owners) == (1,)
    assert owners[1] == decisions
