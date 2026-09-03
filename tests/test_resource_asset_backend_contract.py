from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from PIL import Image

from minecraft_mod_ai import resource_asset_production as assets
from minecraft_mod_ai.complete_spec import (
    AssetRequest,
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.evidence_first_planning import (
    EvidencePlanError,
    compile_evidence_first_plan,
)
from minecraft_mod_ai.evidence_task_receipt_contract import (
    build_task_receipt_extensions,
)
from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.production_contract import compile_production_contract
from minecraft_mod_ai.spec import SpecValidationError
from minecraft_mod_ai.source_transplant import DonorSlice, donor_closure_sha256


def _asset() -> AssetRequest:
    return AssetRequest(
        asset_id="diamond_sword",
        kind="item",
        prompt="blue lightsaber replacement",
        target_path="assets/minecraft/textures/item/diamond_sword.png",
        width=16,
        height=16,
    )


def _evidence_proposal(
    *,
    prompt: str = "Implement trade and quests.",
    plan: dict | None = None,
    with_production_contract: bool = False,
):
    design = {
        "pitch": "Implement two independently verified capabilities.",
        "modules": [
            {"plugin_id": "trade", "reason": "trade"},
            {"plugin_id": "quests", "reason": "quests"},
        ],
        "acceptance_tests": ["trade works", "quests work"],
        "_platform_selection": {
            "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
            "reuse_plan": {
                "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
                "capability_graph": {
                    "nodes": ["trade", "quests"],
                    "edges": [],
                    "sources": [],
                },
                "capabilities": [
                    {"capability": "trade", "mode": "fresh", "source_id": ""},
                    {"capability": "quests", "mode": "fresh", "source_id": ""},
                ],
            },
        },
    }
    plan = plan or compile_evidence_first_plan(prompt, design)
    requirements = {
        item["requirement_id"]: item
        for item in plan["request_catalog"]["requirements"]
    }
    receipt_extensions = build_task_receipt_extensions(plan)
    modules = []
    for task in reversed(plan["tasks"]):
        context = {
            "prompt_sha256": plan["request_catalog"]["prompt_sha256"],
            "requirements": [
                requirements[reference]
                for reference in task["requirement_refs"]
            ],
        }
        assert context == receipt_extensions[task["task_id"]]["request_context"]
        embedded = {**task, **copy.deepcopy(receipt_extensions[task["task_id"]])}
        config = {
            "summary": "misleading unrelated fallback tokens",
            "name": "quests" if "trade" in task["task_id"] else "trade",
            "batch_id": task["task_id"],
            "scope": task["semantic_outcome"],
            "evidence_plan_sha256": plan["plan_sha256"],
            "evidence_task": embedded,
            **{
                key: copy.deepcopy(task[key])
                for key in (
                    "requirement_refs",
                    "gap_refs",
                    "reuse_refs",
                    "owned_anchors",
                    "consumes",
                    "provides",
                    "acceptance",
                    "impact_probes",
                )
            },
            "model_fill": {},
        }
        modules.append(
            ProductionModule(
                module_id=task["task_id"],
                kind="custom_java",
                config=config,
                depends_on=tuple(task["depends_on"]),
                required_gates=tuple(task["required_gates"]),
            )
        )
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(prompt)
    acceptance_tests = tuple(design["acceptance_tests"])
    if with_production_contract:
        compiled = compile_production_contract(
            requested_prompt=prompt,
            game_design={key: value for key, value in design.items() if not key.startswith("_")},
            modules=tuple(modules),
            acceptance_tests=acceptance_tests,
            evidence_plan=plan,
        )
        design = {**design, "_production_contract": compiled.contract}
        acceptance_tests = compiled.acceptance_tests
    return complete_proposal_from_parts(
        requested_prompt=prompt,
        base_proposal=base,
        game_design={**design, "_evidence_first_plan": plan},
        modules=tuple(modules),
        acceptance_tests=acceptance_tests,
    )


def test_image_backend_is_fixed_to_klein9b_q4_pixelart_lora() -> None:
    assert assets.FLUX_MODEL_ID == "black-forest-labs/FLUX.2-klein-9B"
    assert assets.QUANTIZATION == "bnb_4bit_nf4"
    assert assets.PIXEL_LORA_ID == "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B"
    assert assets.PIXEL_LORA_WEIGHT == "[FLUX.2.Klein]PixelArt_Redmond.safetensors"

    config = ModelRegistry().role("t4_local", "image_generator")
    assert config.model_id == assets.FLUX_MODEL_ID
    assert config.quantization == assets.QUANTIZATION
    assert config.extra["lora_model_id"] == assets.PIXEL_LORA_ID
    assert config.extra["lora_weight_name"] == assets.PIXEL_LORA_WEIGHT
    assert config.extra["lora_trigger"] == assets.PIXEL_LORA_TRIGGER


def test_backend_config_refuses_lora_or_model_drift() -> None:
    good = SimpleNamespace(
        model_id=assets.FLUX_MODEL_ID,
        quantization=assets.QUANTIZATION,
        extra={
            "lora_model_id": assets.PIXEL_LORA_ID,
            "lora_weight_name": assets.PIXEL_LORA_WEIGHT,
            "lora_trigger": assets.PIXEL_LORA_TRIGGER,
            "lora_scale": 1.0,
        },
    )
    assets._require_fixed_backend_config(good)
    bad = SimpleNamespace(
        model_id=assets.FLUX_MODEL_ID,
        quantization=assets.QUANTIZATION,
        extra={**good.extra, "lora_model_id": "wrong/lora"},
    )
    with pytest.raises(ModelConfigurationError):
        assets._require_fixed_backend_config(bad)


def test_persisted_plan_requires_four_real_prompts_and_exact_fixed_lora() -> None:
    request = _asset()
    prompts = [
        f"{assets.PIXEL_LORA_TRIGGER}. variant {index} blue lightsaber item, transparent background"
        for index in range(4)
    ]
    plan = {
        "schema_version": assets.PLAN_SCHEMA,
        "model_id": assets.FLUX_MODEL_ID,
        "quantization": assets.QUANTIZATION,
        "lora_model_id": assets.PIXEL_LORA_ID,
        "lora_weight_name": assets.PIXEL_LORA_WEIGHT,
        "lora_trigger": assets.PIXEL_LORA_TRIGGER,
        "prompt_candidates_per_asset": 4,
        "assets": [{
            "asset_id": request.asset_id,
            "kind": request.kind,
            "target_path": request.target_path,
            "width": 16,
            "height": 16,
            "prompts": prompts,
        }],
    }
    assert assets._valid_plan(plan, (request,))
    plan["assets"][0]["prompts"] = prompts[:3]
    assert not assets._valid_plan(plan, (request,))


def test_pixel_postprocess_uses_exact_rgba_size(tmp_path) -> None:
    source = tmp_path / "candidate.png"
    Image.new("RGBA", (256, 256), (10, 20, 30, 255)).save(source)
    result = assets._prepare_candidate(_asset(), 0, "prompt", source)
    with Image.open(result["normalized_path"]) as image:
        image.load()
        assert image.size == (16, 16)
        assert image.mode == "RGBA"


def test_capabilities_are_assigned_once_to_matching_modules() -> None:
    modules = (
        ProductionModule(module_id="trade_engine", kind="economy"),
        ProductionModule(module_id="shop_ui", kind="gui"),
    )
    decisions = (
        {"capability": "trade.transaction", "mode": "fresh"},
        {"capability": "ui.shop_menu", "mode": "fresh"},
    )
    ownership = assets._assign_capability_owners(modules, decisions)
    flat = [item["capability"] for rows in ownership.values() for item in rows]
    assert sorted(flat) == ["trade.transaction", "ui.shop_menu"]
    assert sum(item["capability"] == "trade.transaction" for rows in ownership.values() for item in rows) == 1
    assert sum(item["capability"] == "ui.shop_menu" for rows in ownership.values() for item in rows) == 1


def test_evidence_reuse_ownership_uses_exact_task_provides_not_tokens() -> None:
    proposal = _evidence_proposal()

    rebound = assets.bind_reuse_plan(proposal)

    owners = {
        module.module_id: module.config.get("_owned_capabilities", [])
        for module in rebound.modules
        if module.config.get("_owned_capabilities")
    }
    assert len(owners) == 2
    for module in rebound.modules:
        task = module.config["evidence_task"]
        final_capabilities = {
            value.removeprefix("capability:")
            for value in task["provides"]
            if value.startswith("capability:")
        }
        if final_capabilities:
            assert module.config["_owned_capabilities"] == list(final_capabilities)
            owned = module.config["_owned_reuse_plan"]
            assert owned["evidence_plan_sha256"] == rebound.game_design[
                "_evidence_first_plan"
            ]["plan_sha256"]
            assert {
                item["capability"] for item in owned["capabilities"]
            } == final_capabilities
    assert rebound.game_design["_reuse_plan"]["schema_version"] == (
        "mmm/evidence-bound-reuse-plan-v1"
    )


def test_evidence_reuse_binding_rejects_module_ref_drift() -> None:
    proposal = _evidence_proposal()
    first = proposal.modules[0]
    corrupted = ProductionModule(
        module_id=first.module_id,
        kind=first.kind,
        config={**first.config, "reuse_refs": ["component:not-in-the-plan"]},
        depends_on=first.depends_on,
        required_gates=first.required_gates,
    )
    proposal = type(proposal)(
        **{
            **proposal.__dict__,
            "modules": (corrupted, *proposal.modules[1:]),
            "approval_hash": "",
        }
    ).with_hash()

    with pytest.raises(SpecValidationError, match="module binding changed 'reuse_refs'"):
        assets.bind_reuse_plan(proposal)


def test_evidence_reuse_binding_rejects_stale_plan_hash() -> None:
    proposal = _evidence_proposal()
    plan = copy.deepcopy(proposal.game_design["_evidence_first_plan"])
    plan["plan_sha256"] = "sha256:" + "0" * 64
    proposal = type(proposal)(
        **{
            **proposal.__dict__,
            "game_design": {**proposal.game_design, "_evidence_first_plan": plan},
            "approval_hash": "",
        }
    ).with_hash()

    with pytest.raises(EvidencePlanError, match="plan hash mismatch"):
        assets.bind_reuse_plan(proposal)


def test_evidence_reuse_binding_refreshes_v2_module_hash_contract() -> None:
    proposal = _evidence_proposal(with_production_contract=True)
    old_module_hash = proposal.game_design["_production_contract"]["source_bindings"][
        "module_input_sha256"
    ]

    rebound = assets.bind_reuse_plan(proposal)

    assert rebound.schema_version == "mmm/complete-proposal-v2"
    assert rebound.game_design["_production_contract"]["source_bindings"][
        "module_input_sha256"
    ] != old_module_hash
    rebound.validate()


def test_evidence_reuse_binding_carries_only_exact_hashed_component_refs() -> None:
    prompt = "Implement trade."
    donor = {
        "schema_version": "mmm/source-transplant-slice-v1",
        "capability": "trade",
        "repository": "owner/trade-mod",
        "commit_sha": "a" * 40,
        "license_id": "MIT",
        "target_compatibility": "adapt",
        "files": [
            {
                "path": "src/main/java/example/Trade.java",
                "blob_sha": "b" * 40,
                "sha256": "sha256:" + "c" * 64,
                "size_bytes": 100,
            }
        ],
    }
    closure_hash = donor_closure_sha256(DonorSlice.from_dict(donor))
    reuse = {
        "capability_graph": {"nodes": ["trade"], "edges": [], "sources": []},
        "capabilities": [
            {
                "capability": "trade",
                "mode": "source_transplant",
                "source_id": "owner/trade-mod",
                "component_refs": ["trade_candidate"],
                "donor": donor,
                "proof_receipt": {
                    "schema_version": "mmm/reuse-proof-receipt-v1",
                    "candidate_id": "owner/trade-mod@" + "a" * 40,
                    "capability": "trade",
                    "commit_sha": "a" * 40,
                    "closure_hash": closure_hash,
                    "proof_level": "COMPILE_VERIFIED",
                    "authoritative_compile": True,
                    "compile_passed": True,
                    "verified_capabilities": ["trade"],
                    "verified_artifacts": ["src/main/java/example/Trade.java"],
                },
            }
        ],
    }
    design = {
        "modules": [{"plugin_id": "trade", "reason": "trade"}],
        "acceptance_tests": ["trade works"],
        "_platform_selection": {
            "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
            "reuse_plan": reuse,
        },
    }
    components = [
        {
            "component_id": "trade_candidate",
            "kind": "symbol",
            "locator": "src/main/java/example/Trade.java#Trade",
            "content_sha256": "sha256:" + "d" * 64,
            "provides": ["capability:trade"],
            "provenance": {
                "origin": "external",
                "repository": "owner/trade-mod",
                "revision": "a" * 40,
                "license": "MIT",
                "dependency_closure_verified": True,
            },
            "compatibility": {"minecraft_version": "1.21.1", "loader": "fabric"},
        }
    ]
    plan = compile_evidence_first_plan(
        prompt,
        design,
        component_catalog=components,
        reuse_plan=reuse,
    )
    proposal = _evidence_proposal(prompt=prompt, plan=plan)

    rebound = assets.bind_reuse_plan(proposal)

    owner = next(
        module for module in rebound.modules if module.config.get("_owned_capabilities")
    )
    decision = owner.config["_owned_reuse_plan"]["evidence_decisions"][0]
    assert owner.config["_owned_component_refs"] == ["trade_candidate"]
    assert decision["component_refs"] == ["trade_candidate"]
    assert decision["decision_sha256"].startswith("sha256:")
    assert owner.config["_owned_reuse_plan"]["capabilities"][0]["donor"] == donor
