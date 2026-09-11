from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _patch_proposal_deserialization_installer() -> None:
    path = ROOT / "minecraft_mod_ai/proposal_deserialization_contract.py"
    text = path.read_text(encoding="utf-8")
    old = "__all__ = ['install_proposal_deserialization_contracts']\n"
    new = '''def install(spec_module: object, complete_spec_module: object) -> None:\n    """Bind the centralized strict deserializer to the public spec modules."""\n    from .capabilities import capability_manifest_hash\n    from .knowledge import evidence_snapshot_hash\n\n    install_proposal_deserialization_contracts(\n        proposal_cls=spec_module.Proposal,\n        proposal_status_cls=spec_module.ProposalStatus,\n        spec_validation_error=spec_module.SpecValidationError,\n        content_spec_cls=spec_module.ContentSpec,\n        content_kind_cls=spec_module.ContentKind,\n        boss_spec_cls=spec_module.BossSpec,\n        mod_spec_cls=spec_module.ModSpec,\n        deferred_request_cls=spec_module.DeferredRequest,\n        evidence_source_cls=spec_module.EvidenceSource,\n        capability_manifest_hash=capability_manifest_hash,\n        evidence_snapshot_hash=evidence_snapshot_hash,\n        json_bool=spec_module._json_bool,\n        complete_proposal_cls=complete_spec_module.CompleteProposal,\n        complete_proposal_status_cls=complete_spec_module.CompleteProposalStatus,\n        production_module_cls=complete_spec_module.ProductionModule,\n        asset_request_cls=complete_spec_module.AssetRequest,\n    )\n\n\n__all__ = ['install', 'install_proposal_deserialization_contracts']\n'''
    if text.count(old) != 1:
        raise RuntimeError("proposal deserialization installer anchor mismatch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _patch_resource_architecture_test() -> None:
    path = ROOT / "tests/test_resource_architecture.py"
    text = path.read_text(encoding="utf-8")
    old = '''def test_asset_manifests_do_not_embed_lora_trigger() -> None:\n    for relative in ("asset/block_tile.yaml", "asset/item_sprite.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):\n        text = (ROOT / "minecraft_mod_ai/templates" / relative).read_text(encoding="utf-8")\n        assert "PixArFK" not in text\n        assert "Pixel Art" not in text\n'''
    new = '''def test_obsolete_backend_prompt_manifests_are_removed() -> None:\n    for relative in ("asset/block_tile.yaml", "asset/item_sprite.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):\n        assert not (ROOT / "minecraft_mod_ai/templates" / relative).exists()\n\n    validation = (ROOT / "minecraft_mod_ai/template_contract_validation.py").read_text(encoding="utf-8")\n    for identifier in ("asset/block_tile", "asset/item_sprite", "asset/entity_texture", "asset/gui_panel"):\n        assert identifier not in validation\n'''
    if text.count(old) != 1:
        raise RuntimeError("resource architecture obsolete-manifest test anchor mismatch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    _patch_proposal_deserialization_installer()
    _patch_resource_architecture_test()


if __name__ == "__main__":
    main()
