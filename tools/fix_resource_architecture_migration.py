from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tests/test_resource_architecture.py"
    text = path.read_text(encoding="utf-8")
    old = '''def test_asset_manifests_do_not_embed_lora_trigger() -> None:\n    for relative in ("asset/block_tile.yaml", "asset/item_sprite.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):\n        text = (ROOT / "minecraft_mod_ai/templates" / relative).read_text(encoding="utf-8")\n        assert "PixArFK" not in text\n        assert "Pixel Art" not in text\n'''
    new = '''def test_obsolete_backend_prompt_manifests_are_removed() -> None:\n    for relative in ("asset/block_tile.yaml", "asset/item_sprite.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):\n        assert not (ROOT / "minecraft_mod_ai/templates" / relative).exists()\n\n    validation = (ROOT / "minecraft_mod_ai/template_contract_validation.py").read_text(encoding="utf-8")\n    for identifier in ("asset/block_tile", "asset/item_sprite", "asset/entity_texture", "asset/gui_panel"):\n        assert identifier not in validation\n'''
    if text.count(old) != 1:
        raise RuntimeError("resource architecture obsolete-manifest test anchor mismatch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
