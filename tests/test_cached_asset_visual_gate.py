"""Cached resource receipts must not bypass current image acceptance."""
from __future__ import annotations

import hashlib
from PIL import Image

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator


def _receipt(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "status": "TEXTURE_PRODUCTION_PASS",
        "assets": [{
            "target": str(path),
            "target_path": "src/main/resources/assets/demo/textures/block/crystal.png",
            "sha256": "sha256:" + digest,
        }],
        "documents": [],
        "resource_graph_validation": {"status": "PASS"},
        "resource_contract_validation": {"status": "PASS"},
    }


def test_resume_rejects_old_monochrome_texture(tmp_path):
    image = tmp_path / "crystal.png"
    Image.new("RGBA", (16, 16), (0, 0, 0, 255)).save(image)
    assert not CompleteProductionOrchestrator._cached_asset_shard(_receipt(image))


def test_resume_accepts_unchanged_nonuniform_texture(tmp_path):
    image = tmp_path / "crystal.png"
    rendered = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
    rendered.putpixel((8, 8), (24, 72, 95, 255))
    rendered.save(image)
    assert CompleteProductionOrchestrator._cached_asset_shard(_receipt(image))


def test_resume_rejects_tampered_texture(tmp_path):
    image = tmp_path / "crystal.png"
    rendered = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
    rendered.putpixel((8, 8), (24, 72, 95, 255))
    rendered.save(image)
    receipt = _receipt(image)
    rendered.putpixel((9, 9), (72, 30, 95, 255))
    rendered.save(image)
    assert not CompleteProductionOrchestrator._cached_asset_shard(receipt)
