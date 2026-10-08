from __future__ import annotations

import sys
from types import ModuleType

import pytest
from PIL import Image, ImageDraw

from minecraft_mod_ai import resource_alpha_segmentation as alpha


def test_birefnet_model_selected_explicitly_never_uses_bria_default(monkeypatch):
    calls = []
    module = ModuleType("rembg")
    def session(name, **kw):
        calls.append(("session", name, kw))
        return object()
    def remove(img, *, session, alpha_matting):
        calls.append(("remove", img.size, alpha_matting))
        result = img.convert("RGBA")
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse((20, 20, img.width - 20, img.height - 20), fill=255)
        result.putalpha(mask)
        mask.close()
        return result
    module.new_session = session
    module.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", module)
    alpha._get_session.cache_clear()
    try:
        with Image.new("RGB", (256, 256), "grey") as source:
            result = alpha.segment_foreground(source)
            try:
                assert result.mode == "RGBA"
                assert result.size == source.size
                assert result.getpixel((0, 0))[3] == 0
                assert result.getpixel((128, 128))[3] == 255
                assert result.info["mmm_alpha_matte"] == "rembg:birefnet-general"
            finally:
                result.close()
        assert calls == [
            ("session", "birefnet-general", {"providers": ["CPUExecutionProvider"]}),
            ("remove", (256, 256), False),
        ]
    finally:
        alpha._get_session.cache_clear()


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_alpha_model_rejects_non_rgba_outputs(monkeypatch, mode):
    module = ModuleType("rembg")
    module.new_session = lambda *args, **kwargs: object()
    module.remove = lambda image, **kwargs: Image.new(mode, image.size)
    monkeypatch.setitem(sys.modules, "rembg", module)
    alpha._get_session.cache_clear()
    try:
        with Image.new("RGB", (64, 64)) as source:
            with pytest.raises(ValueError, match="ALPHA_SEGMENTER_MISSING_RGBA_ALPHA"):
                alpha.segment_foreground(source)
    finally:
        alpha._get_session.cache_clear()


def test_alpha_model_rejects_all_opaque_masks(monkeypatch):
    module = ModuleType("rembg")
    module.new_session = lambda *args, **kwargs: object()
    module.remove = lambda image, **kwargs: Image.new("RGBA", image.size, (20, 30, 40, 255))
    monkeypatch.setitem(sys.modules, "rembg", module)
    alpha._get_session.cache_clear()
    try:
        with Image.new("RGB", (64, 64)) as source:
            with pytest.raises(ValueError, match="ALPHA_SEGMENTER_FAILED_TO_EXTRACT_SUBJECT"):
                alpha.segment_foreground(source)
    finally:
        alpha._get_session.cache_clear()


def test_image_profile_hash_is_bound_to_matte_model_identity():
    from minecraft_mod_ai.model_registry import ModelRegistry
    from minecraft_mod_ai.resource_prompt_compiler import image_profile_fingerprint
    cfg = ModelRegistry().role("t4_local", "image_generator")
    assert image_profile_fingerprint(cfg).startswith("sha256:")
    assert alpha.ALPHA_SEGMENTATION_MODEL == "birefnet-general"
    assert alpha.ALPHA_SEGMENTATION_CONTRACT.endswith("-v1")
