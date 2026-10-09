from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def test_isolated_worker_transfers_real_alpha_without_loading_onnx_in_parent(monkeypatch):
    events = []
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)

    def fake_run(command, **kwargs):
        events.append((command, kwargs))
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            output = raw.convert("RGBA")
        with Image.new("L", output.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((10, 10, 50, 50), fill=255)
            output.putalpha(mask)
        output.save(target, "PNG")
        output.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (64, 64), "grey") as source:
        output = alpha.segment_foreground_isolated(source)
        try:
            assert output.mode == "RGBA"
            assert output.getpixel((0, 0))[3] == 0
            assert output.getpixel((32, 32))[3] == 255
            assert output.info["mmm_alpha_matte"] == "rembg:birefnet-general"
        finally:
            output.close()
    assert len(events) == 1
    assert events[0][0][1:4] == ["-m", "minecraft_mod_ai.resource_alpha_segmentation", "--worker"]
    assert events[0][1]["timeout"] == 300


def test_insufficient_ram_does_not_start_native_worker(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 512 * 1024**2)
    monkeypatch.setenv("MMM_ALPHA_MIN_AVAILABLE_MB", "3072")
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Native worker should not be started when host RAM is low")
    monkeypatch.setattr(alpha.subprocess, "run", unexpected_run)
    with Image.new("RGB", (32, 32)) as source:
        with pytest.raises(ValueError, match="ALPHA_SEGMENTER_INSUFFICIENT_HOST_RAM"):
            alpha.segment_foreground_isolated(source)


def test_native_worker_sigkill_becomes_diagnostic_error(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        alpha.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=-9, stderr="", stdout=""),
    )
    with Image.new("RGB", (32, 32)) as source:
        with pytest.raises(ValueError, match="ALPHA_SEGMENTER_WORKER_OOM_SUSPECT"):
            alpha.segment_foreground_isolated(source)


def test_native_worker_timeout_becomes_diagnostic_error(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)
    def timed_out(*args, **kwargs):
        raise alpha.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(alpha.subprocess, "run", timed_out)
    with Image.new("RGB", (32, 32)) as source:
        with pytest.raises(ValueError, match="ALPHA_SEGMENTER_WORKER_TIMEOUT"):
            alpha.segment_foreground_isolated(source)
