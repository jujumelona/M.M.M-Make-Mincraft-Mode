from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from minecraft_mod_ai import resource_alpha_segmentation as alpha


@pytest.fixture(autouse=True)
def fake_onnxruntime_options(monkeypatch):
    ort = ModuleType("onnxruntime")

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0
        execution_mode = None
        enable_mem_pattern = True
        enable_cpu_mem_arena = True

    ort.SessionOptions = Options
    ort.ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    sessions = ModuleType("rembg.sessions")
    session_module = ModuleType("rembg.sessions.birefnet_general_lite")
    mid_module = ModuleType("rembg.sessions.u2net")
    fallback_module = ModuleType("rembg.sessions.u2netp")

    class BiRefNetSessionGeneralLite:
        @classmethod
        def name(cls):
            return "birefnet-general-lite"

        def __init__(self, name, sess_opts, *, providers):
            self.name_used = name
            self.session_options = sess_opts
            self.providers = providers

    class U2netSession(BiRefNetSessionGeneralLite):
        @classmethod
        def name(cls):
            return "u2net"

    class U2netpSession(BiRefNetSessionGeneralLite):
        @classmethod
        def name(cls):
            return "u2netp"

    session_module.BiRefNetSessionGeneralLite = BiRefNetSessionGeneralLite
    mid_module.U2netSession = U2netSession
    fallback_module.U2netpSession = U2netpSession
    monkeypatch.setitem(sys.modules, "rembg.sessions", sessions)
    monkeypatch.setitem(sys.modules, "rembg.sessions.birefnet_general_lite", session_module)
    monkeypatch.setitem(sys.modules, "rembg.sessions.u2net", mid_module)
    monkeypatch.setitem(sys.modules, "rembg.sessions.u2netp", fallback_module)
    alpha._get_session.cache_clear()
    monkeypatch.setattr(alpha, "_LITE_WORKER_SIGKILLED", False)
    monkeypatch.setattr(alpha, "_U2NET_WORKER_SIGKILLED", False)
    yield
    alpha._get_session.cache_clear()


def test_birefnet_model_selected_explicitly_never_uses_bria_default(monkeypatch):
    calls = []
    module = ModuleType("rembg")
    from rembg.sessions.birefnet_general_lite import BiRefNetSessionGeneralLite
    original_init = BiRefNetSessionGeneralLite.__init__

    def track_session(self, name, sess_opts, *, providers):
        calls.append(("session", name, {"providers": providers, "sess_opts": sess_opts}))
        original_init(self, name, sess_opts, providers=providers)

    monkeypatch.setattr(BiRefNetSessionGeneralLite, "__init__", track_session)
    def remove(img, *, session, alpha_matting):
        calls.append(("remove", img.size, alpha_matting))
        result = img.convert("RGBA")
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse((20, 20, img.width - 20, img.height - 20), fill=255)
        result.putalpha(mask)
        mask.close()
        return result
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
                assert result.info["mmm_alpha_matte"] == "rembg:birefnet-general-lite"
            finally:
                result.close()
        assert calls[0][:2] == ("session", "birefnet-general-lite")
        assert calls[0][2]["providers"] == ["CPUExecutionProvider"]
        options = calls[0][2]["sess_opts"]
        assert options.intra_op_num_threads == 1
        assert options.inter_op_num_threads == 1
        assert options.execution_mode == "sequential"
        assert options.enable_mem_pattern is False
        assert options.enable_cpu_mem_arena is False
        assert calls[1] == ("remove", (256, 256), False)
    finally:
        alpha._get_session.cache_clear()


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_alpha_model_rejects_non_rgba_outputs(monkeypatch, mode):
    module = ModuleType("rembg")
    module.new_session = lambda *args, sess_opts=None, **kwargs: object()
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
    module.new_session = lambda *args, sess_opts=None, **kwargs: object()
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
    assert alpha.ALPHA_SEGMENTATION_MODEL == "birefnet-general-lite"
    assert alpha.ALPHA_SEGMENTATION_CONTRACT.endswith("-v3")


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
            assert output.info["mmm_alpha_matte"] == "rembg:birefnet-general-lite"
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


def test_alpha_checkpoint_is_prepared_outside_diffusion(monkeypatch):
    calls = []
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    alpha.prepare_foreground_model_isolated()
    assert len(calls) == 1
    assert calls[0][0][1:4] == [
        "-m", "minecraft_mod_ai.resource_alpha_segmentation", "--prepare"
    ]
    assert calls[0][1]["timeout"] == 480


def test_alpha_checkpoint_preparation_failure_is_not_silent(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        alpha.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=-9, stderr="worker ended", stdout=""
        ),
    )
    with pytest.raises(ValueError, match="ALPHA_SEGMENTER_PREPARE_FAILED"):
        alpha.prepare_foreground_model_isolated()


def test_streamed_checkpoint_works_with_1486_mib_but_inference_remains_guarded(
    monkeypatch,
):
    """The exact Colab log headroom must not block a download-only subprocess."""
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 1486 * 1048576)
    monkeypatch.delenv("MMM_ALPHA_MIN_AVAILABLE_MB", raising=False)
    monkeypatch.delenv("MMM_ALPHA_PREPARE_MIN_AVAILABLE_MB", raising=False)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command[-1])
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    alpha.prepare_foreground_model_isolated()
    assert commands == ["--prepare"]
    with pytest.raises(ValueError, match="ALPHA_SEGMENTER_INSUFFICIENT_HOST_RAM"):
        alpha._preflight_worker_ram()


def test_streamed_checkpoint_still_blocks_unsafe_low_ram(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 200 * 1048576)
    monkeypatch.delenv("MMM_ALPHA_PREPARE_MIN_AVAILABLE_MB", raising=False)
    with pytest.raises(
        ValueError, match="ALPHA_SEGMENTER_PREPARE_INSUFFICIENT_HOST_RAM"
    ):
        alpha.prepare_foreground_model_isolated()


def test_legacy_factory_typeerror_is_reported_as_compatibility_not_oom(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)
    worker_stderr = (
        "Traceback (most recent call last):\n"
        '  File "/usr/local/lib/python3.13/dist-packages/rembg/session_factory.py", line 48\n'
        "TypeError: BaseSession.__init__() got multiple values for argument 'sess_opts'\n"
    )
    monkeypatch.setattr(
        alpha.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr=worker_stderr, stdout=""
        ),
    )
    with Image.new("RGB", (32, 32)) as source:
        with pytest.raises(
            ValueError, match="ALPHA_SEGMENTER_INCOMPATIBLE_REMBG_API"
        ) as err:
            alpha.segment_foreground_isolated(source)
    message = str(err.value)
    assert "Rerun Colab setup cell 2" in message
    assert "multiple values for argument 'sess_opts'" in message
    assert "cgroup_oom_kill_delta=" in message


def test_rembg_2067_direct_session_skips_incompatible_factory(monkeypatch):
    """The legacy factory must not be called, even if it would throw."""
    module = ModuleType("rembg")
    events = []

    def broken_factory(*_args, **_kwargs):
        events.append("factory_called")
        raise TypeError("BaseSession.__init__() got multiple values for argument 'sess_opts'")

    module.new_session = broken_factory
    monkeypatch.setitem(sys.modules, "rembg", module)
    alpha._get_session.cache_clear()

    session = alpha._get_session()
    assert session.name_used == "birefnet-general-lite"
    assert session.providers == ["CPUExecutionProvider"]
    assert session.session_options.intra_op_num_threads == 1
    assert session.session_options.enable_cpu_mem_arena is False
    assert events == []


def test_rembg_direct_session_rejects_wrong_model_class(monkeypatch):
    from rembg.sessions.birefnet_general_lite import BiRefNetSessionGeneralLite

    monkeypatch.setattr(BiRefNetSessionGeneralLite, "name", classmethod(lambda cls: "bria-rmbg"))
    alpha._get_session.cache_clear()
    with pytest.raises(ValueError, match="ALPHA_SEGMENTER_MODEL_CLASS_MISMATCH"):
        alpha._get_session()

def test_sigkill_retries_once_with_licensed_lightweight_fallback(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 12 * 1024**3)
    attempts = []

    def fake_run(command, **kwargs):
        model = kwargs["env"]["MMM_ALPHA_WORKER_MODEL"]
        attempts.append(model)
        if model == alpha.ALPHA_SEGMENTATION_MODEL:
            return SimpleNamespace(returncode=-9, stderr="", stdout="")
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
        with Image.new("L", rgba.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((8, 8, 24, 24), fill=255)
            rgba.putalpha(mask)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
    try:
        assert result.getpixel((16, 16))[3] == 255
        assert result.getpixel((0, 0))[3] == 0
        assert result.info["mmm_alpha_matte"] == "rembg:u2netp"
    finally:
        result.close()
    assert attempts == ["birefnet-general-lite", "u2netp"]


def test_non_sigkill_backend_failure_never_masks_error_with_fallback(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 8 * 1024**3)
    attempts = []

    def fail(command, **kwargs):
        attempts.append(kwargs["env"]["MMM_ALPHA_WORKER_MODEL"])
        return SimpleNamespace(returncode=1, stderr="invalid onnx graph", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fail)
    with Image.new("RGB", (32, 32)) as source:
        with pytest.raises(ValueError, match="ALPHA_SEGMENTER_WORKER_FAILED") as error:
            alpha.segment_foreground_isolated(source)
    assert "invalid onnx graph" in str(error.value)
    assert attempts == ["birefnet-general-lite"]


def test_fallback_uses_explicit_onnx_class_and_provider(monkeypatch):
    fallback = alpha._get_session("u2netp")
    assert fallback.name_used == "u2netp"
    assert fallback.providers == ["CPUExecutionProvider"]
    assert fallback.session_options.enable_cpu_mem_arena is False
    with pytest.raises(ValueError, match="ALPHA_SEGMENTER_MODEL_NOT_ALLOWED"):
        alpha._get_session("bria-rmbg")

def test_colab_observed_6517_mib_skips_sigkill_prone_lite(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 6517 * 1048576)
    attempts = []

    def fake_run(command, **kwargs):
        model = kwargs["env"]["MMM_ALPHA_WORKER_MODEL"]
        attempts.append(model)
        assert model == "u2net"
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as original:
            rgba = original.convert("RGBA")
        with Image.new("L", rgba.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((6, 6, 26, 26), fill=255)
            rgba.putalpha(mask)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
    try:
        assert result.info["mmm_alpha_matte"] == "rembg:u2net"
        assert result.getpixel((16, 16))[3] == 255
        assert result.getpixel((0, 0))[3] == 0
    finally:
        result.close()
    assert attempts == ["u2net"]


def test_sigkill_sets_global_policy_to_skip_lite_next_time(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 12 * 1024**3)
    attempts = []

    def fake_run(command, **kwargs):
        name = kwargs["env"]["MMM_ALPHA_WORKER_MODEL"]
        attempts.append(name)
        if name == "birefnet-general-lite":
            return SimpleNamespace(returncode=-9, stderr="", stdout="")
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            result = raw.convert("RGBA")
        with Image.new("L", result.size, 0) as mask:
            ImageDraw.Draw(mask).rectangle((2, 2, 28, 28), fill=255)
            result.putalpha(mask)
        result.save(target, "PNG")
        result.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    for _ in range(2):
        with Image.new("RGB", (32, 32)) as source:
            output = alpha.segment_foreground_isolated(source)
            output.close()
    assert attempts == ["birefnet-general-lite", "u2netp", "u2netp"]


def test_full_u2net_uses_explicit_licensed_session(monkeypatch):
    session = alpha._get_session("u2net")
    assert session.name_used == "u2net"
    assert session.providers == ["CPUExecutionProvider"]
    assert session.session_options.enable_cpu_mem_arena is False


def test_low_memory_still_uses_tiny_u2netp(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 3900 * 1048576)
    attempts = []

    def fake_run(command, **kwargs):
        attempts.append(kwargs["env"]["MMM_ALPHA_WORKER_MODEL"])
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
        with Image.new("L", rgba.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((4, 4, 28, 28), fill=255)
            rgba.putalpha(mask)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
        result.close()
    assert attempts == ["u2netp"]


def test_intermediate_memory_full_u2net_oom_falls_back_safely(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 5900 * 1048576)
    attempts = []

    def fake_run(command, **kwargs):
        name = kwargs["env"]["MMM_ALPHA_WORKER_MODEL"]
        attempts.append(name)
        if name == "u2net":
            return SimpleNamespace(returncode=-9, stderr="", stdout="")
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
        with Image.new("L", rgba.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((4, 4, 28, 28), fill=255)
            rgba.putalpha(mask)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
        try:
            assert result.info["mmm_alpha_matte"] == "rembg:u2netp"
        finally:
            result.close()
    assert attempts == ["u2net", "u2netp"]

    # The next candidate must not kill another full-size U2Net worker.
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
        result.close()
    assert attempts == ["u2net", "u2netp", "u2netp"]


def test_unusable_full_u2net_mask_recovers_without_regenerating_source(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 5700 * 1048576)
    attempts = []
    sources = []

    def fake_run(command, **kwargs):
        model = kwargs["env"]["MMM_ALPHA_WORKER_MODEL"]
        attempts.append(model)
        source, target = Path(command[-2]), Path(command[-1])
        sources.append(source.read_bytes())
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
        if model == "u2net":
            # A confidently wrong mask must be rejected, not accepted.
            rgba.putalpha(255)
        else:
            with Image.new("L", rgba.size, 0) as mask:
                ImageDraw.Draw(mask).ellipse((4, 4, 28, 28), fill=255)
                rgba.putalpha(mask)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
        try:
            assert result.info["mmm_alpha_matte"] == "rembg:u2netp"
            assert result.getpixel((0, 0))[3] == 0
            assert result.getpixel((16, 16))[3] == 255
        finally:
            result.close()
    assert attempts == ["u2net", "u2netp"]
    assert sources[0] == sources[1]  # No diffusion regeneration.


def test_all_mask_models_fail_without_fabricating_alpha(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 5700 * 1048576)
    attempts = []

    def fake_run(command, **kwargs):
        attempts.append(kwargs["env"]["MMM_ALPHA_WORKER_MODEL"])
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
        rgba.putalpha(255)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        with pytest.raises(ValueError, match="ALPHA_SEGMENTER_UNUSABLE_MASK"):
            alpha.segment_foreground_isolated(source)
    assert attempts == ["u2net", "u2netp"]


def test_explicit_no_subject_worker_error_may_try_next_real_model(monkeypatch):
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 5700 * 1048576)
    attempts = []

    def fake_run(command, **kwargs):
        name = kwargs["env"]["MMM_ALPHA_WORKER_MODEL"]
        attempts.append(name)
        if name == "u2net":
            return SimpleNamespace(
                returncode=1, stderr="ALPHA_SEGMENTER_FAILED_TO_EXTRACT_SUBJECT", stdout=""
            )
        source, target = Path(command[-2]), Path(command[-1])
        with Image.open(source) as raw:
            rgba = raw.convert("RGBA")
        with Image.new("L", rgba.size, 0) as mask:
            ImageDraw.Draw(mask).ellipse((4, 4, 28, 28), fill=255)
            rgba.putalpha(mask)
        rgba.save(target, "PNG")
        rgba.close()
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(alpha.subprocess, "run", fake_run)
    with Image.new("RGB", (32, 32), "grey") as source:
        result = alpha.segment_foreground_isolated(source)
        result.close()
    assert attempts == ["u2net", "u2netp"]
