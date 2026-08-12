from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_setup_module():
    path = Path("tools/colab_runtime_setup.py").resolve()
    spec = importlib.util.spec_from_file_location("_mmm_test_colab_runtime_setup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_torch(major: int = 7, minor: int = 5):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_capability=lambda _index: (major, minor),
        )
    )


def test_prebuilt_success_never_touches_source_build_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_setup_module()
    binary = tmp_path / "prebuilt/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("prebuilt", encoding="utf-8")

    monkeypatch.delenv("MMM_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_DISTRIBUTION", raising=False)
    monkeypatch.setattr(module, "_find_verified_native_server", lambda: None)

    calls: list[str] = []

    def prebuilt(*, cuda_arch: str):
        calls.append(cuda_arch)
        os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] = "prebuilt-test"
        return str(binary)

    monkeypatch.setattr(module, "_ensure_prebuilt_native_server", prebuilt)
    monkeypatch.setattr(
        module,
        "_verify_native_server",
        lambda path: (path.resolve() == binary.resolve(), "verified"),
    )

    def forbidden_which(_tool: str):
        raise AssertionError("prebuilt success must not probe git/cmake/nvcc")

    monkeypatch.setattr(module.shutil, "which", forbidden_which)
    monkeypatch.setattr(
        module,
        "_prepare_native_source",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("prebuilt success must not prepare source")
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_logged",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prebuilt success must not compile source")
        ),
    )

    resolved = module._ensure_native_server(_fake_torch())

    assert resolved == str(binary.resolve())
    assert calls == ["75"]
    assert os.environ["MMM_LLAMA_SERVER_BIN"] == str(binary.resolve())
    assert os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] == "prebuilt-test"


def test_prebuilt_failure_falls_back_to_exact_pinned_source_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_setup_module()
    source = tmp_path / "llama.cpp"
    final_binary = source / "build/bin/llama-server"

    monkeypatch.delenv("MMM_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_SOURCE_DIR", raising=False)
    monkeypatch.delenv("MMM_LLAMA_SERVER_DISTRIBUTION", raising=False)
    monkeypatch.setattr(module, "_find_verified_native_server", lambda: None)
    monkeypatch.setattr(
        module,
        "_ensure_prebuilt_native_server",
        lambda *, cuda_arch: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(module, "_native_source_dir", lambda: source)
    monkeypatch.setattr(module, "_prepare_native_source", lambda path: path.mkdir(parents=True))

    probed: list[str] = []

    def which(tool: str):
        probed.append(tool)
        return f"/usr/bin/{tool}"

    monkeypatch.setattr(module.shutil, "which", which)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "_run_logged",
        lambda command, *, cwd=None: commands.append(list(command)),
    )
    monkeypatch.setattr(module, "_verify_native_server", lambda _path: (True, "verified"))

    resolved = module._ensure_native_server(_fake_torch())

    assert resolved == str(final_binary.resolve())
    assert probed == ["git", "cmake", "nvcc"]
    assert commands[0][:2] == ["cmake", "-S"]
    assert "-DGGML_CUDA_GRAPHS=ON" in commands[0]
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in commands[0]
    assert commands[1][:3] == ["cmake", "--build", str(source / "build")]
    assert os.environ["MMM_LLAMA_SERVER_SOURCE_DIR"] == str(source)
    assert os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] == "source-build"


def test_bundle_loader_is_source_local_and_exposes_verified_installer() -> None:
    module = _load_setup_module()
    bundle = module._load_native_bundle_module()
    assert callable(bundle.ensure_prebuilt_native_server)
    assert bundle.BUNDLE_SCHEMA_VERSION == "mmm/native-llama-cuda-bundle-v2"
