from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_setup_module():
    path = ROOT / "tools" / "colab_runtime_setup.py"
    spec = importlib.util.spec_from_file_location("mmm_colab_runtime_setup_integration_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_torch():
    return SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_capability=lambda _index: (7, 5),
        )
    )


def test_prebuilt_success_never_touches_source_build_toolchain(monkeypatch, tmp_path: Path) -> None:
    module = _load_setup_module()
    binary = tmp_path / "prebuilt" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native")
    binary.chmod(0o755)

    monkeypatch.setattr(module, "_find_verified_native_server", lambda: None)
    monkeypatch.setattr(
        module,
        "_ensure_prebuilt_native_server",
        lambda *, cuda_arch: str(binary),
    )
    monkeypatch.setattr(module, "_verify_native_server", lambda _path: (True, "verified"))
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: (_ for _ in ()).throw(AssertionError(f"source tool queried: {name}")),
    )

    resolved = module._ensure_native_server(_fake_torch())

    assert resolved == str(binary.resolve())
    assert os.environ["MMM_LLAMA_SERVER_BIN"] == str(binary.resolve())


def test_prebuilt_failure_uses_source_build_only_when_explicitly_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_setup_module()
    source = tmp_path / "llama.cpp"
    final_binary = source / "build" / "bin" / "llama-server"
    final_binary.parent.mkdir(parents=True)
    final_binary.write_bytes(b"source")
    final_binary.chmod(0o755)

    monkeypatch.setattr(module, "_find_verified_native_server", lambda: None)
    monkeypatch.setattr(
        module,
        "_ensure_prebuilt_native_server",
        lambda *, cuda_arch: (_ for _ in ()).throw(RuntimeError("no release")),
    )
    monkeypatch.setenv("MMM_LLAMA_SERVER_SOURCE_DIR", str(source))
    monkeypatch.delenv("MMM_LLAMA_ALLOW_SOURCE_BUILD", raising=False)

    with pytest.raises(RuntimeError, match="automatic source compilation is disabled"):
        module._ensure_native_server(_fake_torch())

    monkeypatch.setenv("MMM_LLAMA_ALLOW_SOURCE_BUILD", "1")
    monkeypatch.setattr(module, "_prepare_native_source", lambda _source: None)

    probed: list[str] = []

    def fake_which(name: str):
        probed.append(name)
        return f"/usr/bin/{name}"

    monkeypatch.setattr(module.shutil, "which", fake_which)
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
    assert "-DGGML_CUDA_CUB_3DOT2=ON" in commands[0]
    assert "-DGGML_CUDA_FA=ON" in commands[0]
    assert "-DGGML_LTO=ON" in commands[0]
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in commands[0]
    assert commands[1][:3] == ["cmake", "--build", str(source / "build")]
    assert os.environ["MMM_LLAMA_SERVER_SOURCE_DIR"] == str(source)
    assert os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] == "source-build"


def test_bundle_loader_is_source_local_and_exposes_verified_installer() -> None:
    module = _load_setup_module()
    bundle = module._load_native_bundle_module()
    assert callable(bundle.ensure_prebuilt_native_server)
    assert bundle.BUNDLE_SCHEMA_VERSION == "mmm/native-llama-cuda-bundle-v4-immutable"
    assert bundle.BUNDLE_RELEASE_TAG == "native-llama-1d2869c-cuda12.4-max-v4-immutable"
