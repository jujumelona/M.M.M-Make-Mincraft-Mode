from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_bundle_helper():
    path = ROOT / "tools" / "native_llama_bundle.py"
    spec = importlib.util.spec_from_file_location("mmm_native_llama_bundle_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_prefers_verified_prebuilt_before_source_toolchain() -> None:
    source = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")

    prebuilt = source.index("ensure_prebuilt_native_server")
    source_toolchain = source.index('for tool in ("git", "cmake", "nvcc")')
    assert prebuilt < source_toolchain
    assert "prebuilt CUDA bundle ready" in source
    assert "falling back to source build" in source


def test_bundle_workflow_builds_shared_cuda_for_supported_colab_arches() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "build-native-llama-cuda.yml"
    ).read_text(encoding="utf-8")

    assert 'cuda_arch: ["75", "80", "89"]' in workflow
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in workflow
    assert "-DBUILD_SHARED_LIBS=ON" in workflow
    assert "-DGGML_CUDA=ON" in workflow
    assert "libggml-cuda.so*" in workflow
    assert "sha256sum -c" in workflow


def test_bundle_loader_rejects_archive_path_traversal(tmp_path: Path) -> None:
    helper = _load_bundle_helper()
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"escape"

    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe prebuilt archive path"):
        helper._safe_extract(archive, tmp_path / "out")

    assert not (tmp_path.parent / "escape").exists()
