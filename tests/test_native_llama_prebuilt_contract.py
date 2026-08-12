from __future__ import annotations

import importlib.util
import io
import json
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
    assert "using verified prebuilt" in source
    assert "falling back to pinned source build" in source


def test_source_fallback_build_enables_cuda_graphs() -> None:
    source = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")
    fallback = source[source.index('for tool in ("git", "cmake", "nvcc")') :]
    assert '"-DGGML_CUDA=ON"' in fallback
    assert '"-DGGML_CUDA_GRAPHS=ON"' in fallback


def test_runtime_package_never_rebuilds_native_llama() -> None:
    runtime_root = ROOT / "minecraft_mod_ai"
    forbidden = (
        'shutil.which("cmake")',
        '["cmake", "--build"',
        '"cmake",\n            "--build"',
        "MMM_LLAMA_CUDA_GRAPHS_BUILD",
    )
    offenders: list[str] = []
    for path in runtime_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(token in source for token in forbidden):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
    assert not (runtime_root / "llama_server_max_performance.py").exists()


def test_bundle_workflow_builds_graph_enabled_shared_cuda_for_supported_colab_arches() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "build-native-llama-cuda.yml"
    ).read_text(encoding="utf-8")

    assert 'cuda_arch: ["75", "80", "89"]' in workflow
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in workflow
    assert "BUNDLE_SCHEMA: mmm/native-llama-cuda-bundle-v2" in workflow
    assert "RELEASE_TAG: native-llama-b10375-cuda12.4-v2" in workflow
    assert "-DBUILD_SHARED_LIBS=ON" in workflow
    assert "-DGGML_CUDA=ON" in workflow
    assert "-DGGML_CUDA_GRAPHS=ON" in workflow
    assert '"cuda_graphs": True' in workflow
    assert "libggml-cuda.so*" in workflow
    assert "sha256sum -c" in workflow


def test_bundle_workflow_uses_driver_stub_only_for_linking() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "build-native-llama-cuda.yml"
    ).read_text(encoding="utf-8")

    assert "stub_dir=/usr/local/cuda/lib64/stubs" in workflow
    assert 'test -f "$stub_dir/libcuda.so"' in workflow
    assert 'ln -sf libcuda.so "$stub_dir/libcuda.so.1"' in workflow
    assert "-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" in workflow
    assert "Shared library: [libcuda.so.1]" in workflow
    assert "forbidden runtime path to the CI CUDA stub" in workflow
    assert "test ! -e bundle/bin/libcuda.so" in workflow
    assert "test ! -e bundle/bin/libcuda.so.1" in workflow


def test_bundle_loader_requires_graph_enabled_v2_manifest(tmp_path: Path) -> None:
    helper = _load_bundle_helper()
    assert helper.BUNDLE_SCHEMA_VERSION == "mmm/native-llama-cuda-bundle-v2"
    assert helper.BUNDLE_RELEASE_TAG == "native-llama-b10375-cuda12.4-v2"

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": helper.BUNDLE_SCHEMA_VERSION,
                "llama_source_ref": "source-ref",
                "cuda_arch": "75",
                "platform": "linux-x86_64",
                "cuda_graphs": False,
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requires cuda_graphs=true"):
        helper._validate_bundle(root, cuda_arch="75", source_ref="source-ref")


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
