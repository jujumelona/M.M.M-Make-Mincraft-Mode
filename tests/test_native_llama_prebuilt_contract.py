from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_bundle_helper():
    path = ROOT / "tools" / "native_llama_bundle.py"
    spec = importlib.util.spec_from_file_location("mmm_native_llama_bundle_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_colab_setup():
    path = ROOT / "tools" / "colab_runtime_setup.py"
    spec = importlib.util.spec_from_file_location("mmm_colab_runtime_setup_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_requires_verified_prebuilt_before_emergency_source_toolchain() -> None:
    source = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")

    prebuilt = source.index("ensure_prebuilt_native_server")
    source_toolchain = source.index('for tool in ("git", "cmake", "nvcc")')
    assert prebuilt < source_toolchain
    assert "using verified prebuilt" in source
    assert "MMM_LLAMA_ALLOW_SOURCE_BUILD" in source
    assert "automatic source compilation is disabled" in source
    assert "falling back to pinned source build" not in source


def test_emergency_source_build_enables_cuda_graphs() -> None:
    source = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")
    fallback = source[source.index('for tool in ("git", "cmake", "nvcc")') :]
    assert '"-DGGML_CUDA=ON"' in fallback
    assert '"-DGGML_CUDA_GRAPHS=ON"' in fallback


def test_colab_project_install_reuses_satisfied_packages_and_prefers_wheels() -> None:
    source = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")
    install = source[source.index("def _install_project") : source.index("def _configure_output")]
    assert '"--prefer-binary"' in install
    assert '"--upgrade"' not in install


def test_native_version_probe_is_memoized_for_unchanged_binary(monkeypatch, tmp_path: Path) -> None:
    setup = _load_colab_setup()
    root = tmp_path / "bundle" / "bin"
    root.mkdir(parents=True)
    binary = root / "llama-server"
    backend = root / "libggml-cuda.so.0"
    binary.write_bytes(b"server")
    backend.write_bytes(b"cuda")
    binary.chmod(0o755)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="llama-server test")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    first = setup._verify_native_server(binary)
    second = setup._verify_native_server(binary)
    assert first[0] is True
    assert second == first
    assert calls == [(str(binary), "--version")]

    backend.write_bytes(b"cuda-changed")
    third = setup._verify_native_server(binary)
    assert third[0] is True
    assert len(calls) == 2


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


def test_bundle_workflows_are_independent_graph_enabled_cuda_builds() -> None:
    worker = (
        ROOT / ".github" / "workflows" / "build-native-llama-cuda.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_call:" in worker
    assert "cuda_arch:" in worker
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in worker
    assert "BUNDLE_SCHEMA: mmm/native-llama-cuda-bundle-v2" in worker
    assert "RELEASE_TAG: native-llama-b10375-cuda12.4-v2" in worker
    assert "-DBUILD_SHARED_LIBS=ON" in worker
    assert "-DGGML_CUDA=ON" in worker
    assert "-DGGML_CUDA_GRAPHS=ON" in worker
    assert '"cuda_graphs": True' in worker
    assert "libggml-cuda.so*" in worker
    assert "sha256sum -c" in worker
    assert "gh git" in worker
    assert "gh --version" in worker
    assert "Publish this architecture immediately" in worker

    for arch in ("75", "80", "89"):
        wrapper = (
            ROOT / ".github" / "workflows" / f"build-native-llama-sm{arch}.yml"
        ).read_text(encoding="utf-8")
        assert "push:" in wrapper
        assert "uses: ./.github/workflows/build-native-llama-cuda.yml" in wrapper
        assert f'cuda_arch: "{arch}"' in wrapper
        assert "contents: write" in wrapper


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
