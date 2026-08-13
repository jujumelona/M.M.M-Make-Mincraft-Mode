from __future__ import annotations

import hashlib
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


def test_emergency_source_build_enables_max_cuda_path() -> None:
    source = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")
    fallback = source[source.index('for tool in ("git", "cmake", "nvcc")') :]
    assert '"-DGGML_CUDA=ON"' in fallback
    assert '"-DGGML_CUDA_GRAPHS=ON"' in fallback
    assert '"-DGGML_CUDA_CUB_3DOT2=ON"' in fallback
    assert '"-DGGML_CUDA_FA=ON"' in fallback
    assert '"-DGGML_LTO=ON"' in fallback


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


def test_bundle_workflows_are_parallel_graph_enabled_cuda_builds() -> None:
    worker = (
        ROOT / ".github" / "workflows" / "build-native-llama-cuda.yml"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / ".github" / "workflows" / "build-native-llama-cuda-all.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_call:" in worker
    assert "cuda_arch:" in worker
    assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in worker
    assert "BUNDLE_SCHEMA: mmm/native-llama-cuda-bundle-v3-max-t4" in worker
    assert "RELEASE_TAG: native-llama-4a84b0a-cuda12.4-max-v3" in worker
    assert "-DBUILD_SHARED_LIBS=ON" in worker
    assert "-DGGML_CUDA=ON" in worker
    assert "-DGGML_CUDA_GRAPHS=ON" in worker
    assert "-DGGML_CUDA_CUB_3DOT2:BOOL=ON" in worker
    assert "-DGGML_CUDA_FA=ON" in worker
    assert "-DGGML_LTO=ON" in worker
    assert '"cuda_graphs": True' in worker
    assert "sha256sum -c" in worker
    assert "gh git" in worker
    assert "gh --version" in worker
    assert "Publish this architecture immediately" in worker

    # Packaging stores one real copy per shared object and reconstructs aliases later.
    assert "Package one copy of each shared library" in worker
    assert "resolved = entry.resolve(strict=True)" in worker
    assert '"aliases": dict(sorted(aliases.items()))' in worker
    assert "bundle must contain regular files only" in worker
    assert "cp -L" not in worker

    # One launcher creates three independent reusable-workflow jobs concurrently.
    assert 'cuda_arch: ["75", "80", "89"]' in launcher
    assert "fail-fast: false" in launcher
    assert "uses: ./.github/workflows/build-native-llama-cuda.yml" in launcher
    assert "cuda_arch: ${{ matrix.cuda_arch }}" in launcher
    assert "contents: write" in launcher
    assert "cancel-in-progress: true" in launcher

    # Native compilation must not run for unrelated main commits.
    assert "paths:" in launcher
    assert "- .github/workflows/build-native-llama-cuda-all.yml" in launcher
    assert "- .github/workflows/build-native-llama-cuda.yml" in launcher
    assert "- tools/native_llama_bundle.py" in launcher

    for arch in ("75", "80", "89"):
        assert not (
            ROOT / ".github" / "workflows" / f"build-native-llama-sm{arch}.yml"
        ).exists()


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
    assert helper.BUNDLE_SCHEMA_VERSION == "mmm/native-llama-cuda-bundle-v3-max-t4"
    assert helper.BUNDLE_RELEASE_TAG == "native-llama-4a84b0a-cuda12.4-max-v3"

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
                "cuda_cub_3dot2": True,
                "cuda_fa": True,
                "lto": True,
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requires cuda_graphs=true"):
        helper._validate_bundle(root, cuda_arch="75", source_ref="source-ref")


def test_bundle_loader_materializes_only_verified_bin_aliases(tmp_path: Path) -> None:
    helper = _load_bundle_helper()
    root = tmp_path / "bundle"
    bindir = root / "bin"
    bindir.mkdir(parents=True)
    server = bindir / "llama-server"
    cuda_real = bindir / "libggml-cuda.so.0.19.0"
    server.write_bytes(b"server")
    cuda_real.write_bytes(b"cuda-real")
    files = {
        "bin/llama-server": hashlib.sha256(server.read_bytes()).hexdigest(),
        "bin/libggml-cuda.so.0.19.0": hashlib.sha256(cuda_real.read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": helper.BUNDLE_SCHEMA_VERSION,
        "llama_source_ref": "source-ref",
        "cuda_arch": "75",
        "platform": "linux-x86_64",
        "cuda_graphs": True,
        "cuda_cub_3dot2": True,
        "cuda_fa": True,
        "lto": True,
        "files": files,
        "aliases": {
            "bin/libggml-cuda.so": "bin/libggml-cuda.so.0.19.0",
            "bin/libggml-cuda.so.0": "bin/libggml-cuda.so.0.19.0",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    binary = helper._validate_bundle(root, cuda_arch="75", source_ref="source-ref")
    assert binary == server.resolve()
    for name in ("libggml-cuda.so", "libggml-cuda.so.0"):
        alias = bindir / name
        assert alias.is_symlink()
        assert alias.resolve() == cuda_real.resolve()

    malicious = dict(manifest)
    malicious["aliases"] = {"../escape": "bin/libggml-cuda.so.0.19.0"}
    (root / "manifest.json").write_text(json.dumps(malicious), encoding="utf-8")
    with pytest.raises(RuntimeError, match="alias path is unsafe"):
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
