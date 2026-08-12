from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _forbidden_markers() -> tuple[str, ...]:
    # Build the literals so this regression test does not flag itself.
    return (
        "llama-cpp" + "-python",
        "LLAMA_CPP_" + "PYTHON_VERSION",
        "llama_cpp_" + "python-",
        "abetlen/" + "llama-cpp-python",
        "from llama_cpp" + " import",
        "import " + "llama_cpp",
        "colab_" + "mtp_server",
        "mmm_llama_" + "mtp_server",
    )


def _production_files() -> list[Path]:
    files: list[Path] = []
    for directory in (ROOT / "minecraft_mod_ai", ROOT / "tools"):
        files.extend(path for path in directory.rglob("*.py") if path.is_file())
    for name in (
        "pyproject.toml",
        "requirements-colab.txt",
        "M.M.M_Make_Mincraft_Mode_Colab.ipynb",
    ):
        path = ROOT / name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def test_production_source_contains_no_legacy_python_llama_runtime() -> None:
    forbidden = _forbidden_markers()
    violations: list[str] = []
    for path in _production_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in forbidden:
            if marker in text:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    assert violations == []


def test_local_runtime_is_native_server_only() -> None:
    setup = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")
    adapter = (
        ROOT / "minecraft_mod_ai" / "model_adapters" / "llama_cpp_adapter.py"
    ).read_text(encoding="utf-8")
    hardware = (
        ROOT / "minecraft_mod_ai" / "llama_server_hardware_policy.py"
    ).read_text(encoding="utf-8")

    assert "-DGGML_CUDA=ON" in setup
    assert "-DLLAMA_BUILD_SERVER=ON" in setup
    assert "LLAMA_SERVER_SOURCE_REF" in setup
    assert "ensure_tuned_server" in adapter
    assert "_strict_server_generate" in hardware

    for legacy in (
        "minecraft_mod_ai/colab_mtp_server.py",
        "minecraft_mod_ai/colab_llama_request_routing_contract.py",
        "minecraft_mod_ai/colab_server_config_contract.py",
    ):
        assert not (ROOT / legacy).exists()
