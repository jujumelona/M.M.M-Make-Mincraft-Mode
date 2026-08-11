from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "tools" / "colab_runtime_setup.py"
BUILDER = ROOT / "tools" / "build_colab_notebook.py"
AUTOTUNE = ROOT / "minecraft_mod_ai" / "llama_server_autotune.py"
NOTEBOOKS = (
    ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb",
    ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb",
)


def _patch_setup() -> None:
    text = SETUP.read_text(encoding="utf-8")
    start = text.index("def _install_llama_cpp() -> None:")
    end = text.index("\n\ndef _install_project(", start)
    replacement = '''LLAMA_CPP_CUDA_WHEEL_VERSION = "0.3.34"
LLAMA_CPP_CUDA_WHEEL_URL = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v0.3.34-cu124/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"
)


def _llama_cpp_gpu_available() -> bool:
    try:
        import llama_cpp
        return bool(
            hasattr(llama_cpp, "llama_supports_gpu")
            and llama_cpp.llama_supports_gpu()
        )
    except Exception:
        return False


def _install_llama_cpp() -> None:
    """Install and verify the pinned pre-built CUDA wheel without source builds."""
    if _llama_cpp_gpu_available():
        print("llama-cpp-python CUDA wheel: available", flush=True)
        return

    print("llama-cpp-python CUDA wheel: installing", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--only-binary=:all:",
        LLAMA_CPP_CUDA_WHEEL_URL,
        "--no-cache-dir",
    ]
    subprocess.run(cmd, check=True)

    for name in tuple(sys.modules):
        if name == "llama_cpp" or name.startswith("llama_cpp."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    if not _llama_cpp_gpu_available():
        raise RuntimeError(
            "Installed llama-cpp-python wheel does not report CUDA GPU support."
        )
    print(
        "llama-cpp-python CUDA wheel: installed",
        LLAMA_CPP_CUDA_WHEEL_VERSION,
        flush=True,
    )
'''
    SETUP.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def _patch_notebook_builder() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    start_marker = '    (\n        "code",\n        "mtp-server",'
    end_marker = '    (\n        "code",\n        "plan",'
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    block = '''    (
        "code",
        "mtp-server",
        """# @title 4-1. [선택] 로컬 CUDA MTP 서버 실행
from minecraft_mod_ai.colab_mtp_server import start_colab_mtp_server

assert_current_colab_setup()
LLAMA_SERVER_URL = start_colab_mtp_server(planner_config)
print("llama MTP server:", LLAMA_SERVER_URL)
""",
    ),
'''
    BUILDER.write_text(text[:start] + block + text[end:], encoding="utf-8")


def _patch_external_server_probe() -> None:
    text = AUTOTUNE.read_text(encoding="utf-8")
    old = '''def _external_server_is_ready() -> bool:
    explicit = os.environ.get("LLAMA_SERVER_URL", "").strip()
    if not explicit:
        return False
    try:
        import httpx

        origin = explicit.removesuffix("/v1").rstrip("/")
        return httpx.get(f"{origin}/health", timeout=0.5).status_code == 200
    except Exception:
        return False
'''
    new = '''def _external_server_is_ready() -> bool:
    explicit = os.environ.get("LLAMA_SERVER_URL", "").strip()
    if not explicit:
        return False
    try:
        import httpx

        origin = explicit.removesuffix("/v1").rstrip("/")
        for endpoint in ("/v1/models", "/healthz", "/health"):
            try:
                if httpx.get(f"{origin}{endpoint}", timeout=0.5).status_code == 200:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False
'''
    if old not in text:
        raise RuntimeError("expected external-server readiness function not found")
    AUTOTUNE.write_text(text.replace(old, new), encoding="utf-8")


def _regenerate_notebooks() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_colab_notebook.py")],
        cwd=ROOT,
        check=True,
    )


def _verify() -> None:
    setup = SETUP.read_text(encoding="utf-8")
    assert 'LLAMA_CPP_CUDA_WHEEL_VERSION = "0.3.34"' in setup
    assert "v0.3.34-cu124/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl" in setup
    assert '"--only-binary=:all:"' in setup

    autotune = AUTOTUNE.read_text(encoding="utf-8")
    assert '("/v1/models", "/healthz", "/health")' in autotune

    launcher = (ROOT / "minecraft_mod_ai" / "colab_mtp_server.py").read_text(
        encoding="utf-8"
    )
    assert '"draft_model": "draft-mtp"' in launcher
    assert "72adc790598eac9574aec6fc0bf6e994a9cfe732" in launcher

    banned = (
        "git clone",
        "cmake",
        "nvcc",
        "make -j",
        "약 1분",
        "3배",
        "80토큰",
        "무결성 100%",
        "에러 위험 요소 0%",
    )
    for path in (BUILDER, *NOTEBOOKS, ROOT / "minecraft_mod_ai" / "colab_mtp_server.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                raise AssertionError(f"{path}: forbidden token remains: {token}")
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        assert "start_colab_mtp_server" in text


def main() -> int:
    _patch_setup()
    _patch_notebook_builder()
    _patch_external_server_probe()
    _regenerate_notebooks()
    _verify()
    print("colab MTP synchronization: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
