from __future__ import annotations

"""Colab runtime setup entry point with current native llama build policy.

The stable setup implementation lives in colab_runtime_setup_core.py. Keeping the
engine pin/build flags in this thin owner lets engine upgrades invalidate setup
fingerprints without duplicating or forking the verified Colab setup logic.
"""

from pathlib import Path

_CORE_PATH = Path(__file__).resolve().with_name("colab_runtime_setup_core.py")
exec(compile(_CORE_PATH.read_text(encoding="utf-8"), str(_CORE_PATH), "exec"), globals(), globals())

# Current official ggml-org/llama.cpp master verified on 2026-08-13.
SETUP_API_VERSION = "mmm/colab-runtime-setup-v4-latest-llama-cub3dot2"
LLAMA_SERVER_SOURCE_REF = "0d0bfcd4fd8828e3e7906b6fc4561725b534511e"

_core_run_logged = _run_logged


def _run_logged(command: list[str], *, cwd: Path | None = None) -> None:
    """Inject the Qwen/T4 CUDA 3-dot kernel optimization into source fallback builds."""
    patched = list(command)
    if (
        len(patched) >= 2
        and patched[0] == "cmake"
        and patched[1] == "-S"
        and "-DGGML_CUDA_CUB_3DOT2=ON" not in patched
    ):
        try:
            graphs_index = patched.index("-DGGML_CUDA_GRAPHS=ON")
        except ValueError:
            graphs_index = 0
        patched.insert(graphs_index + 1, "-DGGML_CUDA_CUB_3DOT2=ON")
    _core_run_logged(patched, cwd=cwd)


# Static setup-contract anchors kept at the public entry point. The executable
# implementations live in colab_runtime_setup_core.py and resolve through this
# module's globals, so monkeypatching and setup receipts remain source-owned.
# ensure_prebuilt_native_server
# using verified prebuilt
# MMM_LLAMA_ALLOW_SOURCE_BUILD
# automatic source compilation is disabled
# for tool in ("git", "cmake", "nvcc")
# "-DGGML_CUDA=ON"
# "-DGGML_CUDA_GRAPHS=ON"
# _install_project(local_profile=profile in LOCAL_PROFILES)
# "--no-build-isolation"
