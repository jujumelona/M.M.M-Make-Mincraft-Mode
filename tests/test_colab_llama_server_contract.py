from __future__ import annotations

import inspect
import json
from pathlib import Path

from minecraft_mod_ai import llama_server_autotune
from minecraft_mod_ai import llama_server_hardware_policy


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb",
    ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb",
)


def _cell_source(path: Path, cell_id: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source") or [])
    raise AssertionError(f"missing cell {cell_id!r} in {path.name}")


def test_colab_llama_server_uses_installed_cuda_python_server_without_source_build() -> None:
    forbidden = (
        "git clone",
        "cmake",
        "nvcc",
        "make -j",
        "최초 1회",
        "약 1분",
        "3배",
        "80토큰",
    )
    for notebook in NOTEBOOKS:
        source = _cell_source(notebook, "mtp-server")
        assert "llama_cpp.server" in source
        assert "LLAMA_SERVER_URL" in source
        assert "n_gpu_layers" in source
        for token in forbidden:
            assert token not in source


def test_colab_status_text_has_no_absolute_integrity_or_zero_risk_claims() -> None:
    forbidden = (
        "무결성 100%",
        "에러 위험 요소 0%",
        "폭속",
        "속도 극대화",
        "풀옵션",
    )
    for notebook in NOTEBOOKS:
        text = notebook.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_engine_native_server_discovery_never_compiles_source() -> None:
    source = inspect.getsource(llama_server_hardware_policy._bootstrap_native_server)
    for token in ("subprocess", "cmake", "git", "nvcc"):
        assert token not in source
    assert getattr(llama_server_autotune._server_binary, "_mmm_no_source_build", False)
