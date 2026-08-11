from __future__ import annotations

import inspect
import json
from pathlib import Path

from minecraft_mod_ai import colab_mtp_server
from minecraft_mod_ai import llama_server_autotune
from minecraft_mod_ai import llama_server_hardware_policy


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb",
    ROOT / "Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb",
)
SETUP_SCRIPT = ROOT / "tools" / "colab_runtime_setup.py"
LLAMA_ADAPTER = ROOT / "minecraft_mod_ai" / "model_adapters" / "llama_cpp_adapter.py"
MTP_LAUNCHER = ROOT / "minecraft_mod_ai" / "colab_mtp_server.py"


def _cell_source(path: Path, cell_id: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source") or [])
    raise AssertionError(f"missing cell {cell_id!r} in {path.name}")


def test_colab_mtp_cell_uses_pinned_launcher_without_source_build() -> None:
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
        assert "start_colab_mtp_server" in source
        assert "LLAMA_SERVER_URL" in source
        for token in forbidden:
            assert token not in source


def test_pinned_low_level_server_enables_actual_draft_mtp() -> None:
    text = MTP_LAUNCHER.read_text(encoding="utf-8")
    assert colab_mtp_server.LLAMA_CPP_PYTHON_VERSION == "0.3.34"
    assert (
        colab_mtp_server.SERVER_SOURCE_GIT_BLOB_SHA1
        == "72adc790598eac9574aec6fc0bf6e994a9cfe732"
    )
    assert "examples/server/server.py" in colab_mtp_server.SERVER_SOURCE_URL
    assert '"draft_model": "draft-mtp"' in text
    assert '"draft_model_num_pred_tokens": width' in text
    assert "_git_blob_sha1(data)" in text
    for token in ("git clone", "cmake", "nvcc", "make -j"):
        assert token not in text


def test_colab_status_text_has_no_absolute_integrity_or_zero_risk_claims() -> None:
    forbidden = (
        "무결성 100%",
        "에러 위험 요소 0%",
        "폭속",
        "속도 극대화",
        "풀옵션",
    )
    for path in (*NOTEBOOKS, SETUP_SCRIPT, LLAMA_ADAPTER, MTP_LAUNCHER):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_colab_setup_status_is_plain_and_factual() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")
    for token in ("✅", "⚡", "🔍", "🔧", "ℹ️", "🔄"):
        assert token not in text
    assert 'print("checkout: validating"' in text
    assert 'print("CUDA: checking"' in text
    assert 'print("project dependencies: installing"' in text
    assert 'print("project dependencies: installed"' in text


def test_llama_adapter_does_not_label_generic_server_as_cpp_mtp() -> None:
    text = LLAMA_ADAPTER.read_text(encoding="utf-8")
    assert "C++ MTP" not in text
    assert "llama-server MTP" not in text
    assert 'print("llama server: connected"' in text


def test_engine_native_server_discovery_never_compiles_source() -> None:
    source = inspect.getsource(llama_server_hardware_policy._bootstrap_native_server)
    for token in ("subprocess", "cmake", "git", "nvcc"):
        assert token not in source
    assert getattr(llama_server_autotune._server_binary, "_mmm_no_source_build", False)
