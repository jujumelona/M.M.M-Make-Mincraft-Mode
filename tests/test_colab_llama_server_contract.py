from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import colab_mtp_server
from minecraft_mod_ai import complete_orchestrator_services
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


def _literal_print_prefixes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            values.append(first.value)
    return values


def test_colab_mtp_cell_uses_pinned_launcher_without_source_build() -> None:
    for notebook in NOTEBOOKS:
        source = _cell_source(notebook, "mtp-server")
        assert "start_colab_mtp_server" in source
        assert "LLAMA_SERVER_URL" in source
        for token in ("git clone", "cmake", "nvcc", "make -j"):
            assert token not in source


def test_colab_installs_exact_binary_only_cuda_wheel() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert 'LLAMA_CPP_CUDA_WHEEL_VERSION = "0.3.34"' in text
    assert (
        "v0.3.34-cu124/"
        "llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"
    ) in text
    assert '"--only-binary=:all:"' in text
    assert "_LLAMA_CPP_CUDA_PROBE" in text
    assert "libggml-cuda.so" in text
    assert "llama_supports_gpu" not in text


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
    assert "libggml-cuda.so" in text
    assert "llama_supports_gpu" not in text
    for token in ("git clone", "cmake", "nvcc", "make -j"):
        assert token not in text


def test_colab_mtp_startup_is_fail_fast_and_bounded() -> None:
    text = MTP_LAUNCHER.read_text(encoding="utf-8")
    assert 'START_TIMEOUT_ENV = "MMM_COLAB_MTP_SERVER_START_TIMEOUT"' in text
    assert '"-u"' in text
    assert "_PROCESS.poll()" in text
    assert "exited during startup" in text
    assert "startup timed out" in text
    assert "Port 8910 is already serving a different unmanaged process" in text
    assert "CUDA backend library is missing" in text


def test_colab_mtp_status_lines_are_state_only() -> None:
    prefixes = _literal_print_prefixes(MTP_LAUNCHER)
    assert prefixes
    assert all(value.startswith("MTP server:") for value in prefixes)
    required = {
        "MTP server: checking CUDA binding",
        "MTP server: resolving model",
        "MTP server: model ready",
        "MTP server: preparing launcher",
        "MTP server: starting",
        "MTP server: ready",
    }
    assert required.issubset(set(prefixes))


def test_colab_mtp_ready_requires_expected_model_alias(monkeypatch) -> None:
    class Response:
        status_code = 200

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def json(self):
            return {"data": [{"id": self.model_id}]}

    monkeypatch.setattr(colab_mtp_server.httpx, "get", lambda *args, **kwargs: Response("other"))
    assert colab_mtp_server._ready() is False

    monkeypatch.setattr(colab_mtp_server.httpx, "get", lambda *args, **kwargs: Response("local"))
    assert colab_mtp_server._ready() is True


def test_colab_mtp_stop_releases_process_and_preserves_restart_intent(monkeypatch) -> None:
    class Process:
        def __init__(self) -> None:
            self.alive = True
            self.terminated = False

        def poll(self):
            return None if self.alive else 0

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def wait(self, timeout=None):
            return 0

    process = Process()
    monkeypatch.setattr(colab_mtp_server, "_PROCESS", process)
    monkeypatch.setattr(colab_mtp_server, "_LOG_HANDLE", None)
    monkeypatch.setenv(colab_mtp_server.ENABLED_ENV, "1")
    monkeypatch.setenv("LLAMA_SERVER_URL", colab_mtp_server.SERVER_API_URL)

    colab_mtp_server.stop_colab_mtp_server(keep_enabled=True)

    assert process.terminated is True
    assert colab_mtp_server._PROCESS is None
    assert os.environ.get(colab_mtp_server.ENABLED_ENV) == "1"
    assert "LLAMA_SERVER_URL" not in os.environ


def test_autotune_restarts_enabled_colab_mtp_server(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(colab_mtp_server, "colab_mtp_server_enabled", lambda: True)

    def fake_start(config):
        calls.append(config)
        return "http://127.0.0.1:8910/v1"

    monkeypatch.setattr(colab_mtp_server, "start_colab_mtp_server", fake_start)
    config = SimpleNamespace()
    result = llama_server_autotune.ensure_tuned_server(config, SimpleNamespace())

    assert result == "http://127.0.0.1:8910/v1"
    assert calls == [config]
    assert getattr(
        llama_server_autotune.ensure_tuned_server,
        "_mmm_colab_mtp_restart",
        False,
    )


def test_image_generation_wrapper_releases_colab_mtp_before_exclusive_gpu() -> None:
    assert getattr(
        complete_orchestrator_services.generate_assets,
        "_mmm_releases_colab_mtp",
        False,
    )
    source = inspect.getsource(llama_server_hardware_policy.install)
    assert "stop_colab_mtp_server" in source
    assert "keep_enabled=True" in source


def test_external_server_probe_accepts_new_server_endpoints() -> None:
    source = inspect.getsource(llama_server_autotune._external_server_is_ready)
    assert '"/v1/models"' in source
    assert '"/healthz"' in source
    assert '"/health"' in source


def test_colab_setup_status_is_plain_and_factual() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert 'print("checkout: validating"' in text
    assert 'print("CUDA: checking"' in text
    assert 'print("project dependencies: installing"' in text
    assert 'print("project dependencies: installed"' in text


def test_llama_adapter_does_not_label_generic_server_as_mtp() -> None:
    text = LLAMA_ADAPTER.read_text(encoding="utf-8")
    assert 'print("llama server: connected"' in text


def test_engine_native_server_discovery_never_compiles_source() -> None:
    source = inspect.getsource(llama_server_hardware_policy._bootstrap_native_server)
    for token in ("subprocess", "cmake", "git", "nvcc"):
        assert token not in source
    assert getattr(llama_server_autotune._server_binary, "_mmm_no_source_build", False)
