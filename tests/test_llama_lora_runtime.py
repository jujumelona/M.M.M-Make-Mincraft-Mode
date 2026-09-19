from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_lora_runtime as lora


def _config(*, role: str = "coder", base: str = "Qwen/Qwen3.5-9B"):
    return SimpleNamespace(
        role=role,
        extra={
            "foundation_base_model_id": base,
            "lora_adapters": {
                "coding_agentic": {
                    "base_model_id": "Qwen/Qwen3.5-9B",
                    "repo_id": "example/coding-agentic",
                    "filename": "coding-agentic.gguf",
                    "revision": "abc123",
                    "scale": 1.0,
                    "priority": 100,
                    "roles": ["coder", "coder_safe"],
                    "when_tools": "any",
                }
            },
        },
    )


def test_configured_lora_rejects_base_mismatch() -> None:
    with pytest.raises(RuntimeError, match="targets base"):
        lora.configured_lora_specs(_config(base="Qwen/Qwen3.8-27B"))


def test_request_lora_routes_only_matching_role(monkeypatch) -> None:
    monkeypatch.setattr(
        lora,
        "_adapter_id_map",
        lambda *_args, **_kwargs: {"coding_agentic": 7},
    )
    coder_request = SimpleNamespace(
        metadata={},
        tools=(),
    )
    assert lora.request_lora_payload(
        "http://127.0.0.1:8910/v1",
        _config(),
        coder_request,
    ) == [{"id": 7, "scale": 1.0}]

    planner = _config(role="planner")
    planner_request = SimpleNamespace(
        metadata={"tool_stage": "planning"},
        tools=(),
    )
    assert lora.request_lora_payload(
        "http://127.0.0.1:8910/v1",
        planner,
        planner_request,
    ) == []


def test_lora_launch_args_preload_gguf_and_zero_by_default(monkeypatch, tmp_path) -> None:
    adapter = tmp_path / "coding-agentic.gguf"
    adapter.write_bytes(b"gguf")
    monkeypatch.setattr(lora, "_download_hf_adapter", lambda _spec: str(adapter))
    assert lora.lora_launch_args(_config()) == [
        "--lora",
        str(adapter.resolve()),
        "--lora-init-without-apply",
    ]


def test_huggingface_snapshot_symlink_keeps_gguf_name(monkeypatch, tmp_path) -> None:
    blob = tmp_path / "blobs" / "64e8d88ba7057e8c0c65a07649975232"
    blob.parent.mkdir()
    blob.write_bytes(b"gguf")
    snapshot = tmp_path / "snapshots" / "main" / "coding-agentic.gguf"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    monkeypatch.setattr(lora, "_download_hf_adapter", lambda _spec: str(snapshot))

    artifacts = lora.resolve_lora_artifacts(_config())
    assert artifacts == ((lora.configured_lora_specs(_config())[0], str(snapshot.absolute())),)
    assert artifacts[0][1].endswith("coding-agentic.gguf")
    assert Path(artifacts[0][1]).is_file()


def test_initialize_managed_server_explicitly_zeros_loaded_adapters(monkeypatch) -> None:
    posts = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    monkeypatch.setattr(
        lora.httpx,
        "get",
        lambda *_args, **_kwargs: _Response(
            [{"id": 3, "path": "/cache/coding-agentic.gguf", "scale": 1.0}]
        ),
    )
    monkeypatch.setattr(
        lora.httpx,
        "post",
        lambda _url, *, json, timeout: posts.append((json, timeout)) or _Response([]),
    )
    lora._ADAPTER_IDS_BY_ORIGIN.clear()
    lora.initialize_managed_server_loras(
        "http://127.0.0.1:8910/v1",
        _config(),
    )
    assert posts[0][0] == [{"id": 3, "scale": 0.0}]
    assert lora._ADAPTER_IDS_BY_ORIGIN["http://127.0.0.1:8910"] == {
        "coding_agentic": 3
    }
