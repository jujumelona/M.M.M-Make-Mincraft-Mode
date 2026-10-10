"""Current native llama.cpp T4 decode policy; no retired Qwen-specific hotpath shim."""
from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.llama_structured_decode_policy import bind_structured_decode_policy


def _mimo_config():
    return SimpleNamespace(
        model_id="MiMo-V2.6-9B",
        extra={"gguf_filename": "MiMo-V2.6-9B.gguf"},
        max_context=32768,
        max_new_tokens=8192,
    )


def test_native_t4_server_launch_keeps_full_gpu_flash_attention_and_fit(monkeypatch):
    monkeypatch.setattr(autotune, "_assert_mimo_server_compatible", lambda *_: None)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_BATCH", raising=False)
    monkeypatch.delenv("MMM_LLAMA_UBATCH", raising=False)
    monkeypatch.delenv("MMM_KV_CACHE_QUANT", raising=False)
    args = autotune._base_args("llama-server", "/tmp/model.gguf", _mimo_config(), 8910)
    for flag, expected in (
        ("--gpu-layers", "all"),
        ("--flash-attn", "on"),
        ("--fit", "on"),
        ("--load-mode", "none"),
    ):
        assert args[args.index(flag) + 1] == expected
    assert "--slots" not in args
    assert "--ctx-size" not in args


def test_native_speculative_variant_keeps_draft_on_gpu():
    variant = autotune.ServerVariant("mtp-4", "draft-mtp", 4)
    args = autotune._variant_args(variant)
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-ngl") + 1] == "all"
    assert args[args.index("--spec-draft-n-max") + 1] == "4"


def test_native_mtp_selection_requires_byte_identical_output():
    baseline = autotune.ServerVariant("baseline")
    good = autotune.ServerVariant("mtp-2", "draft-mtp", 2)
    altered = autotune.ServerVariant("mtp-3", "draft-mtp", 3)

    def probe(variant, sha, speed):
        return autotune.ProbeResult(
            variant=variant,
            ok=True,
            output_sha256=sha,
            predicted_tokens=96,
            predicted_tps=speed,
            prompt_tps=10.0,
            elapsed_seconds=1.0,
        )

    decision = autotune._choose_variant(
        (
            probe(baseline, "original-output", 10.0),
            probe(good, "original-output", 12.0),
            probe(altered, "different-output", 22.0),
        ),
        minimum_speedup=1.03,
    )
    assert decision is not None
    assert decision.selected == good
    assert decision.selected_tps == 12.0


def test_structured_game_design_schema_remains_host_owned_for_mimo():
    module = SimpleNamespace(
        _server_payload=lambda _adapter, _request: {
            "response_format": {"type": "json_object", "schema": {}},
            "reasoning_effort": "none",
            "max_tokens": 8192,
        }
    )
    bind_structured_decode_policy(module)
    adapter = SimpleNamespace(config=_mimo_config())
    schema = {"type": "object", "properties": {"game_design": {}}}
    request = SimpleNamespace(
        response_format="json",
        response_schema=schema,
        metadata={"mmm_force_non_thinking": True},
    )
    payload = module._server_payload(adapter, request)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["json_schema"]["type"] == "object"
    assert "game_design" in payload["json_schema"]["properties"]
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
