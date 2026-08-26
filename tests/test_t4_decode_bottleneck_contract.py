from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from minecraft_mod_ai import llama_server_hardware_policy as hardware_policy
from minecraft_mod_ai.llama_structured_decode_policy import (
    bind_structured_decode_policy,
)
from minecraft_mod_ai.qwen35_mtp_hotpath_contract import (
    _disable_decode_slot_polling,
    _install_draft_kv_benchmark,
    _install_measured_fast_base_args,
    _install_measured_fast_variant_args,
)


def _qwen_config():
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={
            "gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
            "runtime_contract": "qwen",
            "decode_hotpath": "t4_mtp",
        },
        max_context=32768,
        max_new_tokens=8192,
    )


def test_qwen_hotpath_removes_decode_slot_endpoint_and_polls(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)

    autotune = SimpleNamespace(
        _base_args=lambda *_: [
            "llama-server",
            "--slots",
            "--gpu-layers",
            "auto",
            "--flash-attn",
            "off",
            "--ctx-size",
            "16384",
        ]
    )
    _install_measured_fast_base_args(autotune)
    args = autotune._base_args("server", "model", _qwen_config(), 8910)
    assert "--slots" not in args
    assert args[args.index("--gpu-layers") + 1] == "all"
    assert args[args.index("--flash-attn") + 1] == "on"

    calls = []

    def old_snapshot(*args, **kwargs):
        calls.append((args, kwargs))
        return {"output_tokens": 1}

    monkeypatch.setattr(hardware_policy, "_slot_snapshot", old_snapshot)
    _disable_decode_slot_polling()
    assert hardware_policy._slot_snapshot(object(), "http://127.0.0.1:8910/v1") is None
    assert calls == []


def test_qwen_mtp_variant_keeps_draft_gpu_and_selected_draft_kv(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MTP_ACTIVE_TUNING", "1")
    monkeypatch.delenv("MMM_QWEN35_MTP_DRAFT_NGL", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_DRAFT_KV", raising=False)

    autotune = SimpleNamespace(
        _variant_args=lambda variant: [
            "--spec-type",
            variant.spec_type,
            "--spec-draft-ngl",
            "auto",
        ]
    )
    _install_measured_fast_variant_args(autotune)
    variant = SimpleNamespace(spec_type="draft-mtp", name="mtp-4|dkv-q8_0")
    args = autotune._variant_args(variant)

    assert args[args.index("--spec-draft-ngl") + 1] == "all"
    assert args[args.index("--spec-draft-type-k") + 1] == "q8_0"
    assert args[args.index("--spec-draft-type-v") + 1] == "q8_0"


@dataclass(frozen=True)
class _Variant:
    name: str
    spec_type: str = "none"
    draft_n_max: int = 0
    ubatch: int = 0
    parallel: int = 1
    cache_reuse: int = 0
    draft_p_min: float = 0.0


@dataclass(frozen=True)
class _Probe:
    variant: _Variant
    ok: bool
    output_sha256: str
    predicted_tps: float


@dataclass(frozen=True)
class _Decision:
    fingerprint: str
    selected: _Variant
    baseline_tps: float
    selected_tps: float
    speedup: float
    probes: tuple[object, ...]


def test_qwen_draft_kv_stage_selects_fastest_identical_output(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_DRAFT_KV", raising=False)
    monkeypatch.setenv("MMM_QWEN35_MTP_DRAFT_KV_CANDIDATES", "f16,q8_0,q4_0")

    initial = _Decision(
        fingerprint="fp",
        selected=_Variant("mtp-4", "draft-mtp", 4),
        baseline_tps=8.0,
        selected_tps=10.0,
        speedup=1.25,
        probes=(),
    )
    speed = {"f16": 10.0, "q8_0": 12.0, "q4_0": 11.0}

    def run_variant(_binary, _model, _config, _request, variant, *, probe_tokens):
        del probe_tokens
        kv = variant.name.split("|dkv-", 1)[1]
        return _Probe(variant, True, "same-output", speed[kv])

    autotune = SimpleNamespace(
        _benchmark=lambda *_: initial,
        _mmm_run_tuning_variant=run_variant,
        _compact_benchmark_request=lambda request: request,
        _env_int=lambda _name, default: default,
        _env_float=lambda _name, default: default,
        _BENCHMARK_OUTPUT_TOKENS=128,
        AutotuneDecision=_Decision,
    )
    _install_draft_kv_benchmark(autotune)
    decision = autotune._benchmark("server", "model", _qwen_config(), object(), "fp")

    assert decision.selected.name.endswith("|dkv-q8_0")
    assert decision.selected_tps == 12.0
    assert len(decision.probes) == 3


def test_qwen_game_design_schema_uses_native_constraint_fastpath() -> None:
    module = SimpleNamespace(
        _server_payload=lambda _adapter, _request: {
            "response_format": {"type": "json_object", "schema": {}},
            "reasoning_effort": "none",
            "max_tokens": 8192,
        }
    )
    bind_structured_decode_policy(module)
    adapter = SimpleNamespace(config=_qwen_config())
    schema = {"type": "object", "properties": {"game_design": {}}}
    request = SimpleNamespace(
        response_format="json",
        response_schema=schema,
    )

    payload = module._server_payload(adapter, request)
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "mmm_structured_response",
            "strict": True,
            "schema": schema,
        },
    }
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
