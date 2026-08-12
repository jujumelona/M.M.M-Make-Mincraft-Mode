from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import complete_orchestrator_services
from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai.llama_server_autotune import (
    ProbeResult,
    ServerVariant,
    _base_args,
    _candidate_variants,
    _choose_variant,
    _compact_benchmark_request,
    _probe_server,
    _server_binary,
    _variant_args,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def _probe(
    name: str,
    *,
    tps: float,
    output_sha256: str = "same",
    ok: bool = True,
    width: int = 0,
) -> ProbeResult:
    variant = (
        ServerVariant("baseline")
        if name == "baseline"
        else ServerVariant(name, "draft-mtp", width)
    )
    return ProbeResult(
        variant=variant,
        ok=ok,
        output_sha256=output_sha256,
        predicted_tokens=96 if ok else 0,
        predicted_tps=tps,
        prompt_tps=100.0,
        elapsed_seconds=1.0,
    )


def test_server_autotune_contract_is_installed_without_duplicate_probe() -> None:
    assert getattr(LlamaCppAdapter.generate, "_mmm_explicit_server_strict", False)
    assert not getattr(LlamaCppAdapter.generate, "_mmm_server_autotuned", False)
    assert getattr(_server_binary, "_mmm_native_bootstrap", False)
    assert getattr(_base_args, "_mmm_auto_gpu_layers", False)
    assert getattr(_base_args, "_mmm_single_decode_slot", False)
    assert getattr(_base_args, "_mmm_native_telemetry_endpoints", False)
    assert getattr(_variant_args, "_mmm_auto_draft_layers", False)
    assert getattr(_probe_server, "_mmm_compact_decode_probe", False)
    assert not getattr(_probe_server, "_mmm_correctness_sentinel", False)
    assert getattr(
        complete_orchestrator_services.generate_assets,
        "_mmm_releases_managed_llama",
        False,
    )


def test_default_variants_compare_baseline_and_bounded_mtp_widths(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_MTP_WIDTHS", raising=False)
    values = _candidate_variants()
    assert [value.name for value in values] == ["baseline", "mtp-1", "mtp-2", "mtp-3"]
    assert values[0].spec_type == "none"
    assert [value.draft_n_max for value in values[1:]] == [1, 2, 3]


def test_compact_benchmark_never_reuses_real_workflow_prompt() -> None:
    secret = "REAL-WORKFLOW-PROMPT-THAT-MUST-NOT-BE-PREFILLED-PER-VARIANT"
    request = SimpleNamespace(
        messages=({"role": "user", "content": secret},),
        response_format="json",
    )
    compact = _compact_benchmark_request(request)
    rendered = "\n".join(str(message["content"]) for message in compact.messages)
    assert secret not in rendered
    assert compact.response_format == "text"
    assert len(rendered) < 512


def test_autotune_requires_exact_output_match_before_speed() -> None:
    decision = _choose_variant(
        (
            _probe("baseline", tps=20.0, output_sha256="baseline"),
            _probe("mtp-1", tps=40.0, output_sha256="different", width=1),
            _probe("mtp-2", tps=26.0, output_sha256="baseline", width=2),
        ),
        minimum_speedup=1.03,
    )
    assert decision is not None
    assert decision.selected.name == "mtp-2"
    assert decision.selected_tps == 26.0
    assert decision.speedup == 1.3


def test_autotune_keeps_baseline_when_gain_is_below_threshold() -> None:
    decision = _choose_variant(
        (
            _probe("baseline", tps=20.0),
            _probe("mtp-1", tps=20.4, width=1),
        ),
        minimum_speedup=1.03,
    )
    assert decision is not None
    assert decision.selected.name == "baseline"


def test_autotune_ignores_failed_fast_candidate() -> None:
    decision = _choose_variant(
        (
            _probe("baseline", tps=20.0),
            _probe("mtp-1", tps=999.0, ok=False, width=1),
        ),
        minimum_speedup=1.01,
    )
    assert decision is not None
    assert decision.selected.name == "baseline"


def test_autotune_fails_closed_without_valid_baseline() -> None:
    decision = _choose_variant(
        (_probe("baseline", tps=0.0, ok=False),),
        minimum_speedup=1.03,
    )
    assert decision is None


def test_disabling_autotune_still_launches_native_baseline(monkeypatch) -> None:
    selected: list[ServerVariant] = []
    monkeypatch.setenv("MMM_LLAMA_SERVER_AUTOTUNE", "0")
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", None)
    monkeypatch.setattr(autotune, "_MANAGED_URL", None)
    monkeypatch.setattr(autotune, "_ATTEMPTED_KEYS", set())
    monkeypatch.setattr(autotune, "_external_server_is_ready", lambda: False)
    monkeypatch.setattr(autotune, "_server_binary", lambda: "/tmp/llama-server")
    monkeypatch.setattr(autotune, "_resolve_model_path", lambda config: "/tmp/model.gguf")
    monkeypatch.setattr(autotune, "_fingerprint", lambda *args: "fingerprint")
    monkeypatch.setattr(autotune, "_load_cached_decision", lambda fingerprint: None)
    monkeypatch.setattr(
        autotune,
        "_launch_selected",
        lambda binary, model, config, variant: selected.append(variant)
        or "http://127.0.0.1:8910/v1",
    )

    config = SimpleNamespace(model_id="test", extra={}, max_context=1024, max_new_tokens=64)
    request = SimpleNamespace(messages=(), response_format="text")
    url = autotune.ensure_tuned_server(config, request)

    assert url.endswith("/v1")
    assert [value.name for value in selected] == ["baseline"]


def test_server_args_match_quality_neutral_runtime_defaults(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_BATCH", raising=False)
    monkeypatch.delenv("MMM_LLAMA_UBATCH", raising=False)
    monkeypatch.delenv("MMM_KV_CACHE_QUANT", raising=False)
    config = SimpleNamespace(max_context=32768)
    args = _base_args("llama-server", "/tmp/model.gguf", config, 8910)
    assert args[args.index("--ctx-size") + 1] == "16384"
    assert args[args.index("--batch-size") + 1] == "2048"
    assert args[args.index("--ubatch-size") + 1] == "512"
    assert args[args.index("--gpu-layers") + 1] == "auto"
    assert args[args.index("--parallel") + 1] == "1"
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--cache-type-k") + 1] == "q4_0"
    assert args[args.index("--cache-type-v") + 1] == "q4_0"
    assert args[args.index("--load-mode") + 1] == "none"
    assert "--metrics" in args
    assert "--slots" in args


def test_mtp_variant_uses_server_startup_flags_not_request_mutation() -> None:
    args = _variant_args(ServerVariant("mtp-2", "draft-mtp", 2))
    assert args == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--spec-draft-n-min",
        "0",
        "--spec-draft-ngl",
        "auto",
    ]
