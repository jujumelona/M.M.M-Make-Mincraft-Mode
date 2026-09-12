from __future__ import annotations

import json
import os
from types import SimpleNamespace

from minecraft_mod_ai import complete_orchestrator_services
from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import llama_server_runtime_tuning as runtime_tuning
from minecraft_mod_ai import llama_vram_parallel_policy as vram_policy
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
from minecraft_mod_ai.llama_server_runtime_tuning import (
    _cache_reuse_candidates,
    _parallel_candidates,
    _ubatch_candidates,
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


def test_server_runtime_tuning_contract_is_installed() -> None:
    assert getattr(LlamaCppAdapter.generate, "_mmm_explicit_server_strict", False)
    assert getattr(_server_binary, "_mmm_native_bootstrap", False)
    assert getattr(_base_args, "_mmm_auto_gpu_layers", False)
    assert getattr(_base_args, "_mmm_single_decode_slot", False)
    assert getattr(_base_args, "_mmm_load_mode_auto", False)
    assert getattr(_variant_args, "_mmm_auto_draft_layers", False)
    assert getattr(_variant_args, "_mmm_ngram_speculation", False)
    assert getattr(_probe_server, "_mmm_compact_decode_probe", False)
    assert getattr(autotune._benchmark, "_mmm_staged_runtime_tuning", False)
    assert getattr(autotune._benchmark, "_mmm_single_server_cache_stage", False)
    assert getattr(autotune._fingerprint, "_mmm_stable_model_signature", False)
    assert getattr(autotune._fingerprint, "_mmm_runtime_tuning_fingerprint", False)
    assert getattr(autotune._cache_path, "_mmm_persistent_tuning_cache", False)
    assert getattr(autotune.ensure_tuned_server, "_mmm_managed_server_fast_path", False)
    assert getattr(autotune.ensure_tuned_server, "_mmm_llama_no_reload_fast_start_v6", False)
    assert getattr(
        complete_orchestrator_services.generate_assets,
        "_mmm_releases_managed_llama",
        False,
    )


def test_default_fast_search_uses_only_safe_baseline_variant(monkeypatch) -> None:
    for name in (
        "MMM_LLAMA_AUTOTUNE_SEARCH",
        "MMM_LLAMA_MTP_WIDTHS",
        "MMM_LLAMA_MTP_CONFIDENCE_WIDTHS",
        "MMM_LLAMA_MTP_SEED_P_MIN",
        "MMM_LLAMA_NGRAM_SPEC_TYPES",
    ):
        monkeypatch.delenv(name, raising=False)
    values = _candidate_variants()
    assert len(values) == 1
    assert values[0].spec_type == "none"
    assert values[0].draft_n_max == 0


def test_default_fast_search_removes_reload_heavy_candidates(monkeypatch) -> None:
    for name in (
        "MMM_LLAMA_AUTOTUNE_SEARCH",
        "MMM_LLAMA_BATCH",
        "MMM_LLAMA_UBATCH",
        "MMM_LLAMA_UBATCH_CANDIDATES",
        "MMM_LLAMA_CACHE_REUSE_CANDIDATES",
        "MMM_LLAMA_CONCURRENT_REQUESTS",
        "MMM_LLAMA_PARALLEL",
        "MMM_PERFORMANCE_MODE",
        "MMM_LLAMA_TUNING_OBJECTIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _ubatch_candidates(autotune) == (512,)
    assert _cache_reuse_candidates() == ()
    assert _parallel_candidates() == (1,)


def test_fast_start_disables_hidden_kernel_and_kv_sweeps_by_default() -> None:
    assert os.environ.get("MMM_LLAMA_KERNEL_AUTOTUNE") == "0"
    assert os.environ.get("MMM_LLAMA_KV_AUTOTUNE") == "0"


def test_vram_fast_start_chooses_highest_safe_width(monkeypatch) -> None:
    monkeypatch.delenv("MMM_LLAMA_PARALLEL", raising=False)
    monkeypatch.delenv("MMM_LLAMA_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("MMM_PERFORMANCE_MODE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TUNING_OBJECTIVE", raising=False)
    monkeypatch.delenv("MMM_LLAMA_MAXIMIZE_PARALLEL", raising=False)
    fake = SimpleNamespace(
        _explicit_parallel=lambda: None,
        _performance_mode=lambda: "auto",
        _runtime_resources=lambda: object(),
        _MAX_PARALLEL=8,
        _parallel_target=lambda: 8,
        _parallel_resource_feasible=lambda slots, _config, _path, _resources: slots <= 4,
    )
    assert vram_policy._recommended_parallel(fake, object(), "/tmp/model.gguf") == 4


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


def test_tuning_fingerprint_is_stable_across_path_and_mtime(monkeypatch, tmp_path) -> None:
    left = tmp_path / "left" / "model.gguf"
    right = tmp_path / "right" / "model.gguf"
    left.parent.mkdir()
    right.parent.mkdir()
    payload = (b"head" * 300_000) + (b"tail" * 300_000)
    left.write_bytes(payload)
    right.write_bytes(payload)
    os.utime(left, (1_700_000_000, 1_700_000_000))
    os.utime(right, (1_800_000_000, 1_800_000_000))
    monkeypatch.setattr(autotune, "_server_version", lambda binary: "server-v1")
    monkeypatch.setattr(autotune, "_hardware_identity", lambda: "GPU, 16 GiB, driver")
    config = SimpleNamespace(
        model_id="repo/model",
        extra={"gguf_filename": "model.gguf"},
        max_context=32768,
        max_new_tokens=8192,
    )
    assert autotune._fingerprint(config, "llama-server", str(left)) == autotune._fingerprint(
        config,
        "llama-server",
        str(right),
    )


def test_drive_output_reuses_autotune_decision_across_runtimes(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "M.M.M-output"
    monkeypatch.delenv("MMM_LLAMA_AUTOTUNE_CACHE", raising=False)
    monkeypatch.setenv(
        "MMM_COLAB_SETUP_RECEIPT",
        json.dumps({"save_to_google_drive": True, "output_root": str(output_root)}),
    )
    assert autotune._cache_path() == (
        output_root / ".mmm-cache" / "llama-server-autotune.json"
    ).resolve()


def test_managed_server_fast_path_skips_external_health_http(monkeypatch) -> None:
    class _AliveProcess:
        @staticmethod
        def poll():
            return None

    previous_process = autotune._MANAGED_PROCESS
    previous_url = autotune._MANAGED_URL
    try:
        config = object()
        monkeypatch.setattr(autotune, "_MANAGED_PROCESS", _AliveProcess())
        monkeypatch.setattr(autotune, "_MANAGED_URL", "http://127.0.0.1:8910/v1")
        monkeypatch.setattr(
            autotune,
            "_MMM_LLAMA_RUNTIME_RECEIPT",
            {
                "selection_inputs_sha256": runtime_tuning._json_fingerprint(
                    runtime_tuning._selection_inputs(config)
                )
            },
            raising=False,
        )
        monkeypatch.setattr(
            autotune,
            "_external_server_is_ready",
            lambda: (_ for _ in ()).throw(AssertionError("health HTTP must not run")),
        )
        assert autotune.ensure_tuned_server(config, object()) == "http://127.0.0.1:8910/v1"
    finally:
        autotune._MANAGED_PROCESS = previous_process
        autotune._MANAGED_URL = previous_url


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
    assert decision.speedup == 1.3


def test_autotune_keeps_baseline_when_gain_is_below_threshold() -> None:
    decision = _choose_variant(
        (_probe("baseline", tps=20.0), _probe("mtp-1", tps=20.4, width=1)),
        minimum_speedup=1.03,
    )
    assert decision is not None
    assert decision.selected.name == "baseline"


def test_native_server_sizing_requires_explicit_overrides(monkeypatch) -> None:
    from inspect import unwrap

    for key in (
        "MMM_LLAMA_SERVER_CTX",
        "MMM_LLAMA_BATCH",
        "MMM_LLAMA_UBATCH",
        "MMM_KV_CACHE_QUANT",
        "MMM_LLAMA_PARALLEL",
    ):
        monkeypatch.delenv(key, raising=False)
    native_args = unwrap(_base_args)
    config = SimpleNamespace(max_context=32768)
    args = native_args("llama-server", "/tmp/model.gguf", config, 8910)
    for flag in (
        "--ctx-size",
        "--batch-size",
        "--ubatch-size",
        "--cache-type-k",
        "--cache-type-v",
    ):
        assert flag not in args
    assert args[args.index("--parallel") + 1] == "-1"
    assert args[args.index("--gpu-layers") + 1] == "all"
    assert args[args.index("--flash-attn") + 1] == "on"
    assert args[args.index("--load-mode") + 1] == "none"
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "8192")
    monkeypatch.setenv("MMM_LLAMA_UBATCH", "256")
    monkeypatch.setenv("MMM_LLAMA_PARALLEL", "2")
    args = native_args("llama-server", "/tmp/model.gguf", config, 8910)
    assert args[args.index("--ctx-size") + 1] == "8192"
    assert args[args.index("--ubatch-size") + 1] == "256"
    assert args[args.index("--parallel") + 1] == "2"


def test_speculative_server_flags_are_native() -> None:
    from inspect import unwrap

    native_args = unwrap(_variant_args)
    assert native_args(ServerVariant("baseline", "none")) == ["--spec-type", "none"]
    assert native_args(ServerVariant("mtp-2", "draft-mtp", 2)) == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--spec-draft-ngl",
        "all",
    ]
