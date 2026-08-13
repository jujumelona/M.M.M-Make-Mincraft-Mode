from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.qwen35_t4_single_stream_tuning import (
    _EXPECTED_DIGEST,
    _EXPECTED_OBJECT,
    _bucket_for_request,
    _context_buckets,
    _is_t4_runtime,
    _kv_candidates,
    _kv_mode,
    _p_min_candidates,
    _select,
    _semantic_digest,
    _ubatch_candidates,
    _widths,
)


def _probe(name: str, tps: float, digest: str, *, width: int = 0):
    return SimpleNamespace(
        variant=SimpleNamespace(
            name=name,
            spec_type="draft-mtp" if width else "none",
            draft_n_max=width,
            draft_p_min=0.0,
        ),
        ok=True,
        output_sha256=digest,
        predicted_tps=tps,
    )


def test_t4_detection_uses_native_hardware_identity() -> None:
    assert _is_t4_runtime(
        SimpleNamespace(_hardware_identity=lambda: "Tesla T4, 15360 MiB, 570.0")
    )
    assert not _is_t4_runtime(
        SimpleNamespace(_hardware_identity=lambda: "NVIDIA A100-SXM4-40GB")
    )


def test_width_candidates_are_bounded_and_deduplicated(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_WIDTHS", "16,12,8,4,3,3,0,33,bad,2")
    assert _widths() == (16, 12, 8, 4, 3, 2)


def test_p_min_candidates_are_bounded_deduplicated_and_keep_zero(monkeypatch) -> None:
    monkeypatch.setenv(
        "MMM_QWEN35_T4_P_MIN_CANDIDATES",
        "0.9,0.8,0.8,1.0,-0.1,bad,0.7,0.6,0.5",
    )
    assert _p_min_candidates() == (0.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def test_kv_candidates_always_keep_native_reference(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_KV_CANDIDATES", "q8_0,q4_0,q8_0,bad")
    assert _kv_candidates() == ("native-default", "q8_0", "q4_0")


def test_semantic_digest_accepts_formatting_but_rejects_wrong_payload() -> None:
    spaced = '{ "checksum": "mmm-qwen35-t4-single-stream-v2", "values": [' + (
        ", ".join(str(value) for value in range(256))
    ) + "] }"
    assert _semantic_digest(spaced) == _EXPECTED_DIGEST

    wrong = dict(_EXPECTED_OBJECT)
    wrong["values"] = list(range(255))
    import json

    assert _semantic_digest(json.dumps(wrong)) == ""


def test_single_stream_selection_requires_same_valid_semantics(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_MIN_GAIN", "1.01")
    baseline = _probe("baseline", 25.0, _EXPECTED_DIGEST)
    valid = _probe("mtp-3", 31.0, _EXPECTED_DIGEST, width=3)
    divergent = _probe("mtp-4", 50.0, "different", width=4)

    selected, baseline_tps, selected_tps = _select(
        baseline,
        [baseline, valid, divergent],
    )

    assert selected is valid.variant
    assert baseline_tps == 25.0
    assert selected_tps == 31.0


def test_single_stream_selection_ignores_noise_below_minimum_gain(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_MIN_GAIN", "1.01")
    baseline = _probe("baseline", 30.0, _EXPECTED_DIGEST)
    noisy = _probe("mtp-3", 30.2, _EXPECTED_DIGEST, width=3)

    selected, _, selected_tps = _select(baseline, [baseline, noisy])

    assert selected is baseline.variant
    assert selected_tps == 30.0



def test_maximum_defaults_cover_all_t4_search_axes(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_WIDTHS", raising=False)
    monkeypatch.delenv("MMM_QWEN35_T4_P_MIN_CANDIDATES", raising=False)
    monkeypatch.delenv("MMM_QWEN35_T4_UBATCH_CANDIDATES", raising=False)
    monkeypatch.delenv("MMM_QWEN35_T4_KV_CANDIDATES", raising=False)
    monkeypatch.delenv("MMM_QWEN35_T4_KV_CONTEXT_BUCKETS", raising=False)
    assert _widths() == (1, 2, 3, 4, 6, 8, 12, 16)
    assert _p_min_candidates() == (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
    autotune = SimpleNamespace(_env_int=lambda _name, default, **_kwargs: default)
    assert _ubatch_candidates(autotune) == (512, 1024, 2048)
    assert _kv_candidates() == ("native-default", "f16", "q8_0", "q4_0")
    config = SimpleNamespace(max_context=32768, extra={})
    assert _context_buckets(config) == (2048, 8192, 16384, 28672)


def test_kv_mode_supports_auto_and_manual(monkeypatch) -> None:
    config = SimpleNamespace(extra={})
    monkeypatch.setenv("MMM_QWEN35_T4_KV_MODE", "auto")
    assert _kv_mode(config) == "auto"
    monkeypatch.setenv("MMM_QWEN35_T4_KV_MODE", "q8")
    assert _kv_mode(config) == "q8_0"


def test_context_bucket_tracks_request_size(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_KV_CONTEXT_BUCKETS", raising=False)
    config = SimpleNamespace(max_context=32768, extra={})
    short = SimpleNamespace(messages=({"role": "user", "content": "x" * 900},))
    long = SimpleNamespace(messages=({"role": "user", "content": "x" * 30000},))
    assert _bucket_for_request(config, short) == 2048
    assert _bucket_for_request(config, long) == 16384
