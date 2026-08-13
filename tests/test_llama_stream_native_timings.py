from __future__ import annotations

from minecraft_mod_ai.llama_stream_efficiency_contract import _native_timing_summary


def test_native_timing_summary_reports_decode_and_mtp_acceptance() -> None:
    summary = _native_timing_summary(
        {
            "predicted_per_second": 82.4,
            "draft_n": 300,
            "draft_n_accepted": 210,
        }
    )

    assert summary is not None
    assert summary["predicted_per_second"] == 82.4
    assert summary["draft_n"] == 300
    assert summary["draft_n_accepted"] == 210
    assert summary["draft_acceptance_pct"] == 70.0


def test_native_timing_summary_handles_missing_speculative_counters() -> None:
    summary = _native_timing_summary({"predicted_per_second": "27.5"})

    assert summary == {"predicted_per_second": 27.5}


def test_native_timing_summary_ignores_invalid_values() -> None:
    assert _native_timing_summary(None) is None
    assert _native_timing_summary({"predicted_per_second": "bad"}) is None


def test_native_timing_summary_clamps_impossible_accepted_count() -> None:
    summary = _native_timing_summary(
        {"predicted_per_second": 50.0, "draft_n": 10, "draft_n_accepted": 15}
    )

    assert summary is not None
    assert summary["draft_n_accepted"] == 10
    assert summary["draft_acceptance_pct"] == 100.0
