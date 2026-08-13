from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.qwen35_t4_single_stream_tuning import (
    _is_t4_runtime,
    _select,
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
    monkeypatch.setenv("MMM_QWEN35_T4_WIDTHS", "4,3,3,0,9,bad,2")
    assert _widths() == (4, 3, 2)


def test_single_stream_selection_requires_identical_output(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_T4_MIN_GAIN", "1.01")
    baseline = _probe("baseline", 25.0, "same")
    valid = _probe("mtp-3", 31.0, "same", width=3)
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
    baseline = _probe("baseline", 30.0, "same")
    noisy = _probe("mtp-3", 30.2, "same", width=3)

    selected, _, selected_tps = _select(baseline, [baseline, noisy])

    assert selected is baseline.variant
    assert selected_tps == 30.0
