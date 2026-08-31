from __future__ import annotations

from minecraft_mod_ai.pre_design_rag_corrective import (
    _bounded_trace_text,
    _round_has_verified_claims,
)


def test_verified_claim_is_not_invalidated_by_page_local_gap() -> None:
    summary = {
        "claims": [
            {
                "claim": "A source-backed Minecraft mechanic was verified.",
                "evidence_refs": ["page:1"],
                "support_verification": "model_entailment+host_exact_quote",
            }
        ],
        "gaps": [
            "This individual page does not describe the unrelated colonization mechanic."
        ],
    }

    assert _round_has_verified_claims(summary) is True


def test_page_local_gap_cannot_fake_verified_evidence() -> None:
    summary = {
        "claims": [],
        "gaps": ["This page contains no claim-bearing source content."],
    }

    assert _round_has_verified_claims(summary) is False


def test_terminal_gap_trace_keeps_exact_bounded_failure_text() -> None:
    gaps = [
        "Missing evidence for alien combat behavior",
        "Missing evidence for planet colonization behavior",
    ]

    assert _bounded_trace_text(gaps) == gaps


def test_terminal_gap_trace_is_bounded_but_not_silent() -> None:
    huge = "x" * 1000
    rendered = _bounded_trace_text([huge])

    assert len(rendered) == 1
    assert rendered[0].endswith("...")
    assert len(rendered[0]) == 360
