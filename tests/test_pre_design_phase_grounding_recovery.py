from __future__ import annotations

import pytest

from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai.pre_design_domain_research import (
    _merge_page_notes,
    _root_page_claims,
)


def test_page_claims_are_bound_to_host_issued_root_ref() -> None:
    notes = [
        {
            "claims": [
                {
                    "claim": "Target-neutral architecture evidence is usable during pre-design.",
                    "evidence_refs": ["sha256:model-invented-ref"],
                },
                {
                    "claim": "A second grounded statement is present on the page.",
                    "evidence_refs": [],
                },
            ]
        }
    ]

    assert _root_page_claims(notes, page_ref="sha256:host-page#page=1/1") == [
        {
            "claim": "Target-neutral architecture evidence is usable during pre-design.",
            "evidence_refs": ["sha256:host-page#page=1/1"],
        },
        {
            "claim": "A second grounded statement is present on the page.",
            "evidence_refs": ["sha256:host-page#page=1/1"],
        },
    ]


def test_domain_merge_is_sufficient_only_with_grounded_claims() -> None:
    empty = _merge_page_notes(
        "request",
        [("sha256:host-page#page=1/1", [{"claims": [], "gaps": []}])],
    )
    assert empty["claims"] == []
    assert empty["sufficient"] is False
    assert any("No evidence-backed" in gap for gap in empty["gaps"])

    grounded = _merge_page_notes(
        "request",
        [
            (
                "sha256:host-page#page=1/1",
                [
                    {
                        "claims": [
                            {
                                "claim": "Grounded design fact.",
                                "evidence_refs": ["sha256:not-host-issued"],
                            }
                        ],
                        "gaps": [],
                        "next_queries": [],
                        "procedures": [],
                    }
                ],
            )
        ],
    )
    assert grounded["sufficient"] is True
    assert grounded["claims"] == [
        {
            "claim": "Grounded design fact.",
            "evidence_refs": ["sha256:host-page#page=1/1"],
        }
    ]


def test_strict_research_validator_rejects_missing_or_unissued_refs() -> None:
    allowed = frozenset({"sha256:host-page#page=1/1"})

    with pytest.raises(agentic.SpecValidationError, match="evidence_refs"):
        agentic._validate_sufficient_research(
            {
                "sufficient": True,
                "claims": [{"claim": "uncited", "evidence_refs": []}],
            },
            allowed_refs=allowed,
        )

    with pytest.raises(agentic.SpecValidationError, match="outside bounded input"):
        agentic._validate_sufficient_research(
            {
                "sufficient": True,
                "claims": [
                    {
                        "claim": "invented provenance",
                        "evidence_refs": ["sha256:model-invented-ref"],
                    }
                ],
            },
            allowed_refs=allowed,
        )

    agentic._validate_sufficient_research(
        {
            "sufficient": True,
            "claims": [
                {
                    "claim": "valid provenance",
                    "evidence_refs": ["sha256:host-page#page=1/1"],
                }
            ],
        },
        allowed_refs=allowed,
    )
