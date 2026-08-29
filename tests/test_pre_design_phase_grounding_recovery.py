from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agentic_pre_design_rag as paged_rag
from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai import runtime_stability_contract
from minecraft_mod_ai.pre_design_domain_research import (
    _merge_page_notes,
    _root_page_claims,
)
from minecraft_mod_ai.structured_output import validate_structured_output


def _transcript_page_output(content_chars: int, tail_sha256: str) -> str:
    """Reproduce the Qwen shape from planner_failure (12).log."""

    return json.dumps(
        {
            "research_note": {
                "domain_id": "request",
                "claims": [],
                "gaps": ["This page has no design-relevant API evidence."],
                "next_queries": [],
                "procedures": [],
                "sufficient": False,
            },
            "continuation": {
                "complete": False,
                "next_offset": content_chars,
                "tail_sha256": tail_sha256,
                "reason": "The domain still needs more evidence.",
                "set_sufficient": True,
            },
        }
    )


def test_page_prompt_and_schema_leave_cursor_completion_to_host() -> None:
    messages = paged_rag._research_page_messages(
        prompt="메이플 스토리 모드를 설계해줘",
        domain={"domain_id": "request"},
        document={"document_sha256": "sha256:document"},
        page={
            "schema_version": "page-v1",
            "page_ref": "sha256:document#page=1/1",
            "unit_id": "request",
            "part_index": 0,
            "part_count": 1,
            "content": "A" * 400,
        },
    )
    payload = json.loads(messages[-1]["content"])
    schema = paged_rag._page_response_schema(agentic._RESEARCH_NOTE_SCHEMA)

    assert "continuation_contract" not in payload
    assert "tail_sha256" not in payload["evidence_page"]
    assert "continuation" not in schema["properties"]
    assert "required" not in schema
    assert "host owns" in messages[0]["content"]


@pytest.mark.parametrize("content_chars", [400, 900, 1800])
def test_transcript_false_tail_completion_is_canonicalized_without_retry(
    content_chars: int,
) -> None:
    tail_sha256 = "sha256:" + "a" * 64
    schema = paged_rag._page_response_schema(agentic._RESEARCH_NOTE_SCHEMA)
    validated = validate_structured_output(
        _transcript_page_output(content_chars, tail_sha256),
        response_format="json",
        response_schema=schema,
    )
    envelope = json.loads(validated)

    assert set(envelope) == {"research_note"}
    parsed = paged_rag._parse_page_response(
        agentic,
        validated,
        domain_id="request",
        current_offset=0,
        content_chars=content_chars,
        tail_sha256=tail_sha256,
    )
    assert parsed["continuation"] == {
        "complete": True,
        "next_offset": content_chars,
        "tail_sha256": tail_sha256,
    }


def test_transcript_tail_shape_finishes_in_one_outer_model_call() -> None:
    calls = 0
    content_chars = 1800
    tail_sha256 = "sha256:" + "b" * 64
    raw = _transcript_page_output(content_chars, tail_sha256)

    class Router:
        def generate_text(self, role, messages, **kwargs):
            nonlocal calls
            del role, messages
            calls += 1
            return validate_structured_output(
                raw,
                response_format=str(kwargs["response_format"]),
                response_schema=kwargs["response_schema"],
            )

    class BoundedResearchOutputError(RuntimeError):
        pass

    module = SimpleNamespace(
        _SYNTHESIS_INPUT_BYTES=3600,
        _BoundedResearchOutputError=BoundedResearchOutputError,
        _emit_research_progress=lambda *args, **kwargs: None,
    )
    runtime_stability_contract._install_bounded_research_efficiency(module)
    messages = paged_rag._research_page_messages(
        prompt="메이플 스토리 모드를 설계해줘",
        domain={"domain_id": "request"},
        document={"document_sha256": "sha256:document"},
        page={
            "schema_version": "page-v1",
            "page_ref": "sha256:document#page=1/1",
            "content": "A" * content_chars,
        },
    )

    parsed = module._generate_bounded(
        agentic,
        Router(),
        messages=messages,
        response_schema=paged_rag._page_response_schema(
            agentic._RESEARCH_NOTE_SCHEMA
        ),
        parser=lambda output: paged_rag._parse_page_response(
            agentic,
            output,
            domain_id="request",
            current_offset=0,
            content_chars=content_chars,
            tail_sha256=tail_sha256,
        ),
        progress_label="domain request page 1/1 offset 0",
    )

    assert calls == 1
    assert parsed["continuation"]["complete"] is True
    assert parsed["continuation"]["next_offset"] == content_chars


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

    with pytest.raises(agentic.SpecValidationError, match="host-issued evidence_ref"):
        agentic._validate_sufficient_research(
            {
                "sufficient": True,
                "claims": [{"claim": "uncited", "evidence_refs": []}],
            },
            allowed_refs=allowed,
        )

    with pytest.raises(agentic.SpecValidationError, match="unverified evidence_refs"):
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
