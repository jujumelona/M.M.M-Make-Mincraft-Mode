from __future__ import annotations

import json

from minecraft_mod_ai import agentic_research_game_design as agentic


_PAGE_REF = "sha256:0123456789abcdef#page=1/1"


def _qwen_claim_note() -> dict:
    return {
        "domain_id": "request",
        "claims": [
            {
                "claim_text": "잡몹에서 보스로 이어지는 성장 구조를 설계할 수 있다.",
                "evidence_refs": [_PAGE_REF],
            }
        ],
        "gaps": [],
        "next_queries": [],
        "procedures": [],
        "sufficient": True,
    }


def test_qwen_claim_text_is_canonicalized_before_grounding_validation() -> None:
    # Match the observed llama/Qwen transport: no research_note wrapper and a
    # claim_text field inside claims.
    raw = json.dumps(_qwen_claim_note(), ensure_ascii=False)

    note = agentic._parse_research_note(raw, "request")

    assert note["claims"] == [
        {
            "claim": "잡몹에서 보스로 이어지는 성장 구조를 설계할 수 있다.",
            "evidence_refs": [_PAGE_REF],
        }
    ]
    agentic._validate_sufficient_research(
        note,
        allowed_refs=frozenset({_PAGE_REF}),
    )


def test_qwen_content_and_source_ref_are_canonicalized() -> None:
    raw = json.dumps(
        {
            "research_note": {
                "domain_id": "request",
                "claims": [
                    {
                        "claim_id": "fabric-mod-json-metadata",
                        "content": "Fabric mods declare loader metadata in fabric.mod.json.",
                        "source_ref": _PAGE_REF,
                    }
                ],
                "gaps": [],
                "next_queries": [],
                "procedures": [],
                "sufficient": True,
            }
        }
    )

    note = agentic._parse_research_note(raw, "request")

    assert note["claims"] == [
        {
            "claim": "Fabric mods declare loader metadata in fabric.mod.json.",
            "evidence_refs": [_PAGE_REF],
        }
    ]
    agentic._validate_sufficient_research(
        note,
        allowed_refs=frozenset({_PAGE_REF}),
    )

