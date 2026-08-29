from __future__ import annotations

import json

import minecraft_mod_ai.agentic_research_game_design as agentic


class _ToolResearchRouter:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return json.dumps(
            {
                "research_note": {
                    "domain_id": "request",
                    "claims": [
                        {
                            "claim": "Target-neutral architecture can be researched before exact version selection.",
                            "evidence_refs": ["tool:official_docs"],
                        }
                    ],
                    "gaps": [],
                    "next_queries": ["Verify exact mappings after target freeze"],
                    "procedures": [],
                    "sufficient": True,
                }
            }
        )


def test_domain_research_uses_tools_but_unhosted_tool_ref_cannot_make_it_sufficient(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_TRACE_DIR", str(tmp_path))
    router = _ToolResearchRouter()

    note = agentic._research_domain_with_agent(
        router,
        prompt="design a space mod",
        domain={"domain_id": "request", "queries": ["space mod architecture"]},
        deterministic={
            "technology_radar": {
                "status": "deferred_until_target_freeze",
                "target_frozen": False,
            }
        },
        trace_metadata=None,
    )

    assert note["sufficient"] is False
    assert note["fixed_point"] is True
    assert len(router.calls) == 2
    _role, messages, kwargs = router.calls[0]
    assert kwargs["tool_stage"] == "research"
    assert kwargs["enable_tools"] is True
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "intentionally" in rendered
    assert "host-issued evidence_ref" in rendered
