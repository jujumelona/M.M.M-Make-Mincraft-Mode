from __future__ import annotations

from typing import Any

from minecraft_mod_ai.research_derived_requirements import FACETS, _model_facets


class _ToolDecisionRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_tool_decision(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        tool_name: str,
        parameters: dict[str, Any],
        description: str = "",
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        return {
            # Deliberately try to smuggle an invalid facet name. The host must ignore it.
            "facet": "platform_loader_constraint",
            "disposition": "not_applicable",
            "statement": "",
            "rationale": "No implementation obligation is established for this facet.",
            "evidence_refs": [],
            "acceptance": [],
            "implementation_obligations": [],
        }


def _large_evidence_catalog(count: int = 100) -> list[dict[str, Any]]:
    return [
        {
            "evidence_ref": f"evidence:{index:016x}",
            "path": f"technical_evidence.items[{index}]",
            "summary": {
                "claim": (
                    "server network packet registry lifecycle persistence test integration "
                    + ("x" * 1200)
                ),
                "loader": "fabric",
                "version": "1.0",
            },
        }
        for index in range(count)
    ]


def test_research_facets_are_host_owned_native_tool_decisions() -> None:
    router = _ToolDecisionRouter()
    rows = _model_facets(
        router,
        requirement={
            "requirement_id": "req_spacecraft_upgrade",
            "description": "Upgrade spacecraft performance through trade and purchases.",
        },
        evidence=_large_evidence_catalog(),
    )

    assert [row["facet"] for row in rows] == list(FACETS)
    assert len(router.calls) == len(FACETS)

    for call, expected_facet in zip(router.calls, FACETS, strict=True):
        assert call["role"] == "planner"
        assert call["tool_name"] == "record_research_facet_decision"
        assert "facet" not in call["parameters"]["properties"]

        combined_prompt = "\n".join(message["content"] for message in call["messages"])
        assert "Return JSON only" not in combined_prompt
        assert 'response_format="json"' not in combined_prompt
        assert f"Fixed facet: {expected_facet}" in combined_prompt

        user_prompt = call["messages"][-1]["content"]
        assert user_prompt.count("- evidence_ref:") <= 12
        assert len(user_prompt) < 12_000


def test_research_facet_path_does_not_require_generate_text() -> None:
    router = _ToolDecisionRouter()
    assert not hasattr(router, "generate_text")

    rows = _model_facets(
        router,
        requirement={"requirement_id": "req_one", "description": "One bounded requirement."},
        evidence=_large_evidence_catalog(3),
    )

    assert len(rows) == len(FACETS)
