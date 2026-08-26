from __future__ import annotations

from minecraft_mod_ai import atomic_requirement_contract
from minecraft_mod_ai.semantic_reviewer_role_contract import _SafeReviewerRouter


def test_coverage_reviewer_uses_coder_safe_native_tool_role() -> None:
    calls: list[tuple[str, str]] = []
    expected = {
        "supported": False,
        "implementation_indexes": [],
        "acceptance_indexes": [],
    }

    class Router:
        def generate_tool_decision(
            self,
            role: str,
            _messages,
            *,
            tool_name: str,
            **_kwargs,
        ) -> dict[str, object]:
            calls.append((role, tool_name))
            return dict(expected)

    result = _SafeReviewerRouter(Router()).generate_tool_decision(
        "planner",
        [],
        tool_name="submit_atomic_coverage",
        parameters={"type": "object", "properties": {}},
    )

    assert result == expected
    assert calls == [("coder_safe", "submit_atomic_coverage")]
    assert getattr(
        atomic_requirement_contract.semantic_review,
        "_mmm_independent_reviewer_role",
        False,
    )
