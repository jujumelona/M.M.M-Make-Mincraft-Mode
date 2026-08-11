from __future__ import annotations

import minecraft_mod_ai.atomic_requirement_contract as atomic_requirement_contract
from minecraft_mod_ai.semantic_reviewer_role_contract import _SafeReviewerRouter


def test_coverage_reviewer_uses_coder_safe_role() -> None:
    calls: list[str] = []

    class Router:
        def generate_text(self, role: str, *_args, **_kwargs) -> str:
            calls.append(role)
            return "ok"

    assert _SafeReviewerRouter(Router()).generate_text("planner", []) == "ok"
    assert calls == ["coder_safe"]
    assert getattr(
        atomic_requirement_contract.semantic_review,
        "_mmm_independent_reviewer_role",
        False,
    )
