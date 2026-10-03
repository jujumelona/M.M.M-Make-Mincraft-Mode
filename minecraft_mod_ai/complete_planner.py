from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from .authored_plan import AuthoredPlan
from .complete_spec import CompleteProposal
from .model_router import ModelRouter
from .planner_trace_artifacts import repository_revision
from .spec import SpecValidationError
from .root_cause_trace import emit_root_cause, trace_scope


class CompleteGameDesignPlanner:
    """Write a design freely; compile executable contracts only for production."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> AuthoredPlan:
        """Write the design itself; no schema, critic, evidence or production gate."""
        from .planner_operation import planner_operation

        if not callable(getattr(self.router, "generate_tool_decision", None)):
            raise SpecValidationError(
                "Typed PlanIR planning requires native structured decisions."
            )

        from .authored_structured_design import (
            author_structured_sections,
            render_structured_sections,
        )
        from .typed_host_capabilities import typed_host_capability_contracts
        from .typed_plan_authoring import author_typed_plan_ir

        with planner_operation("author_structured_execution_contract"):
            structured_sections = author_structured_sections(
                self.router,
                prompt,
                media_paths=media_paths,
            )

        text = render_structured_sections(structured_sections)

        with planner_operation("author_typed_plan_ir"):
            typed_plan_ir = author_typed_plan_ir(
                self.router,
                text,
                structured_sections,
                typed_host_capability_contracts(),
            )
        from .typed_plan_support import assert_typed_plan_host_support

        assert_typed_plan_host_support(
            structured_sections,
            typed_plan_ir,
        )
        return AuthoredPlan(
            requested_prompt=prompt,
            text=text,
            existing_input_sha256=existing_input_sha256,
            media_paths=tuple(str(path) for path in media_paths),
            structured_sections=structured_sections,
            typed_plan_ir=typed_plan_ir,
        )

    def compile_for_production(
        self,
        prompt: AuthoredPlan | str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> CompleteProposal:
        from .authored_production import compile_authored_design

        if not isinstance(prompt, AuthoredPlan):
            raise TypeError(
                "compile_for_production requires an AuthoredPlan with Typed PlanIR; "
                "the legacy raw-text production route has been removed."
            )
        plan = prompt
        with trace_scope("production_preparation", trace_id=uuid.uuid4().hex):
            emit_root_cause(
                "production_preparation_start", stage="production", result="START",
                details={**repository_revision(), "input": "saved_authored_design"},
            )
            return compile_authored_design(
                self.router, plan, existing_input_sha256=existing_input_sha256,
            )

__all__ = ["CompleteGameDesignPlanner"]
