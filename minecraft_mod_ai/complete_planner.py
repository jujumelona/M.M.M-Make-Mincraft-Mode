from __future__ import annotations

import uuid
from collections.abc import Sequence
from .platform_backend_contract import effective_target_backend_capabilities
from pathlib import Path
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import CompleteProposal
from .model_router import ModelRouter
from .planner_trace_artifacts import repository_revision
from .root_cause_trace import emit_root_cause, trace_scope


class CompleteGameDesignPlanner:
    """Write a design freely; compile executable contracts only for production."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        adapter: Any = None,
        deterministic_module_kinds: Sequence[str] | frozenset[str] | None = None,
    ) -> None:
        self.router = router
        self.adapter = adapter
        self.deterministic_module_kinds = (
            frozenset(str(k).strip() for k in deterministic_module_kinds if str(k).strip())
            if deterministic_module_kinds is not None
            else None
        )

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
        adapter: Any = None,
        deterministic_module_kinds: Sequence[str] | frozenset[str] | None = None,
    ) -> AuthoredPlan:
        """Write the design itself; no schema, critic, evidence or production gate."""
        from .planner_operation import planner_operation

        from .authored_structured_design import (
            author_structured_sections,
            render_structured_sections,
        )
        from .typed_host_capabilities import typed_host_capability_contracts
        from .typed_plan_authoring import author_typed_plan_ir

        from .planner_budget import PlannerBudget

        budget = PlannerBudget()
        with planner_operation("author_structured_execution_contract"):
            structured_sections = author_structured_sections(
                self.router,
                prompt,
                media_paths=media_paths,
                budget=budget,
            )

        text = render_structured_sections(structured_sections)

        kinds = deterministic_module_kinds
        if kinds is None and adapter is not None:
            kinds = effective_target_backend_capabilities(adapter)
        if kinds is None:
            kinds = self.deterministic_module_kinds
        if kinds is None and self.adapter is not None:
            kinds = effective_target_backend_capabilities(self.adapter)
        if kinds is None:
            kinds = getattr(self.router, "_mmm_deterministic_module_kinds", None)
        if kinds is None:
            router_adapter = getattr(self.router, "_mmm_target_adapter", None)
            if router_adapter is not None:
                kinds = effective_target_backend_capabilities(router_adapter)
        if kinds is None:
            version = getattr(self.router, "_mmm_requested_minecraft_version", None)
            loader = getattr(self.router, "_mmm_requested_loader", None)
            if version and loader:
                try:
                    from .platform_catalog import adapter_for_target
                    resolved_adapter = adapter_for_target(str(version), str(loader))
                    kinds = effective_target_backend_capabilities(resolved_adapter)
                except Exception:
                    pass
        effective_kinds = tuple(sorted(kinds)) if kinds is not None else None

        with planner_operation("author_typed_plan_ir"):
            typed_plan_ir = author_typed_plan_ir(
                self.router,
                text,
                structured_sections,
                typed_host_capability_contracts(),
                deterministic_module_kinds=effective_kinds,
                budget=budget,
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
        prompt: AuthoredPlan,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> CompleteProposal:
        from .authored_production import compile_authored_design

        if not isinstance(prompt, AuthoredPlan):
            raise TypeError("compile_for_production requires AuthoredPlan")
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
