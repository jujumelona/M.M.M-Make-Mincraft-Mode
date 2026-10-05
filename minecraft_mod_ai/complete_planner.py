from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from .platform_backend_contract import deterministic_backend_capabilities
from pathlib import Path
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import CompleteProposal
from .model_router import ModelRouter
from .planner_trace_artifacts import repository_revision
from .root_cause_trace import emit_root_cause, trace_scope


_CONTENT_GRAPH_CONCERNS = (
    "registries",
    "data_resources",
    "assets",
    "interactions",
    "displayed_state",
)


def _content_request_catalog(
    structured_sections: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only concrete resource/UI records into content-design requirements."""

    from .authored_structured_design import active_concern_records

    records = active_concern_records(
        structured_sections,
        "resources_and_ui",
    )
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for concern in _CONTENT_GRAPH_CONCERNS:
        for index, record in enumerate(records.get(concern, ())):
            payload = {
                "concern": concern,
                "record": dict(record),
            }
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
            requirement_id = f"content_{concern}_{digest}"
            if requirement_id in seen:
                continue
            seen.add(requirement_id)
            statement = (
                f"resources_and_ui.{concern}: "
                + "; ".join(
                    f"{key}={value}"
                    for key, value in record.items()
                    if str(value).strip()
                )
            )
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "statement": statement,
                    "source_span": {"text": statement},
                }
            )
    return {"requirements": requirements}


def _serialize_content_design(value: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the canonical content graph without leaking runtime dataclass objects."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "modules":
            result[key] = [
                {
                    "module_id": module.module_id,
                    "kind": module.kind,
                    "config": deepcopy(module.config),
                    "depends_on": list(module.depends_on),
                    "required_gates": list(module.required_gates),
                }
                for module in item
            ]
        elif key == "assets":
            result[key] = [asdict(asset) for asset in item]
        elif key == "_implementation_facts":
            result[key] = [fact.to_dict() for fact in item]
        else:
            result[key] = deepcopy(item)
    return result


def _content_owned_refs(
    structured_sections: Mapping[str, Any],
    content_design: Mapping[str, Any],
) -> frozenset[str]:
    """Return resource/UI concerns whose executable owner is the content artifact graph."""

    if not content_design.get("_implementation_facts"):
        return frozenset()
    from .authored_structured_design import active_concern_records

    records = active_concern_records(
        structured_sections,
        "resources_and_ui",
    )
    owned = {
        f"resources_and_ui.{concern}"
        for concern in _CONTENT_GRAPH_CONCERNS
        if records.get(concern)
    }
    if records.get("paths"):
        owned.add("resources_and_ui.paths")
    return frozenset(owned)


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
        auto_allowed_platform_kinds: frozenset[str] | None = None
        if kinds is None and adapter is not None:
            kinds = deterministic_backend_capabilities(adapter)
        if kinds is None:
            kinds = self.deterministic_module_kinds
        if kinds is None and self.adapter is not None:
            kinds = deterministic_backend_capabilities(self.adapter)
        if kinds is None:
            kinds = getattr(self.router, "_mmm_deterministic_module_kinds", None)
        if kinds is None:
            router_adapter = getattr(self.router, "_mmm_target_adapter", None)
            if router_adapter is not None:
                kinds = deterministic_backend_capabilities(router_adapter)
        if kinds is None:
            version = getattr(self.router, "_mmm_requested_minecraft_version", None)
            loader = getattr(self.router, "_mmm_requested_loader", None)
            if version and loader:
                try:
                    from .platform_catalog import adapter_for_target
                    resolved_adapter = adapter_for_target(str(version), str(loader))
                    kinds = deterministic_backend_capabilities(resolved_adapter)
                except Exception:
                    pass
        if kinds is None:
            existing_version = getattr(
                self.router,
                "_mmm_existing_minecraft_version",
                None,
            )
            existing_loader = getattr(
                self.router,
                "_mmm_existing_loader",
                None,
            )
            if existing_version and existing_loader:
                try:
                    from .platform_catalog import adapter_for_target
                    existing_adapter = adapter_for_target(
                        str(existing_version),
                        str(existing_loader),
                    )
                    kinds = deterministic_backend_capabilities(existing_adapter)
                except Exception:
                    pass
        if kinds is None:
            # AUTO planning filters semantic kinds by *per-target* executability.
            # Never union primitive capabilities from different targets: doing so can
            # synthesize support that no single immutable target receipt actually has.
            try:
                from .platform_backend_contract import production_backend_is_supported
                from .platform_catalog import (
                    adapter_for_target,
                    discover_target_keys,
                )
                from .typed_platform_ir import PLATFORM_HOST_KINDS, PLATFORM_KINDS

                loader_hint = getattr(
                    self.router,
                    "_mmm_requested_loader",
                    None,
                )
                version_hint = getattr(
                    self.router,
                    "_mmm_requested_minecraft_version",
                    None,
                )
                target_keys = discover_target_keys(
                    loader=str(loader_hint) if loader_hint else None,
                    limit_per_loader=32,
                )
                if version_hint:
                    target_keys = tuple(
                        (target_loader, target_version)
                        for target_loader, target_version in target_keys
                        if str(target_version) == str(version_hint)
                    )

                target_capability_sets: list[frozenset[str]] = []
                for target_loader, target_version in target_keys:
                    try:
                        target_capability_sets.append(
                            deterministic_backend_capabilities(
                                adapter_for_target(
                                    str(target_version),
                                    str(target_loader),
                                )
                            )
                        )
                    except Exception:
                        continue

                executable_kinds = set(PLATFORM_HOST_KINDS)
                executable_kinds.update(
                    kind
                    for kind in PLATFORM_KINDS
                    if kind not in PLATFORM_HOST_KINDS
                    and any(
                        production_backend_is_supported(capabilities, kind)
                        for capabilities in target_capability_sets
                    )
                )
                auto_allowed_platform_kinds = frozenset(executable_kinds)
            except Exception:
                auto_allowed_platform_kinds = frozenset()
        effective_kinds = tuple(sorted(kinds)) if kinds is not None else None

        with planner_operation("author_typed_plan_ir"):
            typed_plan_ir = author_typed_plan_ir(
                self.router,
                text,
                structured_sections,
                typed_host_capability_contracts(),
                deterministic_module_kinds=effective_kinds,
                allowed_platform_kinds=auto_allowed_platform_kinds,
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
