from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALL_MARKER = "__mmm_progress_aware_retrieval_v1__"
_GROUNDING_MARKER = "__mmm_repository_grounding_v1__"


def install(model_router_module: Any) -> None:
    """Install progress-aware retrieval and task-adaptive repository grounding.

    The router module remains the canonical owner of ModelRouter and all of its
    globals. This contract replaces only the retrieve/act/observe method and the
    two host-owned repository-context selectors used by generation and repair.
    Existing runtime wrappers and test monkeypatches keep their normal module
    semantics.
    """
    _install_router_loop(model_router_module)
    _install_repository_grounding()


def _install_router_loop(model_router_module: Any) -> None:
    router_cls = model_router_module.ModelRouter
    current = router_cls._generate_with_tools
    if getattr(current, _INSTALL_MARKER, False):
        return

    @wraps(current)
    def _generate_with_tools(
        self: Any,
        *,
        adapter: Any,
        request: Any,
        runtime: Any,
        stage: str,
        role: str,
    ) -> str:
        from .progress_aware_tool_loop import generate_with_tools

        return generate_with_tools(
            self,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage=stage,
            role=role,
        )

    setattr(_generate_with_tools, _INSTALL_MARKER, True)
    router_cls._generate_with_tools = _generate_with_tools


def _install_repository_grounding() -> None:
    from . import custom_module_generator, repair_engine
    from .repository_grounding import (
        build_repair_repository_context,
        build_repository_observation_ledger,
    )

    current_collect = custom_module_generator._collect_initial_observations
    if not getattr(current_collect, _GROUNDING_MARKER, False):

        @wraps(current_collect)
        def _collect_initial_observations(
            router: Any,
            index: Any,
            *,
            query: str,
            byte_budget: int,
            diagnostic_paths=(),
        ) -> dict[str, Any]:
            try:
                return build_repository_observation_ledger(
                    router,
                    index,
                    query=query,
                    byte_budget=byte_budget,
                    diagnostic_paths=diagnostic_paths,
                )
            except ValueError as exc:
                raise custom_module_generator.CustomModuleGenerationError(str(exc)) from exc

        setattr(_collect_initial_observations, _GROUNDING_MARKER, True)
        custom_module_generator._collect_initial_observations = _collect_initial_observations

    current_context = repair_engine.RepairEngine._context
    if getattr(current_context, _GROUNDING_MARKER, False):
        return

    @wraps(current_context)
    def _context(self: Any, root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
        diagnostic_paths: list[str] = []
        query_parts: list[str] = []
        for item in evidence.get("diagnostics", {}).get("diagnostics", []):
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("uri")
            if isinstance(path, str) and path.strip():
                diagnostic_paths.append(path)
            message = item.get("message")
            if isinstance(message, str) and message.strip():
                query_parts.append(message)
        build = evidence.get("build", {})
        if isinstance(build.get("error"), str) and build["error"].strip():
            query_parts.append(build["error"])
        for command in build.get("commands", []):
            if not isinstance(command, dict) or not isinstance(command.get("log_path"), str):
                continue
            log = Path(command["log_path"])
            if log.is_file() and not log.is_symlink():
                query_parts.append(log.read_text(encoding="utf-8", errors="replace")[-32_000:])

        query = "\n".join(query_parts).strip()
        if not query:
            query = json.dumps(
                {
                    "diagnostics_status": evidence.get("diagnostics", {}).get("status"),
                    "build_status": build.get("status"),
                },
                sort_keys=True,
            )
        index = repair_engine.active_repair_project_index(root, self.policy)
        return build_repair_repository_context(
            self.router,
            index,
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=self.policy.model_context_bytes,
        )

    setattr(_context, _GROUNDING_MARKER, True)
    repair_engine.RepairEngine._context = _context
