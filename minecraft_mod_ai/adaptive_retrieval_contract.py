from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

_GROUNDING_MARKER = "__mmm_repository_grounding_v2_live_context__"


def install(model_router_module: Any) -> None:
    """Install repository grounding and adaptive execution hardening."""

    if not bool(
        getattr(
            model_router_module.ModelRouter._generate_with_tools,
            "_mmm_progress_aware_tool_loop_owner",
            False,
        )
    ):
        raise RuntimeError(
            "ModelRouter must directly own the progress-aware production tool loop."
        )

    from .adaptive_execution_hardening import harden_adaptive_execution
    from .agent_tool_allowlist_hardening import harden_agent_tool_allowlist
    from .hybrid_route_hardening import harden_code_search_routes
    from .inference_time_scaling import harden_runtime
    from .runtime_composer_hardening import harden_runtime_composer_identity
    from .temporary_skill_compatibility import harden_temporary_skill_transport

    _install_repository_grounding()
    harden_runtime()
    harden_adaptive_execution()
    harden_code_search_routes()
    harden_runtime_composer_identity()
    harden_agent_tool_allowlist()
    harden_temporary_skill_transport()
    _expose_composed_repair_contracts()


def _inherit_boolean_contract_markers(current: Any) -> None:
    wrapped = getattr(current, "__wrapped__", None)
    while callable(wrapped):
        for name, value in vars(wrapped).items():
            if name.startswith("_mmm_") and value is True:
                setattr(current, name, True)
        wrapped = getattr(wrapped, "__wrapped__", None)


def _expose_composed_repair_contracts() -> None:
    from .repair_engine import RepairEngine

    _inherit_boolean_contract_markers(RepairEngine._signature)


def _runtime_grounding_budget(router: Any, requested: int, *, role: str) -> int:
    """Reserve most of the live model window for task/tool traffic, not source dumps."""

    requested = max(1024, int(requested))
    registry = getattr(router, "registry", None)
    resolve = getattr(registry, "role", None)
    profile = str(getattr(router, "profile", "") or "").strip()
    if not callable(resolve) or not profile:
        return min(requested, 32 * 1024)
    try:
        config = resolve(profile, role)
        from .model_context_budget import request_message_budget

        live_request_bytes = request_message_budget(config, ())
    except Exception:
        return min(requested, 32 * 1024)
    return min(requested, max(4 * 1024, live_request_bytes // 2))


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
                    byte_budget=_runtime_grounding_budget(
                        router,
                        byte_budget,
                        role="coder",
                    ),
                    diagnostic_paths=diagnostic_paths,
                )
            except ValueError as exc:
                raise custom_module_generator.CustomModuleGenerationError(str(exc)) from exc

        setattr(_collect_initial_observations, _GROUNDING_MARKER, True)
        setattr(_collect_initial_observations, "__mmm_repository_grounding_v1__", True)
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
            byte_budget=_runtime_grounding_budget(
                self.router,
                self.policy.model_context_bytes,
                role="coder_safe",
            ),
        )

    setattr(_context, _GROUNDING_MARKER, True)
    setattr(_context, "__mmm_repository_grounding_v1__", True)
    repair_engine.RepairEngine._context = _context
