from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable


def _call_supported(callable_obj: Callable[..., Any], /, *args: Any, **kwargs: Any):
    """Forward only kwargs the concrete callable actually declares.

    Target-aware MCP wrappers add optional platform metadata.  Third-party clients,
    old project-local adapters and test doubles may implement the older callable
    surface.  Signature filtering preserves compatibility without catching a
    ``TypeError`` raised *inside* the callee, which would hide a real bug.
    """

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return callable_obj(*args, **kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(*args, **accepted)


def install(*, mcp_tools_module: Any, platform_contract_module: Any) -> None:
    cls = mcp_tools_module.MMMToolService

    current_revise = cls.revise_complete_plan
    if not getattr(current_revise, "_mmm_signature_compatible_target_forwarding", False):

        @wraps(current_revise)
        def revise_complete_plan(
            self: Any,
            original_prompt: str,
            revision: str,
            media_paths=(),
            existing_input_sha256: str = "",
            minecraft_version: str = "",
            loader: str = "fabric",
            existing_minecraft_version: str = "",
            existing_loader: str = "",
        ) -> dict[str, Any]:
            try:
                merged = mcp_tools_module.merge_design_brief(original_prompt, revision)
            except ValueError as exc:
                raise mcp_tools_module.SpecValidationError(
                    "revision must not be empty."
                ) from exc
            active = getattr(self, "_mmm_last_platform_adapter", None)
            return _call_supported(
                self.plan_complete_game,
                merged,
                media_paths=media_paths,
                existing_input_sha256=existing_input_sha256,
                minecraft_version=(
                    minecraft_version
                    or (active.minecraft_version if active is not None else "")
                ),
                loader=(loader or (active.loader if active is not None else "fabric")),
                existing_minecraft_version=existing_minecraft_version,
                existing_loader=existing_loader,
            )

        revise_complete_plan._mmm_platform_bound = True
        revise_complete_plan._mmm_signature_compatible_target_forwarding = True
        cls.revise_complete_plan = revise_complete_plan

    current_discover = cls.discover_ecosystem_resources
    if not getattr(current_discover, "_mmm_signature_compatible_target_forwarding", False):

        @wraps(current_discover)
        def discover_ecosystem_resources(
            self: Any,
            provider: str,
            query: str,
            cursor: str = "",
            limit: int = 20,
            target_profile: str = "minecraft_mod",
            minecraft_version: str = "",
            loader: str = "fabric",
        ) -> dict[str, Any]:
            adapter = platform_contract_module._service_adapter(
                self,
                mcp_tools_module,
                minecraft_version,
                loader,
            )
            client = self.discovery_client_factory()
            return _call_supported(
                client.search,
                provider,
                query,
                cursor=cursor,
                limit=limit,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                target_profile=target_profile,
            )

        discover_ecosystem_resources._mmm_platform_bound = True
        discover_ecosystem_resources._mmm_signature_compatible_target_forwarding = True
        cls.discover_ecosystem_resources = discover_ecosystem_resources

    current_inspect = cls.inspect_modrinth_project
    if not getattr(current_inspect, "_mmm_signature_compatible_target_forwarding", False):

        @wraps(current_inspect)
        def inspect_modrinth_project(
            self: Any,
            project_id: str,
            minecraft_version: str = "",
            loader: str = "fabric",
        ) -> dict[str, Any]:
            adapter = platform_contract_module._service_adapter(
                self,
                mcp_tools_module,
                minecraft_version,
                loader,
            )
            client = self.discovery_client_factory()
            return _call_supported(
                client.inspect_modrinth_project,
                project_id,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
            )

        inspect_modrinth_project._mmm_platform_bound = True
        inspect_modrinth_project._mmm_signature_compatible_target_forwarding = True
        cls.inspect_modrinth_project = inspect_modrinth_project

    # Release packaging is the last authority boundary for imported-source repair.
    # Keep this installation adjacent to the target-aware MCP surface so every
    # package_release caller, including CompleteProductionOrchestrator, receives it.
    from .platform_release_contract import install as install_platform_release

    install_platform_release(mcp_tools_module)
