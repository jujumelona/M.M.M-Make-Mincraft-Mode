from __future__ import annotations

import json
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping, Sequence

from .dependency_decode_monitor import activate_dependency_decode_monitor
from .platform_catalog import adapter_for_target


_ACTIVE_CODER_TARGET: ContextVar[Any | None] = ContextVar(
    "mmm_custom_coder_platform_target",
    default=None,
)


def _model_router_owner(router: Any) -> Any | None:
    """Resolve the concrete ModelRouter through research/search proxy layers."""

    current = router
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            hasattr(current, "_generation_lock")
            and hasattr(current, "_agent_workspace_root")
            and hasattr(current, "_agent_tool_runtime")
            and hasattr(current, "_agent_require_fresh_evidence")
        ):
            return current
        current = getattr(current, "_router", None)
    return None


def _capture_agent_binding(router: Any) -> tuple[Any, Any, Any, bool] | None:
    """Capture mutable agent binding state before one custom-coder transaction."""

    owner = _model_router_owner(router)
    if owner is None:
        return None
    with owner._generation_lock:
        return (
            owner,
            owner._agent_workspace_root,
            owner._agent_tool_runtime,
            bool(owner._agent_require_fresh_evidence),
        )


def _restore_agent_binding(snapshot: tuple[Any, Any, Any, bool] | None) -> None:
    """Prevent custom-coder workspace/evidence policy from leaking to later calls."""

    if snapshot is None:
        return
    owner, workspace_root, runtime, require_fresh_evidence = snapshot
    with owner._generation_lock:
        owner._agent_workspace_root = workspace_root
        owner._agent_tool_runtime = runtime
        owner._agent_require_fresh_evidence = require_fresh_evidence


def install(custom_module_generator_module: Any) -> None:
    """Bind the custom coder to one executable host-selected platform target."""

    activate_dependency_decode_monitor()
    _install_custom_generator_scope(custom_module_generator_module)
    _install_gradle_metadata_scope(custom_module_generator_module)
    _install_router_target_binding()


def _required_target(
    module_api: Any,
    minecraft_version: str | None,
    loader: str | None,
    mappings: str | None,
) -> tuple[str, str, str]:
    version = str(minecraft_version or "").strip()
    loader_id = str(loader or "").strip().casefold()
    mapping_id = str(mappings or "").strip()
    if not version or not loader_id or not mapping_id:
        raise module_api.CustomModuleGenerationError(
            "Custom coder requires a host-selected minecraft_version, loader and mappings."
        )
    return version, loader_id, mapping_id


def _install_custom_generator_scope(module_api: Any) -> None:
    cls = module_api.CustomModuleGenerator
    original = cls.generate
    if getattr(original, "_mmm_dynamic_coder_target", False):
        return

    @wraps(original)
    def generate(
        self: Any,
        project_root: Any,
        *,
        module: Any,
        research_modules=(),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
    ):
        version, loader_id, mapping_id = _required_target(
            module_api,
            minecraft_version,
            loader,
            mappings,
        )
        try:
            adapter = adapter_for_target(version, loader_id)
        except ValueError as exc:
            raise module_api.CustomModuleGenerationError(
                "Custom coder target could not be resolved: " + str(exc)
            ) from exc
        if mapping_id != adapter.yarn_mappings:
            raise module_api.CustomModuleGenerationError(
                "Custom coder mappings disagree with the approved platform target: "
                f"{mapping_id!r} != {adapter.yarn_mappings!r}."
            )
        agent_binding = _capture_agent_binding(self.router)
        token = _ACTIVE_CODER_TARGET.set(adapter)
        try:
            return original(
                self,
                project_root,
                module=module,
                research_modules=research_modules,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
            )
        finally:
            _ACTIVE_CODER_TARGET.reset(token)
            _restore_agent_binding(agent_binding)

    generate._mmm_dynamic_coder_target = True
    generate._mmm_scoped_agent_workspace = True
    cls.generate = generate


def _install_gradle_metadata_scope(module_api: Any) -> None:
    """Permit only project-owned source/resources and target-safe Gradle metadata."""

    cls = module_api.CustomModuleGenerator
    current = cls._validate_operations
    if getattr(current, "_mmm_live_gradle_metadata_scope", False):
        return

    def validate_operations(
        self: Any,
        operations: list[dict[str, Any]],
    ) -> None:
        gradle_metadata = {
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            "gradle.properties",
            "gradle/libs.versions.toml",
        }
        for item in operations:
            if not isinstance(item, dict):
                raise module_api.CustomModuleGenerationError(
                    "Patch operation must be an object."
                )
            if item.get("operation") not in {"create", "replace", "edit"}:
                raise module_api.CustomModuleGenerationError(
                    "Custom module may not delete files."
                )
            path = module_api._normalized_operation_path(item)
            protected_path = path.casefold()
            if any(
                protected_path == root
                or protected_path.startswith(root + "/")
                for root in (
                    ".minecraft_ai/research",
                    ".minecraft_ai/context-observations",
                )
            ):
                raise module_api.CustomModuleGenerationError(
                    "Model patches may not modify the code-owned research ledger "
                    "or context-observation ledger."
                )
            allowed = (
                path.startswith("src/main/java/")
                or path.startswith("src/main/resources/")
                or path.startswith("src/test/java/")
                or path.startswith("src/gametest/")
                or path.startswith(".minecraft_ai/")
                or path in gradle_metadata
            )
            if not allowed:
                raise module_api.CustomModuleGenerationError(
                    f"Custom module path is outside the allowed scope: {path}"
                )

    validate_operations._mmm_live_gradle_metadata_scope = True
    cls._validate_operations = validate_operations


def _install_router_target_binding() -> None:
    """Bind structured coder requests to the active target; never rewrite prose defaults."""

    from . import model_router as router_module

    cls = router_module.ModelRouter
    original = cls.generate_text
    if getattr(original, "_mmm_dynamic_coder_target", False):
        return

    @wraps(original)
    def generate_text(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        adapter = _ACTIVE_CODER_TARGET.get()
        rewritten = messages
        if adapter is not None and role == "coder":
            rewritten = tuple(_bind_target(message, adapter) for message in messages)
        return original(self, role, rewritten, **kwargs)

    generate_text._mmm_dynamic_coder_target = True
    cls.generate_text = generate_text


def _bind_target(message: Mapping[str, Any], adapter: Any) -> dict[str, Any]:
    result = dict(message)
    content = result.get("content")
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        return result
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict) or payload.get("phase") != "generate_patch":
        return result
    payload["target"] = {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "java": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
    }
    result["content"] = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return result


__all__ = [
    "_capture_agent_binding",
    "_model_router_owner",
    "_restore_agent_binding",
    "install",
]
