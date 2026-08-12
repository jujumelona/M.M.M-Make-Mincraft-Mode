from __future__ import annotations

import json
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping, Sequence

from .platform_catalog import adapter_for_target


_ACTIVE_CODER_TARGET: ContextVar[Any | None] = ContextVar(
    "mmm_custom_coder_platform_target",
    default=None,
)


def install(custom_module_generator_module: Any) -> None:
    """Prevent historical 1.20.1 coder prompt defaults from leaking into live targets."""

    _install_custom_generator_scope(custom_module_generator_module)
    _install_router_rewrite()


def _install_custom_generator_scope(module: Any) -> None:
    cls = module.CustomModuleGenerator
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
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
        mappings: str = "1.20.1+build.1",
    ):
        try:
            adapter = adapter_for_target(minecraft_version, loader)
        except ValueError as exc:
            raise module.__class__.__module__ and type(exc)(str(exc))
        if mappings != adapter.yarn_mappings:
            raise ValueError(
                "Custom coder mappings disagree with the approved platform target: "
                f"{mappings!r} != {adapter.yarn_mappings!r}."
            )
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

    generate._mmm_dynamic_coder_target = True
    cls.generate = generate


def _install_router_rewrite() -> None:
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
        *,
        media_paths=(),
        response_format: str = "text",
    ) -> str:
        adapter = _ACTIVE_CODER_TARGET.get()
        rewritten = messages
        if adapter is not None and role == "coder":
            rewritten = tuple(
                _rewrite_message(message, adapter)
                for message in messages
            )
        return original(
            self,
            role,
            rewritten,
            media_paths=media_paths,
            response_format=response_format,
        )

    generate_text._mmm_dynamic_coder_target = True
    cls.generate_text = generate_text


def _rewrite_message(message: Mapping[str, Any], adapter: Any) -> dict[str, Any]:
    result = dict(message)
    content = result.get("content")
    if not isinstance(content, str):
        return result

    # Structured generation requests are authoritative. Rewrite the historical Java
    # 17 placeholder to the exact approved target before it reaches the coder model.
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("phase") == "generate_patch":
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
        result["content"] = json.dumps(payload, ensure_ascii=False)
        return result

    target_label = (
        f"Minecraft Java {adapter.minecraft_version} {adapter.loader.capitalize()}"
    )
    content = content.replace(
        "Minecraft Java 1.20.1 Fabric",
        target_label,
    )
    content = content.replace(
        "Minecraft 1.20.1 Fabric Java mod",
        f"Minecraft {adapter.minecraft_version} {adapter.loader.capitalize()} Java mod",
    )
    content = content.replace(
        "Minecraft Fabric 1.20.1",
        f"Minecraft {adapter.minecraft_version} {adapter.loader.capitalize()}",
    )
    result["content"] = content
    return result
