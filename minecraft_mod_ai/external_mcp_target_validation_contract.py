from __future__ import annotations

from typing import Any, Mapping


_AUTHORITATIVE_STATUS_FIELDS = {
    "server_get_status": ("minecraftVersion",),
    "get_minecraft_status": ("minecraftVersion", "minecraft_version", "version"),
}


def install(router_module: Any) -> None:
    """Avoid mistaking referenced historical versions for the provider target.

    Search, documentation and version-diff tools may legitimately return many
    Minecraft versions. Only routes that explicitly declare target fields, or the
    reviewed runtime status tools below, are allowed to prove the running target.
    """

    cls = router_module.ExternalMCPRouter
    current = cls._validate_reported_target
    if getattr(current, "_mmm_route_explicit_target_check", False):
        return

    def validate_reported_target(
        result: Mapping[str, Any],
        route: Mapping[str, Any],
        target: Any,
    ) -> None:
        fields = route.get("response_target_fields", [])
        if not fields:
            fields = _AUTHORITATIVE_STATUS_FIELDS.get(str(route.get("tool", "")), ())
        if not fields or not getattr(target, "minecraft_version", ""):
            return
        reported = _collect_named_fields(result, tuple(str(value) for value in fields))
        conflicts = sorted(
            value
            for value in reported
            if value and value != target.minecraft_version
        )
        if conflicts:
            raise router_module.ExternalMCPError(
                "External MCP authoritative status conflicts with the approved "
                f"PlatformLock: expected {target.minecraft_version!r}, got {conflicts!r}."
            )
        if not reported:
            raise router_module.ExternalMCPError(
                "External MCP authoritative status did not report the configured "
                "Minecraft target field."
            )

    validate_reported_target._mmm_route_explicit_target_check = True
    cls._validate_reported_target = staticmethod(validate_reported_target)


def _collect_named_fields(value: Any, fields: tuple[str, ...]) -> set[str]:
    allowed = set(fields)
    found: set[str] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                if str(key) in allowed and isinstance(child, (str, int, float)):
                    found.add(str(child).strip())
                elif isinstance(child, (Mapping, list, tuple)):
                    walk(child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for child in node[:200]:
                walk(child, depth + 1)

    walk(value)
    return found
