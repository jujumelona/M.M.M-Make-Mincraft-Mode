from __future__ import annotations

_VANILLA_WORLD_EDIT_ROOTS = frozenset(
    {
        "clone",
        "fill",
        "fillbiome",
        "function",
        "place",
        "schedule",
        "setblock",
    }
)
_WORLD_EDIT_PLUGIN_ROOTS = frozenset(
    {
        "paste",
        "schem",
        "schematic",
        "we",
        "worldedit",
    }
)
_WORLD_EDIT_NAMESPACES = frozenset({"we", "worldedit"})
_EXECUTE_ROOT = "execute"


def _split_command_root(token: str) -> tuple[str | None, str]:
    root = token.casefold()
    if ":" not in root:
        return None, root
    namespace, local = root.split(":", 1)
    return namespace, local


def _direct_scope_violation(namespace: str | None, root: str) -> str | None:
    if namespace in _WORLD_EDIT_NAMESPACES:
        return f"world-edit command namespace {namespace}"
    if namespace is None:
        if root in _VANILLA_WORLD_EDIT_ROOTS:
            return f"world-edit command {root}"
        if root in _WORLD_EDIT_PLUGIN_ROOTS:
            return f"world-edit plugin command {root}"
        return None
    if namespace == "minecraft" and root in _VANILLA_WORLD_EDIT_ROOTS:
        return f"world-edit command minecraft:{root}"
    return None


def _nested_execute_command(
    tokens: list[str],
    *,
    namespace: str | None,
    root: str,
) -> str | None:
    if root != _EXECUTE_ROOT or namespace not in {None, "minecraft"}:
        return None
    for index, token in enumerate(tokens[1:], start=1):
        if token.casefold() == "run" and index + 1 < len(tokens):
            return " ".join(tokens[index + 1 :])
    return None


def server_command_scope_violation(command: str) -> str | None:
    """Return why a runtime command escapes mod playtest scope, or ``None``.

    Runtime profiles retain their positive allowlist authority. This invariant only
    prevents that configurable allowlist from turning the disposable test server into
    a generic world editor. Namespaced mod commands are deliberately not classified by
    their local command name; e.g. ``mymod:fill`` remains a normal mod API and is still
    subject to the profile's allowlist.
    """

    text = str(command or "").strip()
    if not text:
        return None
    if text.startswith("//"):
        return "WorldEdit double-slash command"

    # Server stdin normally omits '/', but accepting one syntactically must not create
    # a second policy path. A second slash was handled above before stripping.
    if text.startswith("/"):
        text = text[1:].lstrip()
    tokens = text.split()
    if not tokens:
        return None

    namespace, root = _split_command_root(tokens[0])
    violation = _direct_scope_violation(namespace, root)
    if violation is not None:
        return violation

    # ``execute`` is useful for scoped verification, so preserve it while recursively
    # checking the command behind its final ``run`` clause. This also catches nested
    # execute chains without trying to implement Minecraft's full command grammar.
    nested = _nested_execute_command(tokens, namespace=namespace, root=root)
    if nested is not None:
        return server_command_scope_violation(nested)
    return None


def validate_server_command_scope(command: str) -> None:
    violation = server_command_scope_violation(command)
    if violation is None:
        return
    raise ValueError(
        "Server command is outside M.M.M's mod playtest scope: "
        f"{command!r} ({violation})."
    )


__all__ = ["server_command_scope_violation", "validate_server_command_scope"]
