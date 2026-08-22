from __future__ import annotations

"""Protocol-safe launcher for first-party MCP stdio servers."""

import importlib
import sys
from collections.abc import Callable
from types import ModuleType

from .mcp_stdio_support import install_mcp_protocol_print_guard

_ALLOWED_MODULES = frozenset(
    {
        "minecraft_mod_ai.mcp_server",
        "minecraft_mod_ai.mod_generation_mcp_server",
    }
)


def _server_main(module_name: str) -> Callable[[], None]:
    if module_name not in _ALLOWED_MODULES:
        raise SystemExit(f"unsupported first-party MCP server module: {module_name}")
    module: ModuleType = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise SystemExit(f"MCP server module has no callable main(): {module_name}")
    return main


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit(
            "usage: python -m minecraft_mod_ai.mcp_stdio_entrypoint "
            "<minecraft_mod_ai.mcp_server|minecraft_mod_ai.mod_generation_mcp_server>"
        )
    install_mcp_protocol_print_guard()
    _server_main(args[0])()


if __name__ == "__main__":
    main()
