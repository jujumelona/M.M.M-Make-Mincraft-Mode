"""The host-owned ABI between authored Java units and the single mod entrypoint."""
from __future__ import annotations

import re
from collections.abc import Iterable

ACTIVATION_METHOD = "initialize"
ACTIVATION_API = f"public static void {ACTIVATION_METHOD}()"


def activation_public_api(declarations: Iterable[str], *, active: bool) -> list[str]:
    """Add the host hook without changing any model-authored member semantics."""
    result = list(declarations)
    if not active:
        return result
    for declaration in result:
        if re.search(rf"\b{ACTIVATION_METHOD}\s*\(\s*\)", declaration) and declaration != ACTIVATION_API:
            raise ValueError(
                "IMPLEMENTATION_IR_ACTIVATION_API_CONFLICT: activation requires "
                f"{ACTIVATION_API}; conflicting declaration: {declaration}"
            )
    if ACTIVATION_API not in result:
        result.append(ACTIVATION_API)
    return result


def activation_call(symbol: str) -> str:
    return f"{symbol}.{ACTIVATION_METHOD}();"
