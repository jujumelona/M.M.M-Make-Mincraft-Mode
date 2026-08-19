from __future__ import annotations

import dis
from functools import partial, wraps
from typing import Any, Callable

_MARKER = "__mmm_static_runtime_composer_identity_v1__"


def harden_runtime_composer_identity() -> None:
    """Keep composition graph identity implementation-static across runtime mutation.

    Read-only callable/module/type closure dependencies remain part of graph identity,
    so swapping an installer target still fails closed. Closure cells written by the
    installer itself are execution state and are represented only by type; counters,
    markers and other STORE_DEREF mutations therefore cannot make a completed graph
    appear to have changed on safe re-entry.
    """
    from . import runtime_contract_composer as composer

    current = composer._installer_identity
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def installer_identity(value: Callable[[], None]) -> tuple[Any, ...]:
        partial_identity = (
            composer._dependency_identity(value) if isinstance(value, partial) else None
        )
        target: Any = value.func if isinstance(value, partial) else value
        bound_owner = getattr(target, "__self__", None)
        target = getattr(target, "__func__", target)
        code = getattr(target, "__code__", None)
        closure = getattr(target, "__closure__", None) or ()
        freevars = tuple(getattr(code, "co_freevars", ()))
        written = _written_freevars(target)
        captured: list[tuple[str, Any]] = []
        for name, cell in zip(freevars, closure, strict=False):
            try:
                item = cell.cell_contents
            except ValueError:
                captured.append((name, "<empty-cell>"))
                continue
            if name in written:
                captured.append(
                    (
                        name,
                        (
                            "runtime-cell-type",
                            type(item).__module__,
                            type(item).__qualname__,
                        ),
                    )
                )
            else:
                captured.append((name, composer._dependency_identity(item)))

        return (
            composer._static_callable_identity(target),
            composer._code_digest(target),
            tuple(captured),
            composer._dependency_identity(bound_owner) if bound_owner is not None else None,
            partial_identity,
        )

    setattr(installer_identity, _MARKER, True)
    composer._installer_identity = installer_identity


def _written_freevars(value: Any) -> frozenset[str]:
    target = getattr(value, "__func__", value)
    try:
        instructions = dis.get_instructions(target)
    except (TypeError, ValueError):
        return frozenset()
    written: set[str] = set()
    for instruction in instructions:
        if instruction.opname in {"STORE_DEREF", "DELETE_DEREF"} and isinstance(
            instruction.argval, str
        ):
            written.add(instruction.argval)
    return frozenset(written)


__all__ = ["harden_runtime_composer_identity"]
