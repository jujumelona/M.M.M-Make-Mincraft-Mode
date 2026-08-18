from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator
from typing import Any

import minecraft_mod_ai  # noqa: F401  # package import must complete runtime bootstrap


_PACKAGE_PREFIX = "minecraft_mod_ai"
_SENTINEL = object()


def _wrapped_functions() -> Iterator[tuple[str, Any]]:
    """Yield MMM-owned installed wrappers from loaded modules and class namespaces."""

    for module_name, module in sorted(sys.modules.items()):
        if module is None or not (
            module_name == _PACKAGE_PREFIX
            or module_name.startswith(_PACKAGE_PREFIX + ".")
        ):
            continue
        for attribute, value in tuple(vars(module).items()):
            if _is_mmm_wrapper(value):
                yield f"{module_name}.{attribute}", value
            if not inspect.isclass(value):
                continue
            if not str(getattr(value, "__module__", "")).startswith(_PACKAGE_PREFIX):
                continue
            for member_name, raw in tuple(vars(value).items()):
                if isinstance(raw, (staticmethod, classmethod)):
                    candidate = raw.__func__
                    if _is_mmm_wrapper(candidate):
                        yield f"{module_name}.{value.__qualname__}.{member_name}", candidate
                    continue
                if isinstance(raw, property):
                    for accessor_name, candidate in (
                        ("fget", raw.fget),
                        ("fset", raw.fset),
                        ("fdel", raw.fdel),
                    ):
                        if _is_mmm_wrapper(candidate):
                            yield (
                                f"{module_name}.{value.__qualname__}.{member_name}."
                                f"{accessor_name}",
                                candidate,
                            )
                    continue
                if _is_mmm_wrapper(raw):
                    yield f"{module_name}.{value.__qualname__}.{member_name}", raw


def _is_mmm_wrapper(value: Any) -> bool:
    return bool(
        callable(value)
        and callable(getattr(value, "__wrapped__", None))
        and str(getattr(value, "__module__", "")).startswith(_PACKAGE_PREFIX)
    )


def _deepest_wrapped(value: Any) -> Any:
    current = value
    seen = {id(current)}
    while callable(getattr(current, "__wrapped__", None)):
        current = current.__wrapped__
        if id(current) in seen:
            raise AssertionError("cyclic __wrapped__ chain")
        seen.add(id(current))
    return current


def _has_kind(signature: inspect.Signature, kind: inspect._ParameterKind) -> bool:
    return any(parameter.kind is kind for parameter in signature.parameters.values())


def _representative_calls(
    signature: inspect.Signature,
) -> tuple[tuple[tuple[object, ...], dict[str, object]], ...]:
    params = tuple(signature.parameters.values())
    positional_only = tuple(
        parameter
        for parameter in params
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    )
    positional_or_keyword = tuple(
        parameter
        for parameter in params
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    keyword_only = tuple(
        parameter
        for parameter in params
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )

    required_positional_only = tuple(
        parameter
        for parameter in positional_only
        if parameter.default is inspect.Parameter.empty
    )
    required_positional_or_keyword = tuple(
        parameter
        for parameter in positional_or_keyword
        if parameter.default is inspect.Parameter.empty
    )
    required_keyword_only = tuple(
        parameter
        for parameter in keyword_only
        if parameter.default is inspect.Parameter.empty
    )

    minimal = (
        (_SENTINEL,) * (
            len(required_positional_only) + len(required_positional_or_keyword)
        ),
        {parameter.name: _SENTINEL for parameter in required_keyword_only},
    )
    all_positional = (
        (_SENTINEL,) * (len(positional_only) + len(positional_or_keyword)),
        {parameter.name: _SENTINEL for parameter in keyword_only},
    )
    keyword_form = (
        (_SENTINEL,) * len(positional_only),
        {
            parameter.name: _SENTINEL
            for parameter in (*positional_or_keyword, *keyword_only)
        },
    )
    return minimal, all_positional, keyword_form


def _compatibility_error(outer: Any, original: Any) -> str:
    try:
        outer_signature = inspect.signature(outer, follow_wrapped=False)
        original_signature = inspect.signature(original, follow_wrapped=False)
    except (TypeError, ValueError) as exc:
        return f"signature unavailable: {type(exc).__name__}: {exc}"

    if _has_kind(original_signature, inspect.Parameter.VAR_POSITIONAL) and not _has_kind(
        outer_signature, inspect.Parameter.VAR_POSITIONAL
    ):
        return "original accepts *args but outer wrapper does not"
    if _has_kind(original_signature, inspect.Parameter.VAR_KEYWORD) and not _has_kind(
        outer_signature, inspect.Parameter.VAR_KEYWORD
    ):
        return "original accepts **kwargs but outer wrapper does not"

    for args, kwargs in _representative_calls(original_signature):
        try:
            outer_signature.bind(*args, **kwargs)
        except TypeError as exc:
            return (
                f"outer={outer_signature} does not preserve original="
                f"{original_signature}: {exc}"
            )
    return ""


def test_all_installed_mmm_wrappers_preserve_original_call_surface() -> None:
    checked = 0
    failures: list[str] = []
    for label, outer in _wrapped_functions():
        checked += 1
        try:
            original = _deepest_wrapped(outer)
        except AssertionError as exc:
            failures.append(f"{label}: {exc}")
            continue
        error = _compatibility_error(outer, original)
        if error:
            failures.append(f"{label}: {error}")

    assert checked > 0, "runtime bootstrap exposed no inspectable MMM wrapper chains"
    assert not failures, "runtime wrapper API drift detected:\n" + "\n".join(failures)
