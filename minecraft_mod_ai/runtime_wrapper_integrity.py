from __future__ import annotations

"""Runtime integrity audit for installed monkey-patch wrapper chains.

Contract wrappers are allowed to change behavior, not the callable surface that
existing production callers rely on. The audit validates the *actual callable* at
every layer of each ``__wrapped__`` chain (``follow_wrapped=False``) against the
deepest original target, so a broad outer wrapper cannot hide a narrowed middle
wrapper that would still fail at runtime.
"""

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Iterator


_PACKAGE_PREFIX = "minecraft_mod_ai"
_SENTINEL = object()


class RuntimeWrapperIntegrityError(RuntimeError):
    """Raised when an installed runtime wrapper no longer preserves its original API."""


@dataclass(frozen=True)
class WrapperIntegrityIssue:
    binding: str
    error: str


@dataclass(frozen=True)
class WrapperIntegrityReport:
    checked: int
    issues: tuple[WrapperIntegrityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def _is_package_module(module_name: str, package_prefix: str) -> bool:
    return module_name == package_prefix or module_name.startswith(package_prefix + ".")


def _is_owned_wrapper(value: Any, package_prefix: str) -> bool:
    return bool(
        callable(value)
        and callable(getattr(value, "__wrapped__", None))
        and _is_package_module(
            str(getattr(value, "__module__", "")),
            package_prefix,
        )
    )


def iter_installed_wrappers(
    package_prefix: str = _PACKAGE_PREFIX,
) -> Iterator[tuple[str, Any]]:
    """Yield package-owned wrapper bindings from loaded modules and classes."""

    for module_name, module in sorted(sys.modules.items()):
        if module is None or not _is_package_module(module_name, package_prefix):
            continue
        for attribute, value in tuple(vars(module).items()):
            if _is_owned_wrapper(value, package_prefix):
                yield f"{module_name}.{attribute}", value
            if not inspect.isclass(value):
                continue
            if not _is_package_module(
                str(getattr(value, "__module__", "")),
                package_prefix,
            ):
                continue
            for member_name, raw in tuple(vars(value).items()):
                if isinstance(raw, (staticmethod, classmethod)):
                    candidate = raw.__func__
                    if _is_owned_wrapper(candidate, package_prefix):
                        yield f"{module_name}.{value.__qualname__}.{member_name}", candidate
                    continue
                if isinstance(raw, property):
                    for accessor_name, candidate in (
                        ("fget", raw.fget),
                        ("fset", raw.fset),
                        ("fdel", raw.fdel),
                    ):
                        if _is_owned_wrapper(candidate, package_prefix):
                            yield (
                                f"{module_name}.{value.__qualname__}.{member_name}."
                                f"{accessor_name}",
                                candidate,
                            )
                    continue
                if _is_owned_wrapper(raw, package_prefix):
                    yield f"{module_name}.{value.__qualname__}.{member_name}", raw


def wrapped_chain(value: Any) -> tuple[Any, ...]:
    """Return outer-to-original wrapper chain and reject cycles."""

    chain = [value]
    seen = {id(value)}
    current = value
    while callable(getattr(current, "__wrapped__", None)):
        current = current.__wrapped__
        if id(current) in seen:
            raise RuntimeWrapperIntegrityError("cyclic __wrapped__ chain")
        seen.add(id(current))
        chain.append(current)
    return tuple(chain)


def deepest_wrapped(value: Any) -> Any:
    """Return the deepest callable in a wrapper chain and reject cycles."""

    return wrapped_chain(value)[-1]


def _has_kind(signature: inspect.Signature, kind: Any) -> bool:
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
        (_SENTINEL,)
        * (len(required_positional_only) + len(required_positional_or_keyword)),
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


def wrapper_compatibility_error(outer: Any, original: Any) -> str:
    """Return an explanation if ``outer`` narrows ``original``'s call surface."""

    try:
        outer_signature = inspect.signature(outer, follow_wrapped=False)
        original_signature = inspect.signature(original, follow_wrapped=False)
    except (TypeError, ValueError) as exc:
        return f"signature unavailable: {type(exc).__name__}: {exc}"

    if _has_kind(original_signature, inspect.Parameter.VAR_POSITIONAL) and not _has_kind(
        outer_signature, inspect.Parameter.VAR_POSITIONAL
    ):
        return "original accepts *args but wrapper layer does not"
    if _has_kind(original_signature, inspect.Parameter.VAR_KEYWORD) and not _has_kind(
        outer_signature, inspect.Parameter.VAR_KEYWORD
    ):
        return "original accepts **kwargs but wrapper layer does not"

    for args, kwargs in _representative_calls(original_signature):
        try:
            outer_signature.bind(*args, **kwargs)
        except TypeError as exc:
            return (
                f"layer={outer_signature} does not preserve original="
                f"{original_signature}: {exc}"
            )
    return ""


def _layer_identity(layer: Any) -> str:
    module = str(getattr(layer, "__module__", ""))
    qualname = str(
        getattr(layer, "__qualname__", "")
        or getattr(layer, "__name__", "")
        or type(layer).__qualname__
    )
    return f"{module}:{qualname}"


def audit_installed_wrappers(
    package_prefix: str = _PACKAGE_PREFIX,
) -> WrapperIntegrityReport:
    checked = 0
    issues: list[WrapperIntegrityIssue] = []
    seen_bindings: set[tuple[str, int]] = set()
    for binding, outer in iter_installed_wrappers(package_prefix):
        key = (binding, id(outer))
        if key in seen_bindings:
            continue
        seen_bindings.add(key)
        checked += 1
        try:
            chain = wrapped_chain(outer)
        except RuntimeWrapperIntegrityError as exc:
            issues.append(WrapperIntegrityIssue(binding, str(exc)))
            continue
        original = chain[-1]
        for layer_index, layer in enumerate(chain[:-1]):
            error = wrapper_compatibility_error(layer, original)
            if error:
                issues.append(
                    WrapperIntegrityIssue(
                        binding,
                        f"layer[{layer_index}] {_layer_identity(layer)}: {error}",
                    )
                )
    return WrapperIntegrityReport(checked=checked, issues=tuple(issues))


def verify_installed_wrappers(
    package_prefix: str = _PACKAGE_PREFIX,
) -> WrapperIntegrityReport:
    """Fail fast when bootstrap produced a narrowed or cyclic wrapper chain."""

    report = audit_installed_wrappers(package_prefix)
    if report.issues:
        rendered = "\n".join(
            f"- {issue.binding}: {issue.error}" for issue in report.issues
        )
        raise RuntimeWrapperIntegrityError(
            "runtime wrapper API drift detected:\n" + rendered
        )
    return report


__all__ = [
    "RuntimeWrapperIntegrityError",
    "WrapperIntegrityIssue",
    "WrapperIntegrityReport",
    "audit_installed_wrappers",
    "deepest_wrapped",
    "iter_installed_wrappers",
    "verify_installed_wrappers",
    "wrapped_chain",
    "wrapper_compatibility_error",
]
