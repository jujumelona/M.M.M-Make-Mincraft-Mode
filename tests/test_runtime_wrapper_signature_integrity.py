from __future__ import annotations

from functools import wraps

import pytest

import minecraft_mod_ai  # noqa: F401  # package import must complete runtime bootstrap
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.runtime_wrapper_integrity import (
    RuntimeWrapperIntegrityError,
    audit_installed_wrappers,
    deepest_wrapped,
    wrapper_compatibility_error,
)


def test_all_installed_mmm_wrappers_preserve_original_call_surface() -> None:
    report = audit_installed_wrappers()
    assert report.checked > 0, "runtime bootstrap exposed no inspectable MMM wrapper chains"
    assert report.ok, "runtime wrapper API drift detected:\n" + "\n".join(
        f"{issue.binding}: {issue.error}" for issue in report.issues
    )


def test_live_target_lowering_does_not_rebind_complete_planner_session() -> None:
    current = CompleteGameDesignPlanner._plan_in_session
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        assert not getattr(current, "_mmm_live_ai_module_lowering", False)
        current = getattr(current, "__wrapped__", None)


def test_outer_wrapper_cannot_hide_narrow_signature_with_wraps() -> None:
    def original(left: object, right: object, *, mode: object = None) -> None:
        del left, right, mode

    @wraps(original)
    def narrowed(left: object) -> None:
        del left

    error = wrapper_compatibility_error(narrowed, original)
    assert "does not preserve original" in error


def test_broad_wrapper_preserves_original_call_surface() -> None:
    def original(left: object, right: object = None, *, mode: object = None) -> None:
        del left, right, mode

    @wraps(original)
    def broad(*args: object, **kwargs: object) -> None:
        del args, kwargs

    assert wrapper_compatibility_error(broad, original) == ""


def test_varargs_and_varkw_cannot_be_removed() -> None:
    def original(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @wraps(original)
    def no_varargs(**kwargs: object) -> None:
        del kwargs

    @wraps(original)
    def no_varkw(*args: object) -> None:
        del args

    assert "*args" in wrapper_compatibility_error(no_varargs, original)
    assert "**kwargs" in wrapper_compatibility_error(no_varkw, original)


def test_cyclic_wrapped_chain_is_rejected() -> None:
    def wrapper() -> None:
        return None

    wrapper.__wrapped__ = wrapper  # type: ignore[attr-defined]
    with pytest.raises(RuntimeWrapperIntegrityError, match="cyclic __wrapped__ chain"):
        deepest_wrapped(wrapper)
