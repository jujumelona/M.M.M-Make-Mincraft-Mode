from __future__ import annotations

import pytest

from minecraft_mod_ai.planner_structured_router import structured_planner_router
from minecraft_mod_ai.structured_output import StructuredOutputValidationError


class _Router:
    def __init__(self, outcomes=None) -> None:
        self.calls = []
        self.profile = "t4_local"
        self._outcomes = list(outcomes or ["{}"])

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, dict(kwargs)))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _invalid() -> StructuredOutputValidationError:
    return StructuredOutputValidationError(
        output='{"broken"',
        errors=("$: invalid JSON",),
    )


def test_structured_planner_json_disables_tools() -> None:
    base = _Router()
    proxy = structured_planner_router(base)
    result = proxy.generate_text(
        "planner",
        [{"role": "user", "content": "return json"}],
        response_format="json",
    )
    assert result == "{}"
    assert base.calls[-1][2]["enable_tools"] is False


def test_structured_planner_retries_whole_response_exactly_once() -> None:
    base = _Router([_invalid(), '{"ok":true}'])
    proxy = structured_planner_router(base)

    result = proxy.generate_text(
        "planner",
        [{"role": "user", "content": "return json"}],
        response_format="json",
    )

    assert result == '{"ok":true}'
    assert len(base.calls) == 2
    assert all(call[2]["enable_tools"] is False for call in base.calls)


def test_structured_planner_second_invalid_response_escapes() -> None:
    base = _Router([_invalid(), _invalid()])
    proxy = structured_planner_router(base)

    with pytest.raises(StructuredOutputValidationError):
        proxy.generate_text(
            "planner",
            [{"role": "user", "content": "return json"}],
            response_format="json",
        )

    assert len(base.calls) == 2


def test_non_planner_or_non_json_calls_keep_normal_policy_and_no_retry() -> None:
    base = _Router([_invalid()])
    proxy = structured_planner_router(base)
    with pytest.raises(StructuredOutputValidationError):
        proxy.generate_text(
            "planner",
            [{"role": "user", "content": "design"}],
            response_format="text",
        )
    assert len(base.calls) == 1
    assert "enable_tools" not in base.calls[-1][2]

    base = _Router([_invalid()])
    proxy = structured_planner_router(base)
    with pytest.raises(StructuredOutputValidationError):
        proxy.generate_text(
            "coder",
            [{"role": "user", "content": "patch"}],
            response_format="json",
        )
    assert len(base.calls) == 1
    assert "enable_tools" not in base.calls[-1][2]


def test_proxy_is_idempotent_and_delegates_attributes() -> None:
    base = _Router()
    proxy = structured_planner_router(base)
    assert structured_planner_router(proxy) is proxy
    assert proxy.profile == "t4_local"
