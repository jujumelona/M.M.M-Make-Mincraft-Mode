from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
)
from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.llama_finish_reason_contract import (
    LlamaCompletionBoundaryError,
    OUTPUT_EXHAUSTED,
    completion_boundary_error,
)
from minecraft_mod_ai.model_adapters.base import ModelBackendError


_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "minLength": 1, "maxLength": 32},
    },
    "required": ["value"],
    "additionalProperties": False,
}


class _Registry:
    @staticmethod
    def role(_profile: str, _role: str):
        return SimpleNamespace(adapter="llama_cpp")


class _CapturingRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.kwargs = None

    def generate_tool_decision(self, role, messages, **kwargs):
        del role, messages
        self.kwargs = dict(kwargs)
        return {"value": "ok"}


class _BoundaryRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def generate_tool_decision(self, role, messages, **kwargs):
        del role, messages, kwargs
        raise self.error


def test_planner_fixed_template_uses_atomic_concern_output_budget() -> None:
    router = _CapturingRouter()

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "author one concern"},),
        response_schema=_SCHEMA,
        enable_tools=False,
    )

    assert value == {"value": "ok"}
    assert router.kwargs is not None
    assert (
        router.kwargs["output_token_ceiling"]
        == ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING
    )
    assert router.kwargs["force_non_thinking"] is True


def test_planner_completion_boundary_is_not_reclassified_as_semantic_rejection() -> None:
    boundary = LlamaCompletionBoundaryError(
        "bounded output exhausted",
        kind=OUTPUT_EXHAUSTED,
        prompt_tokens=1184,
        completion_tokens=4096,
        max_tokens=4096,
    )
    backend_error = ModelBackendError(
        role="planner",
        model_id="test/model",
        cause=boundary,
    )
    router = _BoundaryRouter(backend_error)

    with pytest.raises(ModelBackendError) as captured:
        generate_fixed_template_value(
            router,
            "planner",
            ({"role": "user", "content": "author one concern"},),
            response_schema=_SCHEMA,
            enable_tools=False,
        )

    assert captured.value is backend_error
    assert completion_boundary_error(captured.value) is boundary
    assert "FIXED_TEMPLATE_PLANNER_SEMANTIC_UNIT_REJECTED" not in str(
        captured.value
    )
