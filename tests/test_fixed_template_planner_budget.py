from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
)
from minecraft_mod_ai.authored_structured_design import _generate_authored_chunk
from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
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


class _ExhaustThenProjectRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_tool_decision(self, role, messages, **kwargs):
        del role, messages
        parameters = kwargs["parameters"]
        self.calls.append(dict(parameters))

        if len(self.calls) == 1:
            boundary = LlamaCompletionBoundaryError(
                "bounded output exhausted",
                kind=OUTPUT_EXHAUSTED,
                completion_tokens=ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
                max_tokens=ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
            )
            raise ModelBackendError(
                role="planner",
                model_id="test/model",
                cause=boundary,
            )

        properties = parameters["properties"]
        concern_names = [
            name
            for name, schema in properties.items()
            if isinstance(schema, dict)
            and schema.get("type") == "array"
            and isinstance(schema.get("items"), dict)
            and schema["items"].get("type") == "object"
            and name != "inapplicable_concerns"
        ]
        assert len(concern_names) == 1
        concern = concern_names[0]
        item_properties = properties[concern]["items"]["properties"]
        assert len(item_properties) == 1
        field_name = next(iter(item_properties))
        field_schema = item_properties[field_name]
        raw_type = field_schema.get("type")
        allowed_types = (
            tuple(raw_type)
            if isinstance(raw_type, list)
            else (raw_type,)
        )
        if "string" in allowed_types:
            field_value = f"value-{field_name}"
        elif "array" in allowed_types:
            field_value = [f"value-{field_name}"]
        elif "integer" in allowed_types:
            field_value = 1
        elif "number" in allowed_types:
            field_value = 1.0
        elif "boolean" in allowed_types:
            field_value = True
        elif "null" in allowed_types:
            field_value = None
        else:
            raise AssertionError(f"unsupported field schema: {field_schema!r}")
        return {concern: [{field_name: field_value}]}


def test_single_concern_output_exhaustion_narrows_to_host_field_projections() -> None:
    selected_section = ""
    selected_concern = ""
    selected_fields: tuple[str, ...] = ()
    for section, concerns in DETAIL_RECORDS.items():
        if section == "state_model":
            continue
        for concern, columns in concerns.items():
            fields = tuple(columns.split())
            if len(fields) >= 2:
                selected_section = section
                selected_concern = concern
                selected_fields = fields
                break
        if selected_concern:
            break

    assert selected_section
    assert selected_concern
    router = _ExhaustThenProjectRouter()

    value = _generate_authored_chunk(
        router,
        "build the requested feature",
        section=selected_section,
        chunk_index=1,
        chunk_count=1,
        concerns=(selected_concern,),
        completed={},
        include_evidence=False,
        media_paths=(),
    )

    assert len(router.calls) == 1 + len(selected_fields)
    assert len(value[selected_concern]) == 1
    assert set(value[selected_concern][0]) == set(selected_fields)
    for field_name in selected_fields:
        assert field_name in value[selected_concern][0]
