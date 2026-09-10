from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import model_output_atomicity_contract as contract


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "maxLength": 256}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _Registry:
    def __init__(self, adapter: str = "llama_cpp") -> None:
        self.adapter = adapter

    def role(self, _profile: str, _role: str):
        return SimpleNamespace(adapter=self.adapter)


class _Router:
    def __init__(self, adapter: str = "llama_cpp") -> None:
        self.registry = _Registry(adapter)
        self.profile = "test"
        self.text_calls = 0
        self.tool_calls = 0
        self.tool_messages = ()
        self.tool_parameters = None

    @staticmethod
    def _tools_enabled(*, enable_tools: bool, stage: str, adapter_name: str) -> bool:
        return bool(enable_tools and stage and adapter_name == "llama_cpp")

    def generate_text(self, _role, _messages, **_kwargs):
        self.text_calls += 1
        return '{"answer":"broken",}'

    def generate_tool_decision(
        self,
        _role,
        messages,
        *,
        tool_name,
        parameters,
        description="",
    ):
        self.tool_calls += 1
        self.tool_messages = tuple(messages)
        self.tool_parameters = parameters
        assert tool_name == "submit_fixed_template"
        assert description
        return {"answer": "filled"}


def _installed_router_class():
    class Router(_Router):
        pass

    module = SimpleNamespace(
        ModelRouter=Router,
        _ROLE_TOOL_STAGE={"planner": "planning"},
    )
    contract.install(model_router_module=module)
    return Router, module


def test_structured_generation_never_uses_free_json_text_path():
    Router, module = _installed_router_class()
    router = Router()

    output = router.generate_text(
        "planner",
        ({"role": "user", "content": "fill it"},),
        response_format="json",
        response_schema=SCHEMA,
        enable_tools=False,
    )

    assert json.loads(output) == {"answer": "filled"}
    assert router.text_calls == 0
    assert router.tool_calls == 1
    assert router.tool_parameters == SCHEMA
    contract.assert_installed(model_router_module=module)


def test_plain_text_generation_is_unchanged():
    Router, _module = _installed_router_class()
    router = Router()

    output = router.generate_text(
        "planner",
        ({"role": "user", "content": "plain"},),
        response_format="text",
        enable_tools=False,
    )

    assert output == '{"answer":"broken",}'
    assert router.text_calls == 1
    assert router.tool_calls == 0


def test_tool_or_media_semantics_run_before_fixed_template_fill():
    Router, _module = _installed_router_class()
    router = Router()

    output = router.generate_text(
        "planner",
        ({"role": "user", "content": "research then fill"},),
        response_format="json",
        response_schema=SCHEMA,
        enable_tools=True,
    )

    assert json.loads(output) == {"answer": "filled"}
    assert router.text_calls == 1
    assert router.tool_calls == 1
    assert any(
        "Completed semantic result" in str(message.get("content", ""))
        for message in router.tool_messages
    )


def test_non_object_root_is_wrapped_only_for_function_transport():
    class Router(_Router):
        def generate_tool_decision(self, _role, _messages, *, tool_name, parameters, description=""):
            self.tool_calls += 1
            self.tool_parameters = parameters
            assert tool_name == "submit_fixed_template"
            return {"value": ["a", "b"]}

    module = SimpleNamespace(
        ModelRouter=Router,
        _ROLE_TOOL_STAGE={"planner": "planning"},
    )
    contract.install(model_router_module=module)
    router = Router()
    schema = {
        "type": "array",
        "items": {"type": "string", "maxLength": 256},
        "maxItems": 4,
    }

    output = router.generate_text(
        "planner",
        ({"role": "user", "content": "fill list"},),
        response_format="json",
        response_schema=schema,
        enable_tools=False,
    )

    assert json.loads(output) == ["a", "b"]
    assert router.tool_parameters == {
        "type": "object",
        "properties": {"value": schema},
        "required": ["value"],
        "additionalProperties": False,
    }


def test_mock_profile_keeps_deterministic_fixture_transport_only():
    Router, _module = _installed_router_class()
    router = Router(adapter="mock")

    output = router.generate_text(
        "planner",
        ({"role": "user", "content": "fixture"},),
        response_format="json",
        response_schema=SCHEMA,
        enable_tools=False,
    )

    assert output == '{"answer":"broken",}'
    assert router.text_calls == 1
    assert router.tool_calls == 0
