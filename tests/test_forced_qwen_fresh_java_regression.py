from __future__ import annotations

import copy
from types import SimpleNamespace

from minecraft_mod_ai.llama_server_hardware_policy import _request_max_tokens
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.progress_aware_tool_loop import (
    TargetMutationContext,
    _source_edit_schema_for_context,
)
from minecraft_mod_ai.qwen_agent_family_contract import _apply_family_payload_policy
from minecraft_mod_ai.small_model_task_capsule_contract import (
    TaskAnchor,
    TaskCapsule,
    bind_source_edit_arguments,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


_PATH = "src/main/java/dev/mmm/debugfixture/DebugToken.java"


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one semantic source edit.",
            "parameters": copy.deepcopy(SOURCE_EDIT_SCHEMA),
        },
    }


def _config() -> SimpleNamespace:
    non_thinking = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
    }
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        role="coder",
        max_new_tokens=8192,
        extra={
            "runtime_contract": "qwen",
            "qwen_family": "qwen3.5",
            "qwen_tool_markup": "qwen3_coder_xml",
            "qwen_action_thinking_control": "enable_thinking_false",
            "qwen_preserve_thinking": False,
            "qwen_reasoning_effort": False,
            "qwen_assistant_prefill": True,
            "agent_thinking": True,
            "sampling_profiles": {
                "general_thinking": dict(non_thinking),
                "precise_coding": dict(non_thinking),
                "non_thinking": dict(non_thinking),
            },
        },
    )


def test_forced_fresh_java_tool_turn_is_content_only_host_bound_and_non_thinking() -> None:
    context = TargetMutationContext(
        target_path=_PATH,
        target_symbol="DebugToken",
        is_new_file=True,
        evidence_source="evidence_fresh_owned_anchor",
        writable_paths=(_PATH,),
        creatable_paths=(_PATH,),
        target_pinned=True,
    )
    narrowed = _source_edit_schema_for_context(_tool(), context)
    parameters = narrowed["function"]["parameters"]
    assert set(parameters["properties"]) == {"content"}
    assert parameters["required"] == ["content"]
    assert parameters["additionalProperties"] is False

    capsule = TaskCapsule(
        task_id="debug_token",
        module_kind="custom_java",
        primary_path=_PATH,
        primary_symbol="DebugToken",
        anchors=(
            TaskAnchor(
                kind="symbol",
                path=_PATH,
                symbol="DebugToken",
                status="host_reserved",
                ownership="exclusive",
                module_id=":",
                source_set="main",
            ),
        ),
        reuse_action="fresh",
        required_gates=("target_compile",),
        task_sha256="sha256:test",
        capsule_sha256="sha256:test-capsule",
    )
    source = "package dev.mmm.debugfixture; public final class DebugToken {}"
    bound = bind_source_edit_arguments({"content": source}, capsule)
    assert bound == {
        "content": source,
        "path": _PATH,
        "operation": "create_file",
    }

    request = GenerationRequest(
        messages=({"role": "user", "content": "Implement the host-pinned fresh Java target."},),
        tools=(narrowed,),
        tool_validation_schemas=(narrowed,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
        metadata={"mmm_output_token_ceiling": 4096},
    )
    config = _config()
    payload = _apply_family_payload_policy(
        {"temperature": 0.0},
        config=config,
        role="coder",
        request=request,
    )

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert _request_max_tokens(SimpleNamespace(config=config), request) == 4096
