from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolDefinition
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _native_tool_generation_response

MODEL = "unsloth/Qwen3.5-9B-MTP-GGUF:Qwen3.5-9B-UD-Q4_K_XL.gguf"
# These deterministic fixtures model observed Qwen-family failure shapes. They are
# synthetic regression fixtures, not live captures. A live failure may only be promoted
# to this suite after its source conditions are recorded separately.
FIXTURE_PROVENANCE = "synthetic_qwen35_failure_shape"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "maxLength": 256}},
    "required": ["answer"],
    "additionalProperties": False,
}
MESSAGES = ({"role": "user", "content": "Return the requested fixed template."},)


@dataclass
class TextReplayRouter:
    raw: str

    def generate_text(self, role: str, messages: Any, **kwargs: Any) -> str:
        return self.raw


@dataclass
class ToolReplayRouter:
    raw: str
    profile: str = "t4_local"

    class _Registry:
        @staticmethod
        def role(profile: str, role: str):
            return type("Role", (), {"adapter": "llama_cpp"})()

    registry = _Registry()

    def generate_tool_decision(
        self,
        role: str,
        messages: Any,
        *,
        tool_name: str,
        parameters: Any,
        description: str = "",
    ) -> dict[str, Any]:
        request = GenerationRequest(
            messages=tuple(messages),
            tools=(
                ToolDefinition(
                    name=tool_name,
                    description=description,
                    parameters=dict(parameters),
                ),
            ),
            tool_choice=tool_name,
            parallel_tool_calls=False,
        )
        # This is the exact production post-inference seam used by
        # llama_cpp_adapter._native_tool_completion after llama.cpp returns its
        # assistant message. The replay layer supplies only that raw message; it does
        # not own a parser or reproduce Qwen routing semantics.
        turn = _native_tool_generation_response(
            {"role": "assistant", "content": self.raw},
            request,
        )
        if len(turn.tool_calls) != 1:
            raise ValueError(
                f"expected exactly one production-routed native tool call, got {len(turn.tool_calls)}"
            )
        call = turn.tool_calls[0]
        if call.name != tool_name:
            raise ValueError(f"unexpected tool name: {call.name}")
        return dict(call.arguments)


def replay_text(raw: str):
    return generate_fixed_template_value(
        TextReplayRouter(raw),
        "planner",
        MESSAGES,
        response_schema=SCHEMA,
    )


def replay_tool(raw: str):
    return generate_fixed_template_value(
        ToolReplayRouter(raw),
        "planner",
        MESSAGES,
        response_schema=SCHEMA,
        enable_tools=False,
    )
