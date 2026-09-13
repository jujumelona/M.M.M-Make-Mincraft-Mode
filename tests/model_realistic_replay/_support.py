from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_calls

MODEL = "unsloth/Qwen3.5-9B-MTP-GGUF:Qwen3.5-9B-UD-Q4_K_XL.gguf"
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
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
    def generate_tool_decision(self, role: str, messages: Any, *, tool_name: str, parameters: Any, description: str = "") -> dict[str, Any]:
        calls = parse_qwen_tool_calls(self.raw)
        if len(calls) != 1:
            raise ValueError(f"expected exactly one Qwen tool call, got {len(calls)}")
        call = calls[0]
        if call.name != tool_name:
            raise ValueError(f"unexpected tool name: {call.name}")
        return call.arguments

def replay_text(raw: str):
    return generate_fixed_template_value(TextReplayRouter(raw), "planner", MESSAGES, response_schema=SCHEMA)

def replay_tool(raw: str):
    return generate_fixed_template_value(ToolReplayRouter(raw), "planner", MESSAGES, response_schema=SCHEMA, enable_tools=False)
