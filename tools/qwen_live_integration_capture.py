from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minecraft_mod_ai.llama_server_autotune import ensure_tuned_server
from minecraft_mod_ai.model_adapters.base import GenerationRequest, GenerationResponse
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    LlamaCppAdapter,
    _completion_exchange,
)
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.qwen_agent_family_contract import (
    _ensure_tool_safe_runtime,
    _restore_tool_runtime,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA
from minecraft_mod_ai.structured_output import validate_structured_output

JSON_PROMPT = (
    'Return exactly one JSON object matching this schema: '
    '{"answer": "string"}. Do not add markdown or explanation.'
)
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
SOURCE_EDIT_TARGET = "src/main/java/demo/CaptureProbe.java"
SOURCE_EDIT_OLD = "return 0;"
SOURCE_EDIT_NEW = "return 1;"
SOURCE_EDIT_PROMPT = (
    "You are applying one already-decided Java repair. Call apply_source_edit exactly once. "
    f"Use operation=replace_exact, path={SOURCE_EDIT_TARGET!r}, old={SOURCE_EDIT_OLD!r}, "
    f"new={SOURCE_EDIT_NEW!r}, count=1. Emit no prose."
)
SOURCE_EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_source_edit",
        "description": "Apply one source edit to the host-authorized project.",
        "parameters": SOURCE_EDIT_SCHEMA,
    },
}


def _request_for_scenario(scenario: str, *, max_tokens: int | None) -> GenerationRequest:
    metadata: dict[str, Any] = {}
    if max_tokens is not None and max_tokens > 0:
        metadata["mmm_output_token_ceiling"] = int(max_tokens)

    if scenario == "json":
        return GenerationRequest(
            messages=({"role": "user", "content": JSON_PROMPT},),
            response_format="json",
            response_schema=RESPONSE_SCHEMA,
            metadata=metadata,
        )
    if scenario == "source-edit":
        return GenerationRequest(
            messages=({"role": "user", "content": SOURCE_EDIT_PROMPT},),
            tools=(SOURCE_EDIT_TOOL,),
            tool_validation_schemas=(SOURCE_EDIT_TOOL,),
            tool_choice={
                "type": "function",
                "function": {"name": "apply_source_edit"},
            },
            parallel_tool_calls=False,
            metadata={**metadata, "tool_stage": "generation"},
        )
    raise ValueError(f"unsupported live-capture scenario: {scenario!r}")


def _production_validation(
    turn: GenerationResponse,
    *,
    scenario: str,
) -> tuple[bool, str | None, str]:
    if scenario == "json":
        content = turn.content
        if not content:
            return False, "production response seam returned no visible content", ""
        try:
            validate_structured_output(
                content,
                response_format="json",
                response_schema=RESPONSE_SCHEMA,
            )
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", content
        return True, None, content

    calls = tuple(turn.tool_calls)
    if len(calls) != 1:
        return False, f"expected one admitted tool call, got {len(calls)}", turn.content
    call = calls[0]
    expected = {
        "operation": "replace_exact",
        "path": SOURCE_EDIT_TARGET,
        "old": SOURCE_EDIT_OLD,
        "new": SOURCE_EDIT_NEW,
        "count": 1,
    }
    actual = {key: call.arguments.get(key) for key in expected}
    if call.name != "apply_source_edit":
        return False, f"unexpected tool name: {call.name!r}", turn.content
    if actual != expected:
        return False, f"tool arguments differ: {actual!r}", turn.content
    return True, None, turn.content


def _runtime_receipt() -> Any:
    raw = os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _registry_runtime_identity(config: Any) -> dict[str, Any]:
    extra = getattr(config, "extra", {})
    if not isinstance(extra, dict):
        extra = dict(extra) if hasattr(extra, "items") else {}
    keys = (
        "gguf_filename",
        "foundation_model_id",
        "foundation_revision",
        "chat_template",
        "qwen_family",
        "qwen_tool_markup",
        "request_policy",
        "sampling_profiles",
        "lora_adapters",
    )
    return {
        "model_id": config.model_id,
        "quantization": getattr(config, "quantization", None),
        "max_context": getattr(config, "max_context", None),
        "max_new_tokens": getattr(config, "max_new_tokens", None),
        "extra": {key: extra[key] for key in keys if key in extra},
    }


def _set_capture_timeout(seconds: float) -> str | None:
    previous = os.environ.get("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS")
    os.environ["MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS"] = str(float(seconds))
    return previous


def _restore_capture_timeout(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS", None)
    else:
        os.environ["MMM_LLAMA_COMPLETION_TIMEOUT_SECONDS"] = previous


def run_capture(
    *,
    profile: str,
    role: str,
    scenario: str,
    seed: int,
    temperature: float | None,
    max_tokens: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    registry = ModelRegistry()
    registry.load_profile(profile)
    config = registry.role(profile, role)
    if config.adapter != "llama_cpp":
        raise RuntimeError(
            f"live Qwen capture requires llama_cpp, got {config.adapter!r} for {role!r}"
        )

    request = _request_for_scenario(scenario, max_tokens=max_tokens)
    adapter = LlamaCppAdapter(config)
    tool_page = bool(request.tools)
    if tool_page:
        _ensure_tool_safe_runtime(config, request)
    server_url = ensure_tuned_server(config, request)
    runtime_receipt = _runtime_receipt()

    overrides: dict[str, Any] = {"seed": int(seed)}
    if temperature is not None:
        overrides["temperature"] = float(temperature)

    previous_timeout = _set_capture_timeout(timeout_seconds)
    try:
        turn, payload, raw_message = _completion_exchange(
            adapter,
            server_url,
            request,
            payload_overrides=overrides,
        )
    finally:
        _restore_capture_timeout(previous_timeout)
        if tool_page:
            restore_request = replace(
                request,
                tools=(),
                tool_validation_schemas=(),
                tool_choice=None,
                parallel_tool_calls=False,
            )
            _restore_tool_runtime(config, restore_request)

    accepted, validation_error, production_content = _production_validation(
        turn,
        scenario=scenario,
    )
    return {
        "schema": "mmm/qwen-live-capture-v2",
        "provenance": "real_capture",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "role": role,
        "scenario": scenario,
        "runtime_identity": _registry_runtime_identity(config),
        "runtime_receipt": runtime_receipt,
        "conditions": {
            "seed": seed,
            "temperature_override": temperature,
            "max_tokens_ceiling": max_tokens,
            "timeout_seconds": timeout_seconds,
        },
        "effective_request_payload": payload,
        "raw_assistant_message": dict(raw_message),
        "production_turn": {
            "content": turn.content,
            "reasoning_content": turn.reasoning_content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": dict(call.arguments),
                    "raw_arguments": call.raw_arguments,
                }
                for call in turn.tool_calls
            ],
        },
        "production_content": production_content,
        "production_validation": {
            "accepted": accepted,
            "error": validation_error,
        },
        "model_inference_performed": True,
        "minecraft_runtime_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one real Qwen/llama.cpp request through the production payload, "
            "LoRA routing, response parser, and schema/tool admission boundary."
        )
    )
    parser.add_argument("--profile", default="t4_local")
    parser.add_argument("--role", default="coder")
    parser.add_argument("--scenario", choices=("json", "source-edit"), default="source-edit")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional experiment override. Omit to preserve production sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional production output ceiling for the captured request.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = run_capture(
        profile=args.profile,
        role=args.role,
        scenario=args.scenario,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(capture, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output)
    if not capture["production_validation"]["accepted"]:
        print(capture["production_validation"]["error"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
