from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_registry import ModelRegistry

PROMPT = (
    'Return exactly one JSON object matching this schema: '
    '{"answer": "string"}. Do not add markdown or explanation.'
)


def _chat_url(server_url: str) -> str:
    base = server_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return f"{base}/chat/completions"


def _raw_assistant_message(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("llama-server response is not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("llama-server response has no choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise TypeError("llama-server response has no assistant message")
    return dict(first["message"])


def run_capture(
    *,
    profile: str,
    role: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    stop: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    registry = ModelRegistry()
    registry.load_profile(profile)
    config = registry.role(profile, role)
    if config.adapter != "llama_cpp":
        raise RuntimeError(
            f"live Qwen capture requires llama_cpp, got {config.adapter!r} for {role!r}"
        )
    adapter = LlamaCppAdapter(config)
    request = GenerationRequest(messages=({"role": "user", "content": PROMPT},))
    server_url = adapter._server_url(request)
    payload: dict[str, Any] = {
        "model": "local",
        "messages": [dict(message) for message in request.messages],
        "seed": int(seed),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if stop:
        payload["stop"] = [stop]

    with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
        response = client.post(_chat_url(server_url), json=payload)
        response.raise_for_status()
        body = response.json()

    message = _raw_assistant_message(body)
    return {
        "schema": "mmm/qwen-live-capture",
        "provenance": "real_capture",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "role": role,
        "model_id": config.model_id,
        "conditions": {
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
        },
        "request": {"messages": payload["messages"]},
        "raw_assistant_message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real Qwen/llama.cpp inference condition and preserve raw output."
    )
    parser.add_argument("--profile", default="t4_local")
    parser.add_argument("--role", default="planner")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--stop", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = run_capture(
        profile=args.profile,
        role=args.role,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=args.stop or None,
        timeout_seconds=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
