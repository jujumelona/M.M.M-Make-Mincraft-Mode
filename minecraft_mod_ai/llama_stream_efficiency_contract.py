from __future__ import annotations

import json
import os
import threading
import time
from functools import wraps
from typing import Any


_CLIENT_LOCK = threading.RLock()
_CLIENTS: dict[str, Any] = {}
_CLIENT_LIMIT = 4


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _client(server_url: str) -> Any:
    import httpx

    origin = server_url.rstrip("/")
    with _CLIENT_LOCK:
        client = _CLIENTS.get(origin)
        if client is not None:
            return client
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
        client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=8,
                keepalive_expiry=60.0,
            ),
        )
        _CLIENTS[origin] = client
        while len(_CLIENTS) > _CLIENT_LIMIT:
            key = next(iter(_CLIENTS))
            stale = _CLIENTS.pop(key)
            try:
                stale.close()
            except Exception:
                pass
        return client


def _commit_usage(hardware_module: Any, usage: dict[str, Any], elapsed: float) -> dict[str, float] | None:
    try:
        prompt = max(0, int(usage.get("prompt_tokens", 0) or 0))
        output = max(0, int(usage.get("completion_tokens", 0) or 0))
    except (TypeError, ValueError):
        return None
    if prompt == 0 and output == 0:
        return None
    with hardware_module._TELEMETRY_LOCK:
        totals = hardware_module._TELEMETRY_TOTALS
        totals["prompt_tokens"] += prompt
        totals["output_tokens"] += output
        totals["generation_seconds"] += max(0.0, float(elapsed))
        totals["requests"] += 1
        cumulative = dict(totals)
    return {
        "prompt_tokens": prompt,
        "output_tokens": output,
        "generation_seconds": max(0.0, float(elapsed)),
        "cumulative_prompt_tokens": int(cumulative["prompt_tokens"]),
        "cumulative_output_tokens": int(cumulative["output_tokens"]),
        "cumulative_generation_seconds": float(cumulative["generation_seconds"]),
        "cumulative_requests": int(cumulative["requests"]),
    }


def install(hardware_module: Any) -> None:
    """Use SSE-native usage and persistent HTTP connections on the llama hot path."""

    current = hardware_module._strict_server_generate
    if getattr(current, "_mmm_sse_usage_fast_path", False):
        return

    @wraps(current)
    def fast_stream_generate(adapter: Any, request: Any, server_url: str) -> str:
        # Retain the original exact native /metrics + /slots diagnostics as an explicit
        # opt-in. The default production path avoids those extra HTTP requests.
        if _env_bool("MMM_LLAMA_DETAILED_TELEMETRY", False):
            return current(adapter, request, server_url)

        from .model_adapters import ModelBackendError

        try:
            payload = hardware_module._server_payload(adapter, request)
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            endpoint = f"{server_url.rstrip('/')}/chat/completions"
            client = _client(server_url)

            pieces: list[str] = []
            generated_chars = 0
            reasoning_chars = 0
            output_events = 0
            request_started = time.monotonic()
            first_output_time: float | None = None
            last_progress_report = request_started
            first_output_reported = False
            saw_done = False
            final_usage: dict[str, Any] | None = None

            with client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    body = response.text.strip().replace("\n", " ")
                    if len(body) > 1200:
                        body = body[:1200] + "..."
                    raise RuntimeError(
                        f"llama server returned HTTP {response.status_code}"
                        + (f": {body}" if body else "")
                    )

                if getattr(adapter.__class__, "_reported_server_url", None) != server_url:
                    print("llama server: connected", server_url, flush=True)
                    adapter.__class__._reported_server_url = server_url
                structured = payload.get("response_format") is not None
                print(
                    "llama server: request accepted; streaming",
                    f" input_chars={hardware_module._request_content_chars(payload)}",
                    f" max_tokens={payload['max_tokens']}",
                    f" structured={'json' if structured else 'text'}",
                    f" reasoning={'disabled' if structured else 'model-default'}",
                    sep="",
                    flush=True,
                )

                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("llama server returned malformed SSE JSON") from exc
                    if not isinstance(chunk, dict):
                        continue
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        final_usage = usage
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    reasoning, content = hardware_module._stream_delta_parts(choice)
                    if reasoning:
                        reasoning_chars += len(reasoning)
                        output_events += 1
                    if content:
                        pieces.append(content)
                        generated_chars += len(content)
                        output_events += 1
                    if not reasoning and not content:
                        continue

                    now = time.monotonic()
                    if not first_output_reported:
                        first_output_time = now
                        print(
                            "llama server: first output delta",
                            f" elapsed={now - request_started:.1f}s",
                            f" kind={'content' if content else 'reasoning'}",
                            flush=True,
                        )
                        first_output_reported = True
                    if now - last_progress_report >= 15.0:
                        # No /slots request here: progress logging must not compete with
                        # the active decode for server work. Exact token usage arrives
                        # in the final SSE usage event.
                        print(
                            "llama server: progress",
                            f" content_chars={generated_chars}",
                            f" reasoning_chars={reasoning_chars}",
                            f" events={output_events}",
                            f" elapsed={now - request_started:.1f}s",
                            flush=True,
                        )
                        last_progress_report = now

            if not saw_done:
                raise RuntimeError("llama server stream ended before the [DONE] marker")
            content = "".join(pieces).strip()
            if not content:
                if reasoning_chars:
                    raise RuntimeError(
                        "llama server produced reasoning deltas but no visible content"
                    )
                raise RuntimeError("llama server stream produced no text content")

            ended = time.monotonic()
            elapsed = ended - request_started
            generation_elapsed = (
                ended - first_output_time if first_output_time is not None else elapsed
            )
            committed = (
                _commit_usage(hardware_module, final_usage, generation_elapsed)
                if final_usage is not None
                else None
            )
            if committed is not None:
                request_total = int(committed["prompt_tokens"]) + int(committed["output_tokens"])
                cumulative_total = int(committed["cumulative_prompt_tokens"]) + int(
                    committed["cumulative_output_tokens"]
                )
                request_tps = (
                    float(committed["output_tokens"]) / max(1e-9, generation_elapsed)
                    if int(committed["output_tokens"]) > 0
                    else 0.0
                )
                cumulative_tps = (
                    float(committed["cumulative_output_tokens"])
                    / max(1e-9, float(committed["cumulative_generation_seconds"]))
                    if int(committed["cumulative_output_tokens"]) > 0
                    else 0.0
                )
                print(
                    "llama server: generation complete",
                    f" prompt_tokens={int(committed['prompt_tokens'])}",
                    f" output_tokens={int(committed['output_tokens'])}",
                    f" request_tokens={request_total}",
                    f" tok_s={request_tps:.2f}",
                    f" cumulative_tokens={cumulative_total}",
                    f" cumulative_tok_s={cumulative_tps:.2f}",
                    f" elapsed={elapsed:.1f}s",
                    flush=True,
                )
            else:
                print(
                    "llama server: generation complete",
                    f" content_chars={generated_chars}",
                    f" elapsed={elapsed:.1f}s",
                    " token_telemetry=unavailable",
                    flush=True,
                )
            return content
        except Exception as exc:
            if isinstance(exc, ModelBackendError):
                raise
            raise ModelBackendError(
                role=adapter.config.role,
                model_id=adapter.config.model_id,
                cause=exc,
            ) from exc

    fast_stream_generate._mmm_sse_usage_fast_path = True  # type: ignore[attr-defined]
    hardware_module._strict_server_generate = fast_stream_generate


__all__ = ["install"]
