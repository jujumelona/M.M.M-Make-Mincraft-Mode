from __future__ import annotations

import json
import math
import os
import threading
import time
from functools import wraps
from typing import Any, Mapping


_CLIENT_LOCK = threading.RLock()
_CLIENTS: dict[str, Any] = {}
_CLIENT_LIMIT = 4
_REPORTED_URL_LOCK = threading.RLock()
_REPORTED_SERVER_URLS: set[str] = set()
_DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS = 300.0


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _stream_idle_timeout_seconds() -> float:
    raw = os.environ.get("MMM_LLAMA_STREAM_IDLE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_STREAM_IDLE_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "MMM_LLAMA_STREAM_IDLE_TIMEOUT_SECONDS must be a positive finite number."
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "MMM_LLAMA_STREAM_IDLE_TIMEOUT_SECONDS must be a positive finite number."
        )
    return value


def _append_message_delta(message: dict[str, Any], delta: Mapping[str, Any]) -> int:
    """Append one OpenAI-compatible SSE delta into an assistant message."""

    progressed = 0
    role = delta.get("role")
    if isinstance(role, str) and role:
        message["role"] = role
    for key in ("reasoning_content", "reasoning", "content"):
        value = delta.get(key)
        if not isinstance(value, str) or not value:
            continue
        previous = message.get(key)
        message[key] = (previous if isinstance(previous, str) else "") + value
        progressed += len(value)
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        # Managed llama-server is configured with --skip-chat-parsing, so normal
        # tool turns arrive as raw content. Preserve any unexpected parsed calls
        # anyway so the adapter's existing strict rejection remains effective.
        message["tool_calls"] = tool_calls
    return progressed


class _StreamingCompletionClient:
    """Keep the legacy response API while receiving chat completions over SSE.

    ``llama_cpp_adapter._post_completion`` historically used ``Client.post`` and
    therefore saw no readable response bytes until llama-server finished the whole
    turn. Its 600-second read timeout could consequently fire even while a long
    Qwen/MTP tool turn was actively decoding. This proxy streams that same request,
    aggregates deltas into the response shape expected by the adapter, and makes
    the read timeout an actual *idle* timeout rather than a whole-turn timeout.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def close(self) -> Any:
        return self._client.close()

    def stream(self, method: str, url: str, **kwargs: Any) -> Any:
        return self._client.stream(method, url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        payload = kwargs.get("json")
        if (
            not isinstance(payload, Mapping)
            or not url.rstrip("/").endswith("/chat/completions")
            or payload.get("stream") is True
        ):
            return self._client.post(url, **kwargs)

        import httpx

        streamed_payload = dict(payload)
        streamed_payload["stream"] = True
        streamed_payload["stream_options"] = {"include_usage": True}
        stream_kwargs = dict(kwargs)
        stream_kwargs["json"] = streamed_payload
        request = httpx.Request("POST", url)
        started = time.monotonic()
        last_report = started
        first_delta_reported = False
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        finish_reason: Any = None
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        saw_done = False
        progress_chars = 0

        with self._client.stream("POST", url, **stream_kwargs) as response:
            if response.status_code >= 400:
                body = response.read()
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=body,
                    request=request,
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
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                chunk_timings = chunk.get("timings")
                if isinstance(chunk_timings, dict):
                    timings = chunk_timings
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    continue
                current_finish = choice.get("finish_reason")
                if current_finish is not None:
                    finish_reason = current_finish
                delta = choice.get("delta")
                if not isinstance(delta, Mapping):
                    candidate = choice.get("message")
                    delta = candidate if isinstance(candidate, Mapping) else None
                if delta is None:
                    continue
                progressed = _append_message_delta(message, delta)
                if progressed <= 0:
                    continue
                progress_chars += progressed
                now = time.monotonic()
                if not first_delta_reported:
                    print(
                        "llama server: completion first output delta",
                        f" elapsed={now - started:.1f}s",
                        flush=True,
                    )
                    first_delta_reported = True
                if now - last_report >= 15.0:
                    print(
                        "llama server: completion stream progress",
                        f" chars={progress_chars}",
                        f" elapsed={now - started:.1f}s",
                        flush=True,
                    )
                    last_report = now

        if not saw_done:
            raise RuntimeError("llama server stream ended before the [DONE] marker")
        result: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
        }
        if usage is not None:
            result["usage"] = usage
        if timings is not None:
            result["timings"] = timings
        return httpx.Response(200, json=result, request=request)


def _client(server_url: str) -> Any:
    import httpx

    origin = server_url.rstrip("/")
    with _CLIENT_LOCK:
        client = _CLIENTS.get(origin)
        if client is not None:
            return client
        timeout = httpx.Timeout(
            connect=30.0,
            read=_stream_idle_timeout_seconds(),
            write=30.0,
            pool=30.0,
        )
        raw_client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=8,
                keepalive_expiry=60.0,
            ),
        )
        client = _StreamingCompletionClient(raw_client)
        _CLIENTS[origin] = client
        while len(_CLIENTS) > _CLIENT_LIMIT:
            key = next(iter(_CLIENTS))
            stale = _CLIENTS.pop(key)
            try:
                stale.close()
            except Exception:
                pass
        return client


def _report_server_connection(server_url: str) -> None:
    with _REPORTED_URL_LOCK:
        if server_url in _REPORTED_SERVER_URLS:
            return
        _REPORTED_SERVER_URLS.add(server_url)
    print("llama server: connected", server_url, flush=True)


def _native_timing_summary(timings: Any) -> dict[str, float | int] | None:
    """Normalize llama.cpp aggregate decode/speculative telemetry defensively."""

    if not isinstance(timings, dict):
        return None
    result: dict[str, float | int] = {}
    try:
        predicted_per_second = float(timings.get("predicted_per_second", 0.0) or 0.0)
    except (TypeError, ValueError):
        predicted_per_second = 0.0
    if predicted_per_second > 0.0:
        result["predicted_per_second"] = predicted_per_second

    try:
        draft_n = max(0, int(timings.get("draft_n", 0) or 0))
        accepted = max(0, int(timings.get("draft_n_accepted", 0) or 0))
    except (TypeError, ValueError):
        draft_n = 0
        accepted = 0
    if draft_n > 0:
        accepted = min(accepted, draft_n)
        result["draft_n"] = draft_n
        result["draft_n_accepted"] = accepted
        result["draft_acceptance_pct"] = 100.0 * accepted / draft_n

    return result or None


def _active_decode_profile() -> tuple[str, str, str]:
    spec = os.environ.get("MMM_LLAMA_ACTIVE_SPEC_TYPE", "none").strip() or "none"
    width = os.environ.get("MMM_LLAMA_ACTIVE_DRAFT_N_MAX", "0").strip() or "0"
    p_min = os.environ.get("MMM_LLAMA_ACTIVE_MTP_P_MIN", "0").strip() or "0"
    return spec, width, p_min


def _commit_usage(
    hardware_module: Any,
    usage: dict[str, Any],
    elapsed: float,
) -> dict[str, float] | None:
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
            final_timings: dict[str, Any] | None = None

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

                _report_server_connection(server_url)
                structured = getattr(request, "response_format", None) == "json"
                reasoning_disabled = payload.get("reasoning_effort") == "none"
                spec_type, draft_n_max, draft_p_min = _active_decode_profile()
                print(
                    "llama server: request accepted; streaming",
                    f" input_chars={hardware_module._request_content_chars(payload)}",
                    f" max_tokens={payload['max_tokens']}",
                    f" structured={'json-host-validated' if structured else 'text'}",
                    f" reasoning={'disabled' if reasoning_disabled else 'model-default'}",
                    f" spec={spec_type}",
                    f" n_max={draft_n_max}",
                    f" p_min={draft_p_min}",
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
                    timings = chunk.get("timings")
                    if isinstance(timings, dict):
                        final_timings = timings
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
            native = _native_timing_summary(final_timings)
            if committed is not None:
                request_total = int(committed["prompt_tokens"]) + int(
                    committed["output_tokens"]
                )
                cumulative_total = int(committed["cumulative_prompt_tokens"]) + int(
                    committed["cumulative_output_tokens"]
                )
                wall_tps = (
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
                native_tps = (
                    float(native["predicted_per_second"])
                    if native is not None and "predicted_per_second" in native
                    else wall_tps
                )
                telemetry_fields = [
                    "llama server: generation complete",
                    f" prompt_tokens={int(committed['prompt_tokens'])}",
                    f" output_tokens={int(committed['output_tokens'])}",
                    f" request_tokens={request_total}",
                    f" tok_s={native_tps:.2f}",
                    f" wall_tok_s={wall_tps:.2f}",
                    f" cumulative_tokens={cumulative_total}",
                    f" cumulative_tok_s={cumulative_tps:.2f}",
                ]
                if native is not None and "draft_n" in native:
                    telemetry_fields.extend(
                        [
                            f" mtp_accept={int(native['draft_n_accepted'])}/{int(native['draft_n'])}",
                            f" mtp_accept_pct={float(native['draft_acceptance_pct']):.1f}%",
                        ]
                    )
                elif native is None or "predicted_per_second" not in native:
                    telemetry_fields.append(" native_timing=unavailable")
                telemetry_fields.append(f" elapsed={elapsed:.1f}s")
                print(*telemetry_fields, sep="", flush=True)

                if (
                    spec_type == "draft-mtp"
                    and final_timings is not None
                    and (native is None or "draft_n" not in native)
                ):
                    print(
                        "llama server: warning MTP configured but native timings "
                        "reported no draft counters",
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

    fast_stream_generate._mmm_sse_usage_fast_path = True
    hardware_module._strict_server_generate = fast_stream_generate


__all__ = [
    "_active_decode_profile",
    "_native_timing_summary",
    "_stream_idle_timeout_seconds",
    "install",
]
