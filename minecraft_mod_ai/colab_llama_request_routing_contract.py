from __future__ import annotations

import json
import os
import time
from functools import wraps
from typing import Any


def _stall_seconds() -> float:
    raw = os.environ.get("MMM_LLAMA_STREAM_STALL_SECONDS", "90").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 90.0
    return min(600.0, max(30.0, value))


def _install_local_stream_watchdog(hardware_policy_module: Any) -> None:
    current = hardware_policy_module._strict_server_generate
    if getattr(current, "_mmm_local_stream_watchdog", False):
        return

    @wraps(current)
    def guarded_stream(adapter: Any, request: Any, server_url: str) -> str:
        from .colab_mtp_server import (
            SERVER_API_URL,
            current_server_mode,
            server_log_tail,
        )

        if server_url.rstrip("/") != SERVER_API_URL.rstrip("/"):
            return current(adapter, request, server_url)

        from .model_adapters import ModelBackendError

        try:
            import httpx

            payload = hardware_policy_module._server_payload(adapter, request)
            payload["stream"] = True
            stall = _stall_seconds()
            timeout = httpx.Timeout(
                connect=30.0,
                read=stall,
                write=30.0,
                pool=30.0,
            )
            endpoint = f"{server_url.rstrip('/')}/chat/completions"
            pieces: list[str] = []
            generated_chars = 0
            reasoning_chars = 0
            output_events = 0
            request_started = time.monotonic()
            last_output = request_started
            last_progress_report = request_started
            first_output_reported = False
            saw_done = False

            with httpx.stream(
                "POST",
                endpoint,
                json=payload,
                timeout=timeout,
            ) as response:
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
                    f" mode={current_server_mode() or 'unknown'}",
                    f" input_chars={hardware_policy_module._request_content_chars(payload)}",
                    f" max_tokens={payload['max_tokens']}",
                    f" reasoning={'disabled' if structured else 'model-default'}",
                    sep="",
                    flush=True,
                )

                for raw_line in response.iter_lines():
                    now = time.monotonic()
                    if now - last_output >= stall:
                        raise RuntimeError(
                            f"llama server produced no output delta for {stall:.0f}s"
                        )
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
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
                        raise RuntimeError(
                            "llama server returned malformed SSE JSON."
                        ) from exc
                    choices = chunk.get("choices") if isinstance(chunk, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    reasoning, content = hardware_policy_module._stream_delta_parts(choice)
                    if reasoning:
                        reasoning_chars += len(reasoning)
                        output_events += 1
                    if content:
                        pieces.append(content)
                        generated_chars += len(content)
                        output_events += 1
                    if not reasoning and not content:
                        continue

                    last_output = now
                    if not first_output_reported:
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
                            f" mode={current_server_mode() or 'unknown'}",
                            f" content_chars={generated_chars}",
                            f" reasoning_chars={reasoning_chars}",
                            f" events={output_events}",
                            flush=True,
                        )
                        last_progress_report = now

            if not saw_done:
                raise RuntimeError("llama server stream ended before the [DONE] marker.")
            content = "".join(pieces).strip()
            if not content:
                if reasoning_chars:
                    raise RuntimeError(
                        "llama server produced reasoning deltas but no visible content; "
                        "the chat template/output grammar channels are incompatible."
                    )
                raise RuntimeError("llama server stream produced no text content.")
            print(
                "llama server: generation complete",
                f" mode={current_server_mode() or 'unknown'}",
                f" content_chars={generated_chars}",
                f" reasoning_chars={reasoning_chars}",
                f" elapsed={time.monotonic() - request_started:.1f}s",
                sep="",
                flush=True,
            )
            return content
        except Exception as exc:
            if isinstance(exc, ModelBackendError):
                raise
            tail = server_log_tail(120)
            detail = f"{type(exc).__name__}: {exc}"
            if tail:
                detail += "\nmanaged llama server log tail:\n" + tail
            raise ModelBackendError(
                role=adapter.config.role,
                model_id=adapter.config.model_id,
                cause=RuntimeError(detail),
            ) from exc

    guarded_stream._mmm_local_stream_watchdog = True
    hardware_policy_module._strict_server_generate = guarded_stream


def _install_request_mode_router(hardware_policy_module: Any) -> None:
    from .model_adapters import ModelBackendError
    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    current = LlamaCppAdapter.generate
    if getattr(current, "_mmm_colab_request_mode_router", False):
        return

    @wraps(current)
    def routed_generate(self: Any, request: Any) -> str:
        from .colab_mtp_server import (
            colab_mtp_server_enabled,
            current_server_mode,
            ensure_colab_server_for_request,
            mark_mtp_unhealthy,
            start_colab_mtp_server,
            stop_colab_mtp_server,
        )

        if not colab_mtp_server_enabled():
            return current(self, request)

        ensure_colab_server_for_request(self.config, request)
        selected_mode = current_server_mode()
        try:
            return current(self, request)
        except ModelBackendError as exc:
            if selected_mode != "mtp":
                raise
            mark_mtp_unhealthy(f"request failed in MTP mode: {exc}")
            stop_colab_mtp_server(keep_enabled=True)
            start_colab_mtp_server(self.config, mode="baseline")
            print(
                "llama server: retrying failed MTP request in baseline mode",
                flush=True,
            )
            return current(self, request)

    routed_generate._mmm_colab_request_mode_router = True
    LlamaCppAdapter.generate = routed_generate


def install(hardware_policy_module: Any) -> None:
    _install_local_stream_watchdog(hardware_policy_module)
    _install_request_mode_router(hardware_policy_module)


__all__ = ["install"]
