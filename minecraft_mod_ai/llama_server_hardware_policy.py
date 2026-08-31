from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_TOTALS = {
    "prompt_tokens": 0,
    "prompt_seconds": 0.0,
    "output_tokens": 0,
    "generation_seconds": 0.0,
    "requests": 0,
}


def _existing_built_server() -> str | None:
    explicit_binary = os.environ.get("MMM_LLAMA_SERVER_BIN", "").strip()
    explicit_source = os.environ.get("MMM_LLAMA_SERVER_SOURCE_DIR", "").strip()
    candidates: list[Path] = []
    if explicit_binary:
        candidates.append(Path(explicit_binary).expanduser())
    discovered = shutil.which("llama-server")
    if discovered:
        candidates.append(Path(discovered))
    if explicit_source:
        candidates.append(
            Path(explicit_source).expanduser() / "build" / "bin" / "llama-server"
        )
    candidates.extend(
        [
            Path("/content/llama.cpp/build/bin/llama-server"),
            Path.home()
            / ".cache"
            / "mmm"
            / "llama.cpp"
            / "build"
            / "bin"
            / "llama-server",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
            os.environ["MMM_LLAMA_SERVER_BIN"] = resolved
            return resolved
    return None


def _bootstrap_native_server() -> str | None:
    """Return the source-owned native llama-server installed during setup."""

    return _existing_built_server()


def _named_tool_choice_name(request: Any) -> str:
    choice = getattr(request, "tool_choice", None)
    if not isinstance(choice, Mapping):
        return ""
    function = choice.get("function")
    if not isinstance(function, Mapping):
        raise TypeError("named tool_choice requires function metadata")
    name = str(function.get("name", "")).strip()
    if not name:
        raise ValueError("named tool_choice requires a function name")
    return name


def _required_tool_name(request: Any) -> str:
    named = _named_tool_choice_name(request)
    if named:
        return named
    return ""


def _server_tool_choice(request: Any) -> str:
    """Map host semantics onto llama.cpp's string-only Jinja tool contract."""

    if _required_tool_name(request):
        return "required"

    choice = getattr(request, "tool_choice", None)
    normalized = str(choice or "auto").strip().casefold()
    if normalized not in {"auto", "required", "none"}:
        raise ValueError(f"unsupported llama-server tool_choice: {choice!r}")
    return normalized


def _enforce_required_tool_sampling(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep forced one-tool turns deterministic and non-thinking on the wire."""

    if payload.get("tool_choice") != "required":
        return payload
    payload["temperature"] = 0.0
    for key in (
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repeat_penalty",
        "repetition_penalty",
    ):
        payload.pop(key, None)
    payload["reasoning_effort"] = "none"
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _server_payload(adapter: Any, request: Any) -> dict[str, Any]:
    """Build the base OpenAI-compatible llama-server chat payload.

    Tool-capable turns keep the model's native Jinja tools prompt and host-validated raw
    markup. Final finite token budgets and structured JSON-schema constraints are added
    by the single llama generation payload contract during runtime bootstrap.
    """

    payload: dict[str, Any] = {
        "model": "local",
        "messages": [dict(message) for message in request.messages],
        "max_tokens": int(adapter.config.max_new_tokens),
        "temperature": 0.0,
    }
    tools = getattr(request, "tools", ()) or ()
    if tools:
        required_name = _required_tool_name(request)
        visible_tools = tuple(tools)
        if required_name:
            visible_tools = tuple(
                tool
                for tool in tools
                if isinstance(tool, Mapping)
                and isinstance(tool.get("function"), Mapping)
                and str(tool["function"].get("name", "")).strip() == required_name
            )
            if len(visible_tools) != 1:
                raise ValueError(
                    f"required tool {required_name!r} must match exactly one schema"
                )
        payload["tools"] = [dict(tool) for tool in visible_tools]
        payload["tool_choice"] = _server_tool_choice(request)
        payload["parallel_tool_calls"] = bool(
            getattr(request, "parallel_tool_calls", False)
        )
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    elif getattr(request, "response_format", None) == "json":
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return _enforce_required_tool_sampling(payload)


def _stream_delta_parts(choice: dict[str, Any]) -> tuple[str, str]:
    reasoning = ""
    content = ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        raw_reasoning = delta.get("reasoning_content")
        if raw_reasoning is None:
            raw_reasoning = delta.get("thinking")
        if isinstance(raw_reasoning, str):
            reasoning = raw_reasoning
        raw_content = delta.get("content")
        if isinstance(raw_content, str):
            content = raw_content
    if not content:
        raw_text = choice.get("text")
        if isinstance(raw_text, str):
            content = raw_text
    return reasoning, content


def _request_content_chars(payload: dict[str, Any]) -> int:
    total = 0
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        value = message.get("content")
        if isinstance(value, str):
            total += len(value)
        elif value is not None:
            try:
                total += len(json.dumps(value, ensure_ascii=False))
            except (TypeError, ValueError, RecursionError):
                # Telemetry sizing must never make inference fail on an exotic payload.
                total += len(str(value))
    return total


def _server_origin(server_url: str) -> str:
    value = server_url.rstrip("/")
    return value.removesuffix("/v1")


def _auxiliary_native_telemetry_enabled() -> bool:
    """Keep auxiliary /metrics and /slots requests off the default inference path."""

    return os.environ.get("MMM_LLAMA_AUXILIARY_TELEMETRY", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    prefix = "llamacpp:"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        fields = line.split()
        if len(fields) != 2 or "{" in fields[0]:
            continue
        try:
            values[fields[0][len(prefix) :]] = float(fields[1])
        except ValueError:
            continue
    return values


def _metrics_snapshot(httpx_module: Any, server_url: str) -> dict[str, float] | None:
    try:
        response = httpx_module.get(
            f"{_server_origin(server_url)}/metrics",
            timeout=0.75,
        )
        if response.status_code != 200:
            return None
        values = _parse_prometheus_metrics(response.text)
        required = {"prompt_tokens_total", "tokens_predicted_total"}
        return values if required.issubset(values) else None
    except Exception as exc:  # noqa: BLE001 - optional metrics endpoint boundary
        print(
            "llama server: metrics snapshot unavailable",
            f" error={type(exc).__name__}",
            flush=True,
        )
        return None


def _slot_snapshot(httpx_module: Any, server_url: str) -> dict[str, int] | None:
    """Read current native slot counters; no prompt/generated text is requested."""

    try:
        response = httpx_module.get(
            f"{_server_origin(server_url)}/slots",
            timeout=0.75,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, list):
            return None
        active = next(
            (
                value
                for value in payload
                if isinstance(value, dict) and bool(value.get("is_processing"))
            ),
            None,
        )
        if active is None:
            return None
        next_token = active.get("next_token")
        if not isinstance(next_token, dict):
            next_token = {}
        return {
            "prompt_tokens": max(0, int(active.get("n_prompt_tokens", 0) or 0)),
            "prompt_processed": max(
                0, int(active.get("n_prompt_tokens_processed", 0) or 0)
            ),
            "prompt_cached": max(
                0, int(active.get("n_prompt_tokens_cache", 0) or 0)
            ),
            "output_tokens": max(0, int(next_token.get("n_decoded", 0) or 0)),
        }
    except Exception as exc:  # noqa: BLE001 - optional slot endpoint boundary
        print(
            "llama server: slot snapshot unavailable",
            f" error={type(exc).__name__}",
            flush=True,
        )
        return None


def _telemetry_totals() -> dict[str, float]:
    with _TELEMETRY_LOCK:
        return dict(_TELEMETRY_TOTALS)


def _commit_metrics_delta(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    prompt = max(
        0,
        int(after.get("prompt_tokens_total", 0))
        - int(before.get("prompt_tokens_total", 0)),
    )
    output = max(
        0,
        int(after.get("tokens_predicted_total", 0))
        - int(before.get("tokens_predicted_total", 0)),
    )
    prompt_seconds = max(
        0.0,
        float(after.get("prompt_seconds_total", 0.0))
        - float(before.get("prompt_seconds_total", 0.0)),
    )
    generation_seconds = max(
        0.0,
        float(after.get("tokens_predicted_seconds_total", 0.0))
        - float(before.get("tokens_predicted_seconds_total", 0.0)),
    )
    with _TELEMETRY_LOCK:
        _TELEMETRY_TOTALS["prompt_tokens"] += prompt
        _TELEMETRY_TOTALS["prompt_seconds"] = (
            float(_TELEMETRY_TOTALS.get("prompt_seconds", 0.0)) + prompt_seconds
        )
        _TELEMETRY_TOTALS["output_tokens"] += output
        _TELEMETRY_TOTALS["generation_seconds"] += generation_seconds
        _TELEMETRY_TOTALS["requests"] += 1
        cumulative = dict(_TELEMETRY_TOTALS)
    prompt_tps = prompt / prompt_seconds if prompt_seconds > 0 else 0.0
    cumulative_prompt_seconds = float(cumulative["prompt_seconds"])
    cumulative_prompt_tps = (
        float(cumulative["prompt_tokens"]) / cumulative_prompt_seconds
        if cumulative_prompt_seconds > 0
        else 0.0
    )
    result = {
        "prompt_tokens": prompt,
        "prompt_seconds": prompt_seconds,
        "prompt_tps": prompt_tps,
        "output_tokens": output,
        "generation_seconds": generation_seconds,
        "cumulative_prompt_tokens": int(cumulative["prompt_tokens"]),
        "cumulative_prompt_seconds": cumulative_prompt_seconds,
        "cumulative_prompt_tps": cumulative_prompt_tps,
        "cumulative_output_tokens": int(cumulative["output_tokens"]),
        "cumulative_generation_seconds": float(cumulative["generation_seconds"]),
        "cumulative_requests": int(cumulative["requests"]),
    }
    print(
        "llama server: prefill complete",
        f" prompt_tokens={prompt}",
        f" prompt_seconds={prompt_seconds:.3f}",
        f" prompt_tok_s={prompt_tps:.2f}",
        f" cumulative_prompt_tok_s={cumulative_prompt_tps:.2f}",
        sep="",
        flush=True,
    )
    return result


def _reject_tool_stream_request(adapter: Any, request: Any) -> None:
    """Keep tool semantics out of the text-only streaming transport."""

    if not (getattr(request, "tools", ()) or ()):
        return
    from .model_adapters import ModelBackendError

    raise ModelBackendError(
        role=adapter.config.role,
        model_id=adapter.config.model_id,
        cause=(
            "A tool-aware completion reached the text-only streaming API. "
            "Use ModelRouter.generate_text() or LlamaCppAdapter.generate_turn() so "
            "Qwen tool calls are parsed and executed by the host."
        ),
    )


def _strict_server_generate(adapter: Any, request: Any, server_url: str) -> str:
    """Stream one native text-only server turn with native token telemetry."""

    from .llama_stream_efficiency_contract import _client, _stream_idle_timeout_seconds
    from .model_adapters import ModelBackendError

    _reject_tool_stream_request(adapter, request)
    auxiliary_telemetry = _auxiliary_native_telemetry_enabled()
    metrics_before: dict[str, float] | None = None
    metrics_committed = False
    client: Any | None = None
    try:
        import httpx

        client = _client(server_url)
        payload = _server_payload(adapter, request)
        payload["stream"] = True
        timeout = httpx.Timeout(
            connect=30.0,
            read=_stream_idle_timeout_seconds(),
            write=30.0,
            pool=30.0,
        )
        endpoint = f"{server_url.rstrip('/')}/chat/completions"
        pieces: list[str] = []
        generated_chars = 0
        reasoning_chars = 0
        output_events = 0
        request_started = time.monotonic()
        last_progress_report = request_started
        first_output_time: float | None = None
        last_token_sample_time: float | None = None
        last_token_sample_count = 0
        first_output_reported = False
        saw_done = False
        last_slot: dict[str, int] | None = None
        if auxiliary_telemetry:
            metrics_before = _metrics_snapshot(client, server_url)
        committed_at_start = _telemetry_totals()

        with client.stream("POST", endpoint, json=payload, timeout=timeout) as response:
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
            structured = getattr(request, "response_format", None) == "json"
            print(
                "llama server: request accepted; streaming",
                f" input_chars={_request_content_chars(payload)}",
                f" max_tokens={payload['max_tokens']}",
                f" structured={'json' if structured else 'text'}",
                f" reasoning={'disabled' if payload.get('reasoning_effort') == 'none' else 'model-default'}",
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
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                reasoning, content = _stream_delta_parts(choice)
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
                    slot = (
                        _slot_snapshot(client, server_url)
                        if auxiliary_telemetry
                        else None
                    )
                    if slot is not None:
                        last_slot = slot
                        output_tokens = slot["output_tokens"]
                        sample_base_time = last_token_sample_time or first_output_time or request_started
                        sample_base_count = (
                            last_token_sample_count if last_token_sample_time is not None else 0
                        )
                        sample_seconds = max(1e-9, now - sample_base_time)
                        current_tps = max(0.0, output_tokens - sample_base_count) / sample_seconds
                        last_token_sample_time = now
                        last_token_sample_count = output_tokens
                        request_total = slot["prompt_tokens"] + output_tokens
                        cumulative_total = (
                            int(committed_at_start["prompt_tokens"])
                            + int(committed_at_start["output_tokens"])
                            + slot["prompt_processed"]
                            + output_tokens
                        )
                        print(
                            "llama server: progress",
                            f" prompt_tokens={slot['prompt_tokens']}",
                            f" prompt_processed={slot['prompt_processed']}",
                            f" prompt_cached={slot['prompt_cached']}",
                            f" output_tokens={output_tokens}",
                            f" request_tokens={request_total}",
                            f" tok_s={current_tps:.2f}",
                            f" cumulative_tokens={cumulative_total}",
                            f" elapsed={now - request_started:.1f}s",
                            flush=True,
                        )
                    else:
                        print(
                            "llama server: progress",
                            f" content_chars={generated_chars}",
                            f" reasoning_chars={reasoning_chars}",
                            f" events={output_events}",
                            f" elapsed={now - request_started:.1f}s",
                            " token_telemetry=unavailable",
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

        metrics_after = (
            _metrics_snapshot(client, server_url)
            if metrics_before is not None
            else None
        )
        usage = _commit_metrics_delta(metrics_before, metrics_after)
        metrics_committed = usage is not None
        elapsed = time.monotonic() - request_started
        if usage is not None:
            request_total = int(usage["prompt_tokens"]) + int(usage["output_tokens"])
            cumulative_total = int(usage["cumulative_prompt_tokens"]) + int(
                usage["cumulative_output_tokens"]
            )
            generation_seconds = float(usage["generation_seconds"])
            request_tps = (
                float(usage["output_tokens"]) / generation_seconds
                if generation_seconds > 0
                else 0.0
            )
            cumulative_generation_seconds = float(
                usage["cumulative_generation_seconds"]
            )
            cumulative_tps = (
                float(usage["cumulative_output_tokens"]) / cumulative_generation_seconds
                if cumulative_generation_seconds > 0
                else 0.0
            )
            print(
                "llama server: generation complete",
                f" prompt_tokens={int(usage['prompt_tokens'])}",
                f" output_tokens={int(usage['output_tokens'])}",
                f" request_tokens={request_total}",
                f" tok_s={request_tps:.2f}",
                f" cumulative_prompt_tokens={int(usage['cumulative_prompt_tokens'])}",
                f" cumulative_output_tokens={int(usage['cumulative_output_tokens'])}",
                f" cumulative_tokens={cumulative_total}",
                f" cumulative_tok_s={cumulative_tps:.2f}",
                f" elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            fallback = last_slot or {}
            print(
                "llama server: generation complete",
                f" content_chars={generated_chars}",
                f" output_tokens={fallback.get('output_tokens', 'unknown')}",
                f" elapsed={elapsed:.1f}s",
                " token_telemetry=unavailable",
                flush=True,
            )
        return content
    except Exception as exc:
        if not metrics_committed and client is not None and metrics_before is not None:
            try:
                metrics_after = _metrics_snapshot(client, server_url)
                _commit_metrics_delta(metrics_before, metrics_after)
            except Exception as telemetry_exc:  # noqa: BLE001 - best-effort failure telemetry
                print(
                    "llama server: failure telemetry unavailable",
                    f" error={type(telemetry_exc).__name__}",
                    flush=True,
                )
        if isinstance(exc, ModelBackendError):
            raise
        raise ModelBackendError(
            role=adapter.config.role,
            model_id=adapter.config.model_id,
            cause=exc,
        ) from exc


def _apply_hardware_launch_policy(args: list[str]) -> list[str]:
    """Apply managed llama-server launch policy without enabling unused endpoints."""

    try:
        index = args.index("--gpu-layers")
        args[index + 1] = "auto"
    except (ValueError, IndexError):
        pass
    if "--parallel" not in args and "-np" not in args:
        args.extend(["--parallel", "1"])
    if _auxiliary_native_telemetry_enabled():
        if "--metrics" not in args:
            args.append("--metrics")
        if "--slots" not in args:
            args.append("--slots")
    return args


def install(autotune_module: Any) -> None:
    """Bind local GGUF inference exclusively to managed native llama-server."""

    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    original_server_binary = autotune_module._server_binary
    if not getattr(original_server_binary, "_mmm_native_bootstrap", False):

        @wraps(original_server_binary)
        def bootstrapped_server_binary() -> str | None:
            return original_server_binary() or _bootstrap_native_server()

        bootstrapped_server_binary._mmm_native_bootstrap = True  # type: ignore[attr-defined]
        autotune_module._server_binary = bootstrapped_server_binary

    original_base = autotune_module._base_args
    if not getattr(original_base, "_mmm_auto_gpu_layers", False):

        @wraps(original_base)
        def adaptive_base_args(
            binary: str,
            model_path: str,
            config: Any,
            port: int,
        ) -> list[str]:
            args = original_base(binary, model_path, config, port)
            return _apply_hardware_launch_policy(args)

        adaptive_base_args._mmm_auto_gpu_layers = True  # type: ignore[attr-defined]
        adaptive_base_args._mmm_single_decode_slot = True  # type: ignore[attr-defined]
        adaptive_base_args._mmm_auxiliary_telemetry_opt_in = True  # type: ignore[attr-defined]
        autotune_module._base_args = adaptive_base_args

    original_variant = autotune_module._variant_args
    if not getattr(original_variant, "_mmm_auto_draft_layers", False):

        @wraps(original_variant)
        def adaptive_variant_args(variant: Any) -> list[str]:
            args = original_variant(variant)
            try:
                index = args.index("--spec-draft-ngl")
                args[index + 1] = "auto"
            except (ValueError, IndexError):
                pass
            return args

        adaptive_variant_args._mmm_auto_draft_layers = True  # type: ignore[attr-defined]
        autotune_module._variant_args = adaptive_variant_args

    original_probe = autotune_module._probe_server
    if not getattr(original_probe, "_mmm_correctness_sentinel", False):

        @wraps(original_probe)
        def guarded_probe(
            base_url: str,
            request: Any,
            *,
            max_tokens: int,
            variant: Any,
        ) -> Any:
            measured = original_probe(
                base_url,
                request,
                max_tokens=max_tokens,
                variant=variant,
            )
            if max_tokens <= 1 or not measured.ok:
                return measured
            sentinel_request = SimpleNamespace(
                messages=(
                    {
                        "role": "system",
                        "content": (
                            "You are a deterministic Java 17 code generator. Output "
                            "only valid Java code and no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Write exactly one static int clamp(int value, int min, "
                            "int max) method using Math.min and Math.max."
                        ),
                    },
                ),
                response_format="text",
            )
            sentinel = original_probe(
                base_url,
                sentinel_request,
                max_tokens=min(max_tokens, 64),
                variant=variant,
            )
            combined = hashlib.sha256(
                f"{measured.output_sha256}:{sentinel.output_sha256}".encode()
            ).hexdigest()
            return autotune_module.ProbeResult(
                variant=measured.variant,
                ok=bool(measured.ok and sentinel.ok),
                output_sha256=combined,
                predicted_tokens=measured.predicted_tokens,
                predicted_tps=measured.predicted_tps,
                prompt_tps=measured.prompt_tps,
                elapsed_seconds=measured.elapsed_seconds,
                error=(
                    measured.error
                    if sentinel.ok
                    else "; ".join(
                        value
                        for value in (
                            measured.error,
                            f"sentinel: {sentinel.error}",
                        )
                        if value
                    )
                ),
            )

        guarded_probe._mmm_correctness_sentinel = True  # type: ignore[attr-defined]
        autotune_module._probe_server = guarded_probe

    current_generate = LlamaCppAdapter.generate
    if not getattr(current_generate, "_mmm_explicit_server_strict", False):

        @wraps(current_generate)
        def strict_selected_server_generate(self: Any, request: Any) -> str:
            _reject_tool_stream_request(self, request)
            explicit = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
            if not explicit:
                explicit = (
                    autotune_module.ensure_tuned_server(self.config, request) or ""
                ).strip().rstrip("/")
            if not explicit:
                raise RuntimeError(
                    "native llama-server is required but could not be started"
                )
            return _strict_server_generate(self, request, explicit)

        strict_selected_server_generate._mmm_explicit_server_strict = True  # type: ignore[attr-defined]
        LlamaCppAdapter.generate = strict_selected_server_generate

    from . import complete_orchestrator_services as services

    original_assets = services.generate_assets
    if not getattr(original_assets, "_mmm_releases_managed_llama", False):

        @wraps(original_assets)
        def assets_with_llama_release(router: Any, *args: Any, **kwargs: Any):
            registry = getattr(router, "registry", None)
            profile = getattr(router, "profile", None)
            local_exclusive_image = False
            if registry is not None and profile is not None:
                try:
                    config = registry.role(profile, "image_generator")
                    local_exclusive_image = (
                        config.provider == "local"
                        and config.adapter == "image_diffusion"
                        and config.exclusive_gpu
                    )
                except Exception as exc:  # noqa: BLE001 - optional registry lookup boundary
                    print(
                        "llama server: image role lookup unavailable",
                        f" error={type(exc).__name__}",
                        flush=True,
                    )
                    local_exclusive_image = False
            if local_exclusive_image:
                process = getattr(autotune_module, "_MANAGED_PROCESS", None)
                if process is not None and process.poll() is None:
                    managed_url = getattr(autotune_module, "_MANAGED_URL", None)
                    autotune_module._shutdown_managed_server()
                    if managed_url and os.environ.get("LLAMA_SERVER_URL") == managed_url:
                        os.environ.pop("LLAMA_SERVER_URL", None)
                    autotune_module._ATTEMPTED_KEYS.clear()
            return original_assets(router, *args, **kwargs)

        assets_with_llama_release._mmm_releases_managed_llama = True  # type: ignore[attr-defined]
        services.generate_assets = assets_with_llama_release


__all__ = [
    "_metrics_snapshot",
    "_parse_prometheus_metrics",
    "_request_content_chars",
    "_server_payload",
    "_slot_snapshot",
    "_stream_delta_parts",
    "_strict_server_generate",
    "_telemetry_totals",
    "install",
]
