from __future__ import annotations

import json
import time
from functools import wraps
from typing import Any


_SERVER_KEY: tuple[Any, ...] | None = None


def _server_key(server_module: Any, config: Any, mode: str) -> tuple[Any, ...]:
    effective_mode = mode
    if effective_mode == "mtp" and not server_module._mtp_capable(config):
        effective_mode = "baseline"
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", ""))
        if isinstance(extra, dict)
        else ""
    )
    return (
        str(getattr(config, "model_id", "")),
        filename,
        min(int(getattr(config, "max_context", 0) or 0), server_module.SERVER_CONTEXT_CAP),
        int(getattr(config, "max_new_tokens", 0) or 0),
        effective_mode,
        server_module._mtp_width() if effective_mode == "mtp" else 0,
        server_module._kv_cache_quant(),
    )


def _install_exact_mtp_probe(server_module: Any) -> None:
    current = server_module._probe_mtp_server
    if getattr(current, "_mmm_exact_stream_probe", False):
        return

    def exact_probe() -> tuple[bool, str]:
        expected = " ".join(str(value) for value in range(1, 21))
        payload = {
            "model": "local",
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the user literally. Do not explain.",
                },
                {
                    "role": "user",
                    "content": (
                        "Output the integers 1 through 20 in order, separated by one "
                        "space, and nothing else."
                    ),
                },
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "reasoning_effort": "none",
            "stream": True,
        }
        started = time.monotonic()
        output_parts: list[str] = []
        output_events = 0
        saw_done = False
        try:
            timeout = server_module._mtp_probe_timeout()
            with server_module.httpx.stream(
                "POST",
                f"{server_module.SERVER_API_URL}/chat/completions",
                json=payload,
                timeout=server_module.httpx.Timeout(
                    connect=min(30.0, timeout),
                    read=timeout,
                    write=min(30.0, timeout),
                    pool=min(30.0, timeout),
                ),
            ) as response:
                if response.status_code != 200:
                    response.read()
                    return False, f"HTTP {response.status_code}"
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    if not data:
                        continue
                    chunk = json.loads(data)
                    choices = chunk.get("choices") if isinstance(chunk, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    emitted = "".join(
                        part
                        for part in (
                            delta.get("reasoning_content"),
                            delta.get("reasoning"),
                            delta.get("content"),
                        )
                        if isinstance(part, str)
                    )
                    if emitted:
                        output_parts.append(emitted)
                        output_events += 1

            elapsed = time.monotonic() - started
            if not saw_done:
                return False, f"stream ended before [DONE] after {elapsed:.1f}s"
            rendered = " ".join("".join(output_parts).strip().split())
            if output_events < 2:
                return False, (
                    "probe did not demonstrate multi-step streaming; "
                    f"events={output_events} elapsed={elapsed:.1f}s"
                )
            if rendered != expected:
                preview = rendered[:160]
                return False, (
                    "MTP correctness mismatch for deterministic probe; "
                    f"output={preview!r} events={output_events} elapsed={elapsed:.1f}s"
                )
            return True, (
                f"exact_output_match events={output_events} elapsed={elapsed:.1f}s"
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            return False, f"{type(exc).__name__}: {exc} after {elapsed:.1f}s"

    exact_probe._mmm_exact_stream_probe = True  # type: ignore[attr-defined]
    server_module._probe_mtp_server = exact_probe


def install(server_module: Any) -> None:
    """Bind managed server reuse and MTP enablement to verified decode behavior."""

    _install_exact_mtp_probe(server_module)

    current = server_module.start_colab_mtp_server
    if getattr(current, "_mmm_server_config_bound", False):
        return

    @wraps(current)
    def start_bound(config: Any, *, mode: str = "baseline") -> str:
        global _SERVER_KEY

        desired_key = _server_key(server_module, config, mode)
        if server_module.colab_mtp_server_running() and _SERVER_KEY != desired_key:
            server_module.stop_colab_mtp_server(keep_enabled=True)
            print(
                "llama server: decode configuration changed; restarting",
                f" mode={desired_key[4]}",
                f" kv_cache={desired_key[6]}",
                flush=True,
            )

        url = current(config, mode=mode)
        _SERVER_KEY = _server_key(
            server_module,
            config,
            server_module.current_server_mode() or mode,
        )
        return url

    start_bound._mmm_server_config_bound = True  # type: ignore[attr-defined]
    server_module.start_colab_mtp_server = start_bound


__all__ = ["install"]
