from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any


def _quick_file_signature(path: Path) -> str:
    """Hash only bounded head/tail samples; never scan a multi-GB GGUF for tuning."""

    size = path.stat().st_size
    sample = 1024 * 1024
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample))
        if size > sample:
            handle.seek(max(0, size - sample))
            digest.update(handle.read(sample))
    return digest.hexdigest()


def _drive_cache_from_setup_receipt() -> Path | None:
    raw = os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip()
    if not raw:
        return None
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not receipt.get("save_to_google_drive"):
        return None
    output_root = str(receipt.get("output_root", "")).strip()
    if not output_root:
        return None
    return (
        Path(output_root).expanduser().resolve()
        / ".mmm-cache"
        / "llama-server-autotune.json"
    )


def install(autotune_module: Any, hardware_policy_module: Any) -> None:
    """Install correctness-safe native llama-server efficiency policies."""

    probe = autotune_module._probe_server
    if getattr(probe, "_mmm_correctness_sentinel", False):
        # The compact deterministic benchmark is already an exact-output gate. A
        # second 64-token sentinel per candidate only burns decode time.
        underlying = getattr(probe, "__wrapped__", None)
        if underlying is not None:
            autotune_module._probe_server = underlying
            probe = underlying
    probe._mmm_compact_decode_probe = True  # type: ignore[attr-defined]

    current_payload = hardware_policy_module._server_payload
    if not getattr(current_payload, "_mmm_prompt_cache_reuse", False):

        @wraps(current_payload)
        def payload_with_prompt_cache(adapter: Any, request: Any) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            # Keep prefix-KV reuse explicit. Planner continuation/repair requests
            # commonly repeat the same system/contract prefix.
            payload["cache_prompt"] = True
            return payload

        payload_with_prompt_cache._mmm_prompt_cache_reuse = True  # type: ignore[attr-defined]
        hardware_policy_module._server_payload = payload_with_prompt_cache

    current_cache_path = autotune_module._cache_path
    if not getattr(current_cache_path, "_mmm_persistent_tuning_cache", False):

        @wraps(current_cache_path)
        def persistent_cache_path() -> Path:
            explicit = os.environ.get("MMM_LLAMA_AUTOTUNE_CACHE", "").strip()
            if explicit:
                return Path(explicit).expanduser().resolve()
            drive_cache = _drive_cache_from_setup_receipt()
            return drive_cache if drive_cache is not None else current_cache_path()

        persistent_cache_path._mmm_persistent_tuning_cache = True  # type: ignore[attr-defined]
        autotune_module._cache_path = persistent_cache_path

    current_fingerprint = autotune_module._fingerprint
    if not getattr(current_fingerprint, "_mmm_stable_model_signature", False):

        def stable_fingerprint(config: Any, binary: str, model_path: str) -> str:
            path = Path(model_path)
            payload = {
                "schema": autotune_module._BENCHMARK_SCHEMA_VERSION,
                "model_id": str(config.model_id),
                "gguf_filename": str(config.extra.get("gguf_filename", "")),
                "model_filename": path.name,
                "model_size": int(path.stat().st_size),
                "model_signature": _quick_file_signature(path),
                "max_context": int(config.max_context),
                "kv": os.environ.get("MMM_KV_CACHE_QUANT", "q4_0").lower(),
                "server": autotune_module._server_version(binary),
                "hardware": autotune_module._hardware_identity(),
                "batch": autotune_module._env_int("MMM_LLAMA_BATCH", 2048),
                "ubatch": autotune_module._env_int("MMM_LLAMA_UBATCH", 512),
                "probe_tokens": min(
                    int(config.max_new_tokens),
                    autotune_module._env_int(
                        "MMM_LLAMA_AUTOTUNE_TOKENS",
                        autotune_module._BENCHMARK_OUTPUT_TOKENS,
                    ),
                ),
                "variants": [
                    asdict(value) for value in autotune_module._candidate_variants()
                ],
            }
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            return hashlib.sha256(encoded).hexdigest()

        stable_fingerprint._mmm_stable_model_signature = True  # type: ignore[attr-defined]
        autotune_module._fingerprint = stable_fingerprint

    # Final native-server tuning is layered after the basic safety/telemetry policy so
    # it can benchmark the authoritative server args instead of creating a second
    # execution path.
    from . import llama_server_max_performance as max_performance_module
    from .llama_server_max_performance import install as install_max_performance

    install_max_performance(autotune_module)

    # llama-server exposes n_cache_reuse per request. Keep the same candidate search,
    # but run those candidates on one already-loaded server instead of reloading the
    # multi-GB model once per value.
    from .llama_cache_reuse_efficiency_contract import (
        install as install_cache_reuse_efficiency,
    )

    install_cache_reuse_efficiency(
        autotune_module,
        hardware_policy_module,
        max_performance_module,
    )

    # Successful production streams already carry exact prompt/completion usage.
    # Consume that SSE usage and reuse the local HTTP connection instead of issuing
    # /metrics before+after every request and /slots polls during active decode.
    from .llama_stream_efficiency_contract import install as install_stream_efficiency

    install_stream_efficiency(hardware_policy_module)

    # The server can only benefit from multiple slots when MMM is allowed to issue
    # concurrent requests. Share the resident llama-server GPU allocation between
    # those requests while keeping image/speech/other local GPU runtimes exclusive.
    from . import model_router as model_router_module
    from . import scheduler_parallel_safety_contract as scheduler_module
    from .llama_parallel_runtime_contract import install as install_parallel_runtime

    install_parallel_runtime(model_router_module, scheduler_module)
