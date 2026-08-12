from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any


def _submodule(name: str) -> Any:
    package = __package__ or "minecraft_mod_ai"
    return importlib.import_module(f"{package}.{name}")


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

    # Runtime tuning changes only llama-server startup/request parameters. CUDA Graphs
    # are built into the verified native bundle/setup build, so package import never
    # invokes cmake or recompiles native code. Import siblings directly instead of
    # resolving them as attributes on a partially initialized package __init__.
    runtime_tuning_module = _submodule("llama_server_runtime_tuning")
    runtime_tuning_module.install(autotune_module)

    # n_cache_reuse is request-scoped. Probe all reuse widths on one selected server
    # instead of reloading the multi-GB GGUF for each candidate.
    cache_reuse_module = _submodule("llama_cache_reuse_efficiency_contract")
    cache_reuse_module.install(
        autotune_module,
        hardware_policy_module,
        runtime_tuning_module,
    )

    # Consume usage from the production SSE stream and reuse the HTTP connection;
    # detailed /metrics and /slots telemetry remains an explicit diagnostic opt-in.
    stream_module = _submodule("llama_stream_efficiency_contract")
    stream_module.install(hardware_policy_module)
