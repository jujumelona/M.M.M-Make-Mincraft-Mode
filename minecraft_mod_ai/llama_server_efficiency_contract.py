from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

from .runtime_contract_wrappers import owns_contract_marker

_DECISION_STORE_SCHEMA = "mmm/llama-server-autotune-store-v1"
_DECISION_STORE_LIMIT = 32
_CACHE_LOCK = threading.RLock()
_MODEL_SIGNATURE_CACHE: dict[tuple[str, int, int], str] = {}
_MODEL_PATH_CACHE: dict[tuple[str, str], str] = {}
_SERVER_VERSION_CACHE: dict[tuple[str, int, int], str] = {}
_HARDWARE_IDENTITY_CACHE: str | None = None


def _bounded_put(mapping: dict[Any, Any], key: Any, value: Any, *, limit: int) -> None:
    mapping[key] = value
    while len(mapping) > limit:
        mapping.pop(next(iter(mapping)))


def _quick_file_signature(path: Path) -> str:
    """Hash bounded head/tail samples once per unchanged GGUF in this process."""
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _CACHE_LOCK:
        cached = _MODEL_SIGNATURE_CACHE.get(key)
        if cached is not None:
            return cached

    size = int(stat.st_size)
    sample = 1024 * 1024
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with resolved.open("rb") as handle:
        digest.update(handle.read(sample))
        if size > sample:
            handle.seek(max(0, size - sample))
            digest.update(handle.read(sample))
    value = digest.hexdigest()
    with _CACHE_LOCK:
        _bounded_put(_MODEL_SIGNATURE_CACHE, key, value, limit=16)
    return value


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


def _read_decision_entries(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    if payload.get("store_schema") == _DECISION_STORE_SCHEMA:
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            return {}
        return {
            str(key): dict(value)
            for key, value in raw_entries.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    # Backward-compatible migration from the historical single-decision cache.
    fingerprint = payload.get("fingerprint")
    selected = payload.get("selected")
    if isinstance(fingerprint, str) and fingerprint and isinstance(selected, dict):
        return {fingerprint: dict(payload)}
    return {}


def _decision_from_payload(
    autotune_module: Any,
    fingerprint: str,
    payload: dict[str, Any],
) -> Any | None:
    benchmark_schema = str(payload.get("benchmark_schema", payload.get("schema_version", "")))
    if benchmark_schema and benchmark_schema != autotune_module._BENCHMARK_SCHEMA_VERSION:
        return None
    try:
        selected = autotune_module.ServerVariant(**payload["selected"])
        probes = tuple(
            autotune_module.ProbeResult(
                variant=autotune_module.ServerVariant(**item["variant"]),
                ok=bool(item["ok"]),
                output_sha256=str(item["output_sha256"]),
                predicted_tokens=int(item["predicted_tokens"]),
                predicted_tps=float(item["predicted_tps"]),
                prompt_tps=float(item["prompt_tps"]),
                elapsed_seconds=float(item["elapsed_seconds"]),
                error=str(item.get("error", "")),
            )
            for item in payload.get("probes", [])
            if isinstance(item, dict) and isinstance(item.get("variant"), dict)
        )
        return autotune_module.AutotuneDecision(
            fingerprint=fingerprint,
            selected=selected,
            baseline_tps=float(payload["baseline_tps"]),
            selected_tps=float(payload["selected_tps"]),
            speedup=float(payload["speedup"]),
            probes=probes,
        )
    except Exception:
        return None


def _decision_payload(autotune_module: Any, decision: Any) -> dict[str, Any]:
    return {
        "benchmark_schema": autotune_module._BENCHMARK_SCHEMA_VERSION,
        "selected": asdict(decision.selected),
        "baseline_tps": float(decision.baseline_tps),
        "selected_tps": float(decision.selected_tps),
        "speedup": float(decision.speedup),
        "probes": [
            {
                **asdict(probe),
                "variant": asdict(probe.variant),
            }
            for probe in decision.probes
        ],
    }


def install(autotune_module: Any, hardware_policy_module: Any) -> None:
    """Install native llama-server efficiency primitives owned by this module only.

    Runtime tuning, cache-reuse tuning, streaming and concurrency are composed by
    runtime_bootstrap instead of being imported and installed from this installer.
    """
    global _HARDWARE_IDENTITY_CACHE

    if getattr(autotune_module, "_mmm_server_efficiency_installed", False):
        return

    probe = autotune_module._probe_server
    if owns_contract_marker(probe, "_mmm_correctness_sentinel"):
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

    current_model_resolver = autotune_module._resolve_model_path
    if not getattr(current_model_resolver, "_mmm_process_model_path_cache", False):

        @wraps(current_model_resolver)
        def cached_model_path(config: Any) -> str:
            extra = getattr(config, "extra", {})
            filename = str(extra.get("gguf_filename", "")) if isinstance(extra, dict) else ""
            key = (str(getattr(config, "model_id", "")), filename)
            with _CACHE_LOCK:
                cached = _MODEL_PATH_CACHE.get(key)
            if cached:
                path = Path(cached)
                if path.is_file():
                    return str(path.resolve())
            resolved = str(Path(current_model_resolver(config)).expanduser().resolve())
            if Path(resolved).is_file():
                with _CACHE_LOCK:
                    _bounded_put(_MODEL_PATH_CACHE, key, resolved, limit=16)
            return resolved

        cached_model_path._mmm_process_model_path_cache = True  # type: ignore[attr-defined]
        autotune_module._resolve_model_path = cached_model_path

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

    current_server_version = autotune_module._server_version
    if not getattr(current_server_version, "_mmm_process_metadata_cache", False):

        @wraps(current_server_version)
        def cached_server_version(binary: str) -> str:
            path = Path(binary).expanduser()
            try:
                resolved = path.resolve()
                stat = resolved.stat()
                key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
            except OSError:
                return current_server_version(binary)
            with _CACHE_LOCK:
                cached = _SERVER_VERSION_CACHE.get(key)
                if cached is not None:
                    return cached
            value = current_server_version(binary)
            with _CACHE_LOCK:
                _bounded_put(_SERVER_VERSION_CACHE, key, value, limit=8)
            return value

        cached_server_version._mmm_process_metadata_cache = True  # type: ignore[attr-defined]
        autotune_module._server_version = cached_server_version

    current_hardware_identity = autotune_module._hardware_identity
    if not getattr(current_hardware_identity, "_mmm_process_metadata_cache", False):

        @wraps(current_hardware_identity)
        def cached_hardware_identity() -> str:
            global _HARDWARE_IDENTITY_CACHE
            with _CACHE_LOCK:
                if _HARDWARE_IDENTITY_CACHE is not None:
                    return _HARDWARE_IDENTITY_CACHE
            value = current_hardware_identity()
            with _CACHE_LOCK:
                _HARDWARE_IDENTITY_CACHE = value
            return value

        cached_hardware_identity._mmm_process_metadata_cache = True  # type: ignore[attr-defined]
        autotune_module._hardware_identity = cached_hardware_identity

    current_load_decision = autotune_module._load_cached_decision
    current_save_decision = autotune_module._save_decision
    if not getattr(current_load_decision, "_mmm_multi_decision_store", False):

        def load_cached_decision(fingerprint: str) -> Any | None:
            path = autotune_module._cache_path()
            with _CACHE_LOCK:
                payload = _read_decision_entries(path).get(fingerprint)
            if payload is None:
                return None
            return _decision_from_payload(autotune_module, fingerprint, payload)

        load_cached_decision._mmm_multi_decision_store = True  # type: ignore[attr-defined]
        load_cached_decision.__wrapped__ = current_load_decision  # type: ignore[attr-defined]
        autotune_module._load_cached_decision = load_cached_decision

        def save_decision(decision: Any) -> None:
            path = autotune_module._cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with _CACHE_LOCK:
                entries = _read_decision_entries(path)
                fingerprint = str(decision.fingerprint)
                entries.pop(fingerprint, None)
                entries[fingerprint] = _decision_payload(autotune_module, decision)
                while len(entries) > _DECISION_STORE_LIMIT:
                    entries.pop(next(iter(entries)))
                payload = {
                    "store_schema": _DECISION_STORE_SCHEMA,
                    "entries": entries,
                }
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)

        save_decision._mmm_multi_decision_store = True  # type: ignore[attr-defined]
        save_decision.__wrapped__ = current_save_decision  # type: ignore[attr-defined]
        autotune_module._save_decision = save_decision

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

    current_ensure = autotune_module.ensure_tuned_server
    if not getattr(current_ensure, "_mmm_managed_server_fast_path", False):

        @wraps(current_ensure)
        def ensure_managed_server_first(config: Any, request: Any) -> str:
            # The managed process is authoritative after MMM launches it. Avoid a
            # redundant local HTTP health request before every generation call.
            with autotune_module._AUTOTUNE_LOCK:
                process = autotune_module._MANAGED_PROCESS
                if process is not None and process.poll() is None:
                    url = autotune_module._MANAGED_URL
                    if url:
                        return str(url)
                    raise RuntimeError("managed llama-server process has no URL")
            return current_ensure(config, request)

        ensure_managed_server_first._mmm_managed_server_fast_path = True  # type: ignore[attr-defined]
        autotune_module.ensure_tuned_server = ensure_managed_server_first

    autotune_module._mmm_server_efficiency_installed = True


__all__ = ["install"]
