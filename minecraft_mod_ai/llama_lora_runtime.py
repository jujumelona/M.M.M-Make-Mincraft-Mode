from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_LORA_STATE_LOCK = threading.RLock()
_ADAPTER_IDS_BY_ORIGIN: dict[str, dict[str, int]] = {}


@dataclass(frozen=True)
class LoraAdapterSpec:
    name: str
    base_model_id: str
    repo_id: str
    filename: str
    revision: str
    local_path: str
    scale: float
    priority: int
    roles: tuple[str, ...]
    stages: tuple[str, ...]
    when_tools: str


def _extra(config: Any) -> Mapping[str, Any]:
    extra = getattr(config, "extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _normalized_names(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise RuntimeError(f"{field} must be a string or list of strings")
    result: list[str] = []
    for item in values:
        text = str(item or "").strip().casefold()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _lora_enabled() -> bool:
    raw = os.environ.get("MMM_LLAMA_LORA", "1").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def configured_lora_specs(config: Any) -> tuple[LoraAdapterSpec, ...]:
    if not _lora_enabled():
        return ()
    extra = _extra(config)
    raw = extra.get("lora_adapters")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise RuntimeError("lora_adapters must be a mapping")

    foundation_base = str(extra.get("foundation_base_model_id", "") or "").strip()
    if not foundation_base:
        raise RuntimeError(
            "foundation_base_model_id is required when lora_adapters are configured"
        )

    specs: list[LoraAdapterSpec] = []
    for raw_name, raw_spec in raw.items():
        name = str(raw_name or "").strip()
        if not name:
            raise RuntimeError("LoRA adapter names must not be empty")
        if not isinstance(raw_spec, Mapping):
            raise RuntimeError(f"LoRA adapter {name!r} must be a mapping")
        enabled = raw_spec.get("enabled", True)
        if type(enabled) is not bool:
            raise RuntimeError(f"LoRA adapter {name!r} enabled must be boolean")
        if not enabled:
            continue

        base_model_id = str(raw_spec.get("base_model_id", "") or "").strip()
        if base_model_id != foundation_base:
            raise RuntimeError(
                f"LoRA adapter {name!r} targets base {base_model_id!r}, but the "
                f"foundation base is {foundation_base!r}"
            )
        repo_id = str(raw_spec.get("repo_id", "") or "").strip()
        filename = str(raw_spec.get("filename", "") or "").strip()
        revision = str(raw_spec.get("revision", "") or "").strip()
        local_path = str(raw_spec.get("path", "") or "").strip()
        if not local_path and (not repo_id or not filename):
            raise RuntimeError(
                f"LoRA adapter {name!r} requires path or repo_id+filename"
            )
        if filename and not filename.casefold().endswith(".gguf"):
            raise RuntimeError(f"LoRA adapter {name!r} filename must be GGUF")
        if local_path and not local_path.casefold().endswith(".gguf"):
            raise RuntimeError(f"LoRA adapter {name!r} path must be GGUF")

        try:
            scale = float(raw_spec.get("scale", 1.0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"LoRA adapter {name!r} scale must be numeric") from exc
        if not 0.0 < scale <= 4.0:
            raise RuntimeError(f"LoRA adapter {name!r} scale must be in (0, 4]")
        try:
            priority = int(raw_spec.get("priority", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"LoRA adapter {name!r} priority must be an integer") from exc

        when_tools = str(raw_spec.get("when_tools", "any") or "any").strip().casefold()
        if when_tools not in {"any", "present", "absent"}:
            raise RuntimeError(
                f"LoRA adapter {name!r} when_tools must be any/present/absent"
            )
        specs.append(
            LoraAdapterSpec(
                name=name,
                base_model_id=base_model_id,
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_path=local_path,
                scale=scale,
                priority=priority,
                roles=_normalized_names(raw_spec.get("roles"), field=f"{name}.roles"),
                stages=_normalized_names(raw_spec.get("stages"), field=f"{name}.stages"),
                when_tools=when_tools,
            )
        )
    return tuple(sorted(specs, key=lambda spec: (-spec.priority, spec.name)))


def lora_config_signature(config: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "base_model_id": spec.base_model_id,
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "revision": spec.revision,
            "path": spec.local_path,
            "scale": spec.scale,
            "priority": spec.priority,
            "roles": list(spec.roles),
            "stages": list(spec.stages),
            "when_tools": spec.when_tools,
        }
        for spec in configured_lora_specs(config)
    ]


def _download_hf_adapter(spec: LoraAdapterSpec) -> str:
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {"repo_id": spec.repo_id, "filename": spec.filename}
    if spec.revision:
        kwargs["revision"] = spec.revision
    return hf_hub_download(**kwargs)


def resolve_lora_artifacts(config: Any) -> tuple[tuple[LoraAdapterSpec, str], ...]:
    resolved: list[tuple[LoraAdapterSpec, str]] = []
    for spec in configured_lora_specs(config):
        raw_path = spec.local_path or _download_hf_adapter(spec)
        # Hugging Face snapshot files are commonly symlinks whose targets live in
        # extensionless `blobs/<sha256>` paths. Preserve the user/download-facing
        # GGUF path for llama-server and validate the target through is_file().
        path = Path(raw_path).expanduser().absolute()
        if path.suffix.casefold() != ".gguf" or not path.is_file():
            raise RuntimeError(
                f"LoRA adapter {spec.name!r} did not resolve to a regular GGUF file: {path}"
            )
        resolved.append((spec, str(path)))
    return tuple(resolved)


def lora_launch_args(config: Any) -> list[str]:
    artifacts = resolve_lora_artifacts(config)
    if not artifacts:
        return []
    args: list[str] = []
    for _spec, path in artifacts:
        args.extend(("--lora", path))
    # Do not trust launch defaults as the isolation boundary. MMM explicitly
    # zeroes all loaded adapters after /health becomes ready as well.
    args.append("--lora-init-without-apply")
    return args


def _server_origin(server_url: str) -> str:
    value = str(server_url or "").strip().rstrip("/")
    return value.removesuffix("/v1")


def _read_loaded_adapters(server_url: str) -> tuple[dict[str, Any], ...]:
    origin = _server_origin(server_url)
    response = httpx.get(f"{origin}/lora-adapters", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("llama-server /lora-adapters did not return a list")
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise RuntimeError("llama-server returned a malformed LoRA adapter row")
        try:
            adapter_id = int(raw["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("llama-server LoRA adapter row has no valid id") from exc
        path = str(raw.get("path", "") or "").strip()
        if adapter_id in seen_ids or not path:
            raise RuntimeError("llama-server returned ambiguous LoRA adapter metadata")
        seen_ids.add(adapter_id)
        rows.append({"id": adapter_id, "path": path, "scale": raw.get("scale", 0.0)})
    return tuple(rows)


def _adapter_id_map(
    server_url: str,
    config: Any,
    *,
    refresh: bool = False,
) -> dict[str, int]:
    origin = _server_origin(server_url)
    with _LORA_STATE_LOCK:
        if not refresh and origin in _ADAPTER_IDS_BY_ORIGIN:
            return dict(_ADAPTER_IDS_BY_ORIGIN[origin])

    rows = _read_loaded_adapters(server_url)
    by_filename: dict[str, int] = {}
    for row in rows:
        basename = Path(str(row["path"])).name
        if basename in by_filename:
            raise RuntimeError(f"llama-server loaded duplicate LoRA filename {basename!r}")
        by_filename[basename] = int(row["id"])

    result: dict[str, int] = {}
    for spec in configured_lora_specs(config):
        expected = Path(spec.local_path or spec.filename).name
        if expected not in by_filename:
            raise RuntimeError(
                f"llama-server did not load configured LoRA adapter {spec.name!r} ({expected})"
            )
        result[spec.name] = by_filename[expected]
    with _LORA_STATE_LOCK:
        _ADAPTER_IDS_BY_ORIGIN[origin] = dict(result)
    return result


def initialize_managed_server_loras(server_url: str, config: Any) -> None:
    specs = configured_lora_specs(config)
    if not specs:
        return
    origin = _server_origin(server_url)
    with _LORA_STATE_LOCK:
        rows = _read_loaded_adapters(server_url)
        expected = {Path(spec.local_path or spec.filename).name for spec in specs}
        loaded = {Path(str(row["path"])).name for row in rows}
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                "managed llama-server did not preload configured LoRA adapters: "
                + ", ".join(missing)
            )
        zero_payload = [
            {"id": int(row["id"]), "scale": 0.0}
            for row in sorted(rows, key=lambda row: int(row["id"]))
        ]
        response = httpx.post(
            f"{origin}/lora-adapters",
            json=zero_payload,
            timeout=10.0,
        )
        response.raise_for_status()
        _ADAPTER_IDS_BY_ORIGIN.pop(origin, None)
        _adapter_id_map(server_url, config, refresh=True)
    print(
        "llama lora: preloaded adapters disabled globally; request-scoped routing active",
        flush=True,
    )


def _selected_spec(config: Any, request: Any) -> LoraAdapterSpec | None:
    role = str(getattr(config, "role", "") or "").strip().casefold()
    metadata = getattr(request, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    stage = str(metadata.get("tool_stage", "") or "").strip().casefold()
    has_tools = bool(getattr(request, "tools", ()) or ())

    for spec in configured_lora_specs(config):
        if spec.roles and role not in spec.roles:
            continue
        if spec.stages and stage not in spec.stages:
            continue
        if spec.when_tools == "present" and not has_tools:
            continue
        if spec.when_tools == "absent" and has_tools:
            continue
        return spec
    return None


def request_lora_payload(
    server_url: str,
    config: Any,
    request: Any,
) -> list[dict[str, Any]] | None:
    specs = configured_lora_specs(config)
    if not specs:
        return None
    selected = _selected_spec(config, request)
    if selected is None:
        return []
    adapter_ids = _adapter_id_map(server_url, config)
    adapter_id = adapter_ids.get(selected.name)
    if adapter_id is None:
        raise RuntimeError(f"No runtime id exists for LoRA adapter {selected.name!r}")
    return [{"id": adapter_id, "scale": selected.scale}]


def apply_request_lora(
    payload: dict[str, Any],
    server_url: str,
    config: Any,
    request: Any,
) -> dict[str, Any]:
    routed = request_lora_payload(server_url, config, request)
    if routed is not None:
        payload["lora"] = routed
    return payload


__all__ = [
    "LoraAdapterSpec",
    "apply_request_lora",
    "configured_lora_specs",
    "initialize_managed_server_loras",
    "lora_config_signature",
    "lora_launch_args",
    "request_lora_payload",
    "resolve_lora_artifacts",
]
