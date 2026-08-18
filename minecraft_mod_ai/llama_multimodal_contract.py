from __future__ import annotations

"""Native llama.cpp multimodal support for model profiles that declare an mmproj.

The model registry owns which projector belongs to a GGUF.  This module binds that
artifact to llama-server startup and converts host-owned local image paths into the
OpenAI-compatible ``image_url`` content parts accepted by llama.cpp/libmtmd.
"""

import base64
import hashlib
import json
import mimetypes
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Mapping


_BASE_ARGS_MARKER = "_mmm_llama_multimodal_base_args_v1"
_FINGERPRINT_MARKER = "_mmm_llama_multimodal_fingerprint_v1"
_PAYLOAD_MARKER = "_mmm_llama_multimodal_payload_v1"


def _mmproj_filename(config: Any) -> str:
    extra = getattr(config, "extra", {})
    if not isinstance(extra, Mapping):
        return ""
    return str(extra.get("mmproj_filename", "")).strip()


def _repo_id(config: Any) -> str:
    model_id = str(getattr(config, "model_id", "")).strip()
    if "/" not in model_id:
        return f"bartowski/{model_id}-GGUF"
    if not model_id.lower().endswith("-gguf") and "gguf" not in model_id.lower():
        return f"bartowski/{model_id.split('/')[-1]}-GGUF"
    return model_id


@lru_cache(maxsize=16)
def _download_projector(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return str(hf_hub_download(repo_id=repo_id, filename=filename))


def _resolve_mmproj_path(config: Any) -> str | None:
    filename = _mmproj_filename(config)
    if not filename:
        return None

    model_id = str(getattr(config, "model_id", "")).strip()
    model_path = Path(model_id).expanduser()
    if model_path.is_file():
        sibling = model_path.resolve().parent / filename
        if not sibling.is_file() or sibling.is_symlink():
            raise RuntimeError(
                f"Declared llama.cpp multimodal projector is unavailable: {sibling}"
            )
        return str(sibling.resolve())
    return _download_projector(_repo_id(config), filename)


def _image_data_url(path: Path) -> str:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"Multimodal media must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Multimodal media is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"Multimodal media must be a regular file: {resolved}")
    mime, _encoding = mimetypes.guess_type(resolved.name)
    if not mime or not mime.startswith("image/"):
        raise RuntimeError(
            f"llama.cpp visual media must be an image file, got {resolved.name!r}"
        )
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _messages_with_media(request: Any) -> list[dict[str, Any]]:
    messages = [dict(message) for message in getattr(request, "messages", ())]
    media_paths = tuple(getattr(request, "media_paths", ()) or ())
    if not media_paths:
        return messages

    target_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if str(messages[index].get("role", "")).strip().casefold() == "user"
        ),
        None,
    )
    if target_index is None:
        raise RuntimeError("Multimodal llama.cpp requests require a user message.")

    current = messages[target_index].get("content", "")
    if isinstance(current, str):
        text_parts: list[dict[str, Any]] = (
            [{"type": "text", "text": current}] if current else []
        )
    elif isinstance(current, list):
        text_parts = [dict(part) if isinstance(part, Mapping) else part for part in current]
    else:
        raise RuntimeError(
            "Multimodal llama.cpp user content must be text or OpenAI content parts."
        )

    image_parts = [
        {
            "type": "image_url",
            "image_url": {"url": _image_data_url(Path(path))},
        }
        for path in media_paths
    ]
    messages[target_index]["content"] = [*image_parts, *text_parts]
    return messages


def _install_base_args(autotune: Any) -> None:
    current = autotune._base_args
    if getattr(current, _BASE_ARGS_MARKER, False):
        return

    @wraps(current)
    def base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
        args = list(current(binary, model_path, config, port))
        projector = _resolve_mmproj_path(config)
        if projector is None:
            return args
        for name in ("--mmproj", "-mm"):
            if name in args:
                index = args.index(name)
                if index + 1 < len(args):
                    args[index + 1] = projector
                    return args
        args.extend(["--mmproj", projector])
        return args

    setattr(base_args, _BASE_ARGS_MARKER, True)
    autotune._base_args = base_args


def _install_fingerprint(autotune: Any) -> None:
    current = autotune._fingerprint
    if getattr(current, _FINGERPRINT_MARKER, False):
        return

    @wraps(current)
    def fingerprint(config: Any, binary: str, model_path: str) -> str:
        base = str(current(config, binary, model_path))
        projector = _resolve_mmproj_path(config)
        if projector is None:
            return base
        path = Path(projector)
        stat = path.stat()
        payload = {
            "base": base,
            "mmproj_filename": _mmproj_filename(config),
            "mmproj_path": str(path.resolve()),
            "mmproj_size": int(stat.st_size),
            "mmproj_mtime_ns": int(stat.st_mtime_ns),
            "multimodal_contract": "v1",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    setattr(fingerprint, _FINGERPRINT_MARKER, True)
    autotune._fingerprint = fingerprint


def _install_payload(hardware_policy: Any) -> None:
    current = hardware_policy._server_payload
    if getattr(current, _PAYLOAD_MARKER, False):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = current(adapter, request)
        media_paths = tuple(getattr(request, "media_paths", ()) or ())
        if not media_paths:
            return result
        config = getattr(adapter, "config", None)
        if _resolve_mmproj_path(config) is None:
            raise RuntimeError(
                "This llama.cpp model received media_paths but declares no mmproj_filename."
            )
        result["messages"] = _messages_with_media(request)
        return result

    setattr(payload, _PAYLOAD_MARKER, True)
    hardware_policy._server_payload = payload


def install(autotune: Any, hardware_policy: Any) -> None:
    _install_base_args(autotune)
    _install_fingerprint(autotune)
    _install_payload(hardware_policy)


__all__ = [
    "_messages_with_media",
    "_resolve_mmproj_path",
    "install",
]
