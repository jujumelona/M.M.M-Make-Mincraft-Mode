from __future__ import annotations

"""On-demand native llama.cpp multimodal support.

Text generation keeps the lean GGUF server. A request carrying ``media_paths``
upgrades the managed server to the profile-declared projector only when vision is
actually needed, then transports local images through llama.cpp's OpenAI-compatible
``image_url`` content parts. Decode autotune decisions remain reusable, while
registry-declared MTP runtimes that do not advertise native MTP support launch media
requests non-speculatively.
"""

import base64
import hashlib
import json
import mimetypes
import os
import threading
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Mapping


_BASE_ARGS_MARKER = "_mmm_llama_multimodal_base_args_v2"
_BENCHMARK_MARKER = "_mmm_llama_multimodal_text_benchmark_v1"
_ENSURE_MARKER = "_mmm_llama_multimodal_ensure_v2"
_FINGERPRINT_MARKER = "_mmm_llama_multimodal_fingerprint_v1"
_LAUNCH_MARKER = "_mmm_llama_multimodal_safe_launch_v1"
_PAYLOAD_MARKER = "_mmm_llama_multimodal_payload_v2"
_ACTIVE_MEDIA_ENV = "MMM_LLAMA_MULTIMODAL_ACTIVE"
_MANAGED_MEDIA_URL_ATTR = "_mmm_multimodal_managed_url"
_MANAGED_MEDIA_PROCESS_ATTR = "_mmm_multimodal_managed_process"


def _extra(config: Any) -> Mapping[str, Any]:
    value = getattr(config, "extra", {})
    return value if isinstance(value, Mapping) else {}


def _mmproj_filename(config: Any) -> str:
    return str(_extra(config).get("mmproj_filename", "")).strip()


def _repo_id(config: Any) -> str:
    model_id = str(getattr(config, "model_id", "")).strip()
    if "/" not in model_id:
        return f"bartowski/{model_id}-GGUF"
    if not model_id.lower().endswith("-gguf") and "gguf" not in model_id.lower():
        return f"bartowski/{model_id.split('/')[-1]}-GGUF"
    return model_id


def _requires_media_baseline(config: Any) -> bool:
    """Return whether declared runtime metadata requires vision without MTP."""

    extra = _extra(config)
    return bool(
        str(extra.get("runtime_contract", "")).strip().casefold() == "qwen"
        and _mmproj_filename(config)
        and not bool(extra.get("native_mtp", False))
    )


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
    elif isinstance(current, list) and all(isinstance(part, Mapping) for part in current):
        text_parts = [dict(part) for part in current]
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
        if os.environ.get(_ACTIVE_MEDIA_ENV, "").strip() != "1":
            return args
        projector = _resolve_mmproj_path(config)
        if projector is None:
            raise RuntimeError(
                "A multimodal llama.cpp request requires a profile-declared mmproj_filename."
            )
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


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _install_fingerprint_policy(autotune: Any) -> None:
    """Invalidate pre-policy caches while keeping one decision shared by text/media."""

    current = autotune._fingerprint
    if getattr(current, _FINGERPRINT_MARKER, False):
        return

    @wraps(current)
    def fingerprint(config: Any, binary: str, model_path: str) -> str:
        base = str(current(config, binary, model_path))
        if not _requires_media_baseline(config):
            return base
        payload = {
            "base": base,
            "multimodal_mtp_policy": "text-benchmark-baseline-media-v1",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    setattr(fingerprint, _FINGERPRINT_MARKER, True)
    autotune._fingerprint = fingerprint


def _install_benchmark_policy(autotune: Any) -> None:
    """Keep media cold-start autotuning text-only while final launch stays multimodal."""

    current = autotune._benchmark
    if getattr(current, _BENCHMARK_MARKER, False):
        return

    @wraps(current)
    def benchmark(
        binary: str,
        model_path: str,
        config: Any,
        request: Any,
        fingerprint: str,
    ) -> Any:
        if (
            os.environ.get(_ACTIVE_MEDIA_ENV, "").strip() != "1"
            or not _requires_media_baseline(config)
        ):
            return current(binary, model_path, config, request, fingerprint)

        previous = os.environ.get(_ACTIVE_MEDIA_ENV)
        os.environ.pop(_ACTIVE_MEDIA_ENV, None)
        try:
            return current(binary, model_path, config, request, fingerprint)
        finally:
            _restore_env(_ACTIVE_MEDIA_ENV, previous)

    setattr(benchmark, _BENCHMARK_MARKER, True)
    autotune._benchmark = benchmark


def _install_launch_policy(autotune: Any) -> None:
    """Keep text MTP, but use baseline vision where registry metadata requires it."""

    current = autotune._launch_selected
    if getattr(current, _LAUNCH_MARKER, False):
        return

    @wraps(current)
    def launch_selected(binary: str, model_path: str, config: Any, selected: Any) -> str:
        if (
            os.environ.get(_ACTIVE_MEDIA_ENV, "").strip() == "1"
            and _requires_media_baseline(config)
            and str(getattr(selected, "spec_type", "none")) != "none"
        ):
            selected = autotune.ServerVariant("baseline")
        return current(binary, model_path, config, selected)

    setattr(launch_selected, _LAUNCH_MARKER, True)
    autotune._launch_selected = launch_selected


def _managed_server_ready(autotune: Any) -> tuple[Any | None, str]:
    process = getattr(autotune, "_MANAGED_PROCESS", None)
    url = str(getattr(autotune, "_MANAGED_URL", "") or "").strip()
    if process is None or process.poll() is not None or not url:
        return None, ""
    return process, url


def _clear_media_identity(autotune: Any) -> None:
    for name in (_MANAGED_MEDIA_PROCESS_ATTR, _MANAGED_MEDIA_URL_ATTR):
        if hasattr(autotune, name):
            delattr(autotune, name)


def _is_managed_media_server(autotune: Any, process: Any | None, url: str) -> bool:
    return (
        process is not None
        and process is getattr(autotune, _MANAGED_MEDIA_PROCESS_ATTR, None)
        and bool(url)
        and url == str(getattr(autotune, _MANAGED_MEDIA_URL_ATTR, "") or "")
    )


def _retire_managed_server(autotune: Any, managed_url: str) -> None:
    autotune._shutdown_managed_server()
    if os.environ.get("LLAMA_SERVER_URL", "").strip() == managed_url:
        os.environ.pop("LLAMA_SERVER_URL", None)
    _clear_media_identity(autotune)


def _install_ensure(autotune: Any) -> None:
    current = autotune.ensure_tuned_server
    if getattr(current, _ENSURE_MARKER, False):
        return
    state_lock = getattr(autotune, "_AUTOTUNE_LOCK", threading.RLock())

    @wraps(current)
    def ensure(config: Any, request: Any) -> str:
        with state_lock:
            media_paths = tuple(getattr(request, "media_paths", ()) or ())
            process, managed_url = _managed_server_ready(autotune)
            if process is None:
                _clear_media_identity(autotune)

            if not media_paths:
                if _is_managed_media_server(autotune, process, managed_url):
                    _retire_managed_server(autotune, managed_url)
                return current(config, request)

            if not _mmproj_filename(config):
                raise RuntimeError(
                    "This llama.cpp model received media_paths but declares no mmproj_filename."
                )

            if _is_managed_media_server(autotune, process, managed_url):
                return current(config, request)

            if process is not None:
                _retire_managed_server(autotune, managed_url)

            previous = os.environ.get(_ACTIVE_MEDIA_ENV)
            os.environ[_ACTIVE_MEDIA_ENV] = "1"
            try:
                url = current(config, request)
            finally:
                _restore_env(_ACTIVE_MEDIA_ENV, previous)

            process, managed_url = _managed_server_ready(autotune)
            if process is not None and url == managed_url:
                setattr(autotune, _MANAGED_MEDIA_PROCESS_ATTR, process)
                setattr(autotune, _MANAGED_MEDIA_URL_ATTR, managed_url)
            return url

    setattr(ensure, _ENSURE_MARKER, True)
    autotune.ensure_tuned_server = ensure


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
        if not _mmproj_filename(config):
            raise RuntimeError(
                "This llama.cpp model received media_paths but declares no mmproj_filename."
            )
        result["messages"] = _messages_with_media(request)
        return result

    setattr(payload, _PAYLOAD_MARKER, True)
    hardware_policy._server_payload = payload


def install(autotune: Any, hardware_policy: Any) -> None:
    _install_base_args(autotune)
    _install_fingerprint_policy(autotune)
    _install_benchmark_policy(autotune)
    _install_launch_policy(autotune)
    _install_ensure(autotune)
    _install_payload(hardware_policy)


__all__ = [
    "_install_benchmark_policy",
    "_install_fingerprint_policy",
    "_messages_with_media",
    "_requires_media_baseline",
    "_resolve_mmproj_path",
    "install",
]
