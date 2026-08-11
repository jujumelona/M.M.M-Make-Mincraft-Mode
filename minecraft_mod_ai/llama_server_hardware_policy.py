from __future__ import annotations

import hashlib
import os
import shutil
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any


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
        candidates.append(Path(explicit_source).expanduser() / "build" / "bin" / "llama-server")
    candidates.append(Path("/content/llama.cpp/build/bin/llama-server"))
    candidates.append(Path.home() / ".cache" / "mmm" / "llama.cpp" / "build" / "bin" / "llama-server")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def _source_dir() -> Path:
    raw = os.environ.get("MMM_LLAMA_SERVER_SOURCE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".cache" / "mmm" / "llama.cpp").resolve()


def _bootstrap_native_server() -> str | None:
    """Return an already installed llama-server binary; never compile one."""

    existing = _existing_built_server()
    if existing is None:
        return None
    os.environ["MMM_LLAMA_SERVER_BIN"] = existing
    return existing


def _server_payload(adapter: Any, request: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "local",
        "messages": [dict(message) for message in request.messages],
        "max_tokens": int(adapter.config.max_new_tokens),
        "temperature": 0.0,
    }
    if getattr(request, "response_format", None) == "json":
        payload["response_format"] = {"type": "json_object"}
    return payload


def _strict_server_generate(adapter: Any, request: Any, server_url: str) -> str:
    """Use an explicitly selected server without silently loading a second GGUF."""

    from .model_adapters import ModelBackendError

    try:
        import httpx

        response = httpx.post(
            f"{server_url.rstrip('/')}/chat/completions",
            json=_server_payload(adapter, request),
            timeout=300,
        )
        if response.status_code != 200:
            body = response.text.strip().replace("\n", " ")
            if len(body) > 1200:
                body = body[:1200] + "..."
            raise RuntimeError(
                f"llama server returned HTTP {response.status_code}"
                + (f": {body}" if body else "")
            )
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama server response has no choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("llama server response has no message object.")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("llama server response content is not text.")
        if getattr(adapter.__class__, "_reported_server_url", None) != server_url:
            print("llama server: connected", server_url, flush=True)
            adapter.__class__._reported_server_url = server_url
        return content.strip()
    except Exception as exc:
        if isinstance(exc, ModelBackendError):
            raise
        raise ModelBackendError(
            role=adapter.config.role,
            model_id=adapter.config.model_id,
            cause=exc,
        ) from exc


def install(autotune_module: Any) -> None:
    """Keep managed llama-server tuning hardware-adaptive and correctness-gated."""

    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    if (
        getattr(LlamaCppAdapter.generate, "_mmm_server_autotuned", False)
        and not getattr(LlamaCppAdapter.generate, "_mmm_prefill_tuned", False)
    ):
        LlamaCppAdapter.generate._mmm_prefill_tuned = True

    original_server_binary = autotune_module._server_binary
    if not getattr(original_server_binary, "_mmm_native_bootstrap", False):

        @wraps(original_server_binary)
        def bootstrapped_server_binary() -> str | None:
            discovered = original_server_binary()
            if discovered is not None:
                return discovered
            return _bootstrap_native_server()

        bootstrapped_server_binary._mmm_native_bootstrap = True
        bootstrapped_server_binary._mmm_no_source_build = True
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
            try:
                index = args.index("--gpu-layers")
                args[index + 1] = "auto"
            except (ValueError, IndexError):
                pass
            if "--parallel" not in args and "-np" not in args:
                args.extend(["--parallel", "1"])
            return args

        adaptive_base_args._mmm_auto_gpu_layers = True
        adaptive_base_args._mmm_single_decode_slot = True
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

        adaptive_variant_args._mmm_auto_draft_layers = True
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
                            "You are a deterministic Java 17 code generator. "
                            "Output only valid Java code and no explanation."
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
                f"{measured.output_sha256}:{sentinel.output_sha256}".encode("utf-8")
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
                        for value in (measured.error, f"sentinel: {sentinel.error}")
                        if value
                    )
                ),
            )

        guarded_probe._mmm_correctness_sentinel = True
        autotune_module._probe_server = guarded_probe

    original_ensure = autotune_module.ensure_tuned_server
    if not getattr(original_ensure, "_mmm_colab_mtp_restart", False):

        @wraps(original_ensure)
        def ensure_with_colab_mtp(config: Any, request: Any) -> str | None:
            try:
                from .colab_mtp_server import (
                    SERVER_API_URL,
                    colab_mtp_server_enabled,
                    colab_mtp_server_running,
                    start_colab_mtp_server,
                )

                if colab_mtp_server_enabled():
                    if colab_mtp_server_running():
                        os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
                        return SERVER_API_URL
                    return start_colab_mtp_server(config)
            except Exception:
                pass
            return original_ensure(config, request)

        ensure_with_colab_mtp._mmm_colab_mtp_restart = True
        autotune_module.ensure_tuned_server = ensure_with_colab_mtp

    current_generate = LlamaCppAdapter.generate
    if not getattr(current_generate, "_mmm_explicit_server_strict", False):

        @wraps(current_generate)
        def strict_selected_server_generate(self: Any, request: Any) -> str:
            explicit = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
            if not explicit:
                try:
                    from .colab_mtp_server import colab_mtp_server_enabled

                    if colab_mtp_server_enabled():
                        explicit = (
                            autotune_module.ensure_tuned_server(self.config, request)
                            or ""
                        ).strip().rstrip("/")
                except Exception:
                    explicit = ""
            if explicit:
                return _strict_server_generate(self, request, explicit)
            return current_generate(self, request)

        strict_selected_server_generate._mmm_explicit_server_strict = True
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
                except Exception:
                    local_exclusive_image = False

            if local_exclusive_image:
                try:
                    from .colab_mtp_server import (
                        colab_mtp_server_enabled,
                        stop_colab_mtp_server,
                    )

                    if colab_mtp_server_enabled():
                        stop_colab_mtp_server(keep_enabled=True)
                except Exception:
                    pass

                process = getattr(autotune_module, "_MANAGED_PROCESS", None)
                if process is not None and process.poll() is None:
                    managed_url = getattr(autotune_module, "_MANAGED_URL", None)
                    autotune_module._shutdown_managed_server()
                    if managed_url and os.environ.get("LLAMA_SERVER_URL") == managed_url:
                        os.environ.pop("LLAMA_SERVER_URL", None)
                    autotune_module._ATTEMPTED_KEYS.clear()

            return original_assets(router, *args, **kwargs)

        assets_with_llama_release._mmm_releases_managed_llama = True
        assets_with_llama_release._mmm_releases_colab_mtp = True
        services.generate_assets = assets_with_llama_release
