from __future__ import annotations

import hashlib
import os
from functools import wraps
from types import SimpleNamespace
from typing import Any


def install(autotune_module: Any) -> None:
    """Keep managed llama-server tuning hardware-adaptive and correctness-gated."""

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
            return args

        adaptive_base_args._mmm_auto_gpu_layers = True
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

            # MTP has had prompt-dependent deterministic divergence. The real first
            # workflow request remains the performance measurement, while this short
            # code-generation sentinel makes selection depend on a second independent
            # greedy token stream. Only candidates matching baseline on both survive.
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

    # A managed llama-server is a separate GPU process. The existing image runtime
    # can evict an in-process llama_cpp.Llama object, but it cannot free this external
    # process. Release it before a local FLUX shard acquires the GPU; the cached
    # autotune decision is then reused to restart the winner on the next LLM call.
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

            process = getattr(autotune_module, "_MANAGED_PROCESS", None)
            if (
                local_exclusive_image
                and process is not None
                and process.poll() is None
            ):
                managed_url = getattr(autotune_module, "_MANAGED_URL", None)
                autotune_module._shutdown_managed_server()
                if managed_url and os.environ.get("LLAMA_SERVER_URL") == managed_url:
                    os.environ.pop("LLAMA_SERVER_URL", None)
                # Permit ensure_tuned_server() to enter again. Because the successful
                # decision is already fingerprint-cached, this restarts the selected
                # variant without rerunning baseline/MTP probes.
                autotune_module._ATTEMPTED_KEYS.clear()

            return original_assets(router, *args, **kwargs)

        assets_with_llama_release._mmm_releases_managed_llama = True
        services.generate_assets = assets_with_llama_release
