from __future__ import annotations

"""Single ownership point for native llama-server tuning composition.

The individual tuning modules own one concern each, while this pipeline is the only
place allowed to compose them. Runtime bootstrap installs exactly this pipeline,
which makes ordering explicit and prevents cross-module re-entry or accidental
multiple installation.
"""

import os
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable


_TUNING_PIPELINE_VERSION = 19
_PROFILE_CONTEXT_MARKER = "_mmm_profile_context_authority_v3"


@dataclass(frozen=True)
class TuningStage:
    name: str
    install: Callable[[], None]


class NativeLlamaTuningPipeline:
    """Install the native llama tuning stack once per composition version."""

    def __init__(self, *, autotune: Any, hardware_policy: Any, runtime_tuning: Any) -> None:
        self.autotune = autotune
        self.hardware_policy = hardware_policy
        self.runtime_tuning = runtime_tuning

    @staticmethod
    def _context_value(config: Any) -> int:
        """Resolve the one final llama context value without inventing a host cap.

        Precedence is model-specific explicit override, generic explicit override,
        then the model profile. Zero is meaningful and means llama.cpp/model-native
        context selection; it must never be rewritten to one token or a historical
        fixed window.
        """

        model_id = str(getattr(config, "model_id", "")).casefold()
        extra = getattr(config, "extra", {})
        filename = (
            str(extra.get("gguf_filename", "")).casefold()
            if isinstance(extra, dict)
            else ""
        )
        qwen35_mtp = "qwen3.5-9b" in model_id and (
            "mtp" in model_id or "mtp" in filename
        )
        override_names = (
            ("MMM_QWEN35_MTP_CTX", "MMM_LLAMA_SERVER_CTX")
            if qwen35_mtp
            else ("MMM_LLAMA_SERVER_CTX",)
        )
        for name in override_names:
            raw = os.environ.get(name, "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if value >= 0:
                return value

        try:
            return max(0, int(getattr(config, "max_context", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _install_profile_context_authority(self) -> None:
        """Install the final context owner after every lower-level tuning wrapper."""
        current = getattr(self.autotune, "_base_args", None)
        if not callable(current) or getattr(current, _PROFILE_CONTEXT_MARKER, False):
            return

        @wraps(current)
        def profile_context(binary: str, model_path: str, config: Any, port: int) -> list[str]:
            args = list(current(binary, model_path, config, port))
            context = self._context_value(config)
            for name in ("--ctx-size", "-c"):
                if name in args:
                    index = args.index(name)
                    if index + 1 < len(args):
                        args[index + 1] = str(context)
                        break
            else:
                args.extend(["--ctx-size", str(context)])
            return args

        setattr(profile_context, _PROFILE_CONTEXT_MARKER, True)
        self.autotune._base_args = profile_context

    def stages(self) -> tuple[TuningStage, ...]:
        from . import agentic_optimization_contract, repair_engine
        from .llama_cache_reuse_efficiency_contract import install as install_cache_reuse
        from .llama_decode_speed_contract import install as install_decode_speed
        from .llama_server_efficiency_contract import install as install_efficiency
        from .llama_server_hardware_policy import install as install_hardware
        from .llama_server_kernel_autotune import install as install_kernel_autotune
        from .llama_server_runtime_tuning import install as install_runtime_tuning
        from .llama_structured_decode_policy import bind_structured_decode_policy
        from .planner_single_stream_search_contract import (
            install as install_single_stream_agentic_policy,
        )
        from .qwen35_mtp_hotpath_contract import install as install_qwen35_hotpath
        from .qwen35_runtime_efficiency_contract import (
            install as install_qwen35_runtime_efficiency,
        )

        def install_hardware_stage() -> None:
            install_hardware(self.autotune)
            bind_structured_decode_policy(self.hardware_policy)

        def install_decode_speed_stage() -> None:
            install_decode_speed(
                self.autotune,
                self.runtime_tuning,
                self.hardware_policy,
            )
            install_qwen35_hotpath(self.autotune)
            # Qwen efficiency is outermost over the generic decode/KV tuner so its
            # request-local cold-start defaults are visible before any inner probe
            # decides whether to reload the model. The central pipeline owns this
            # composition; no policy contract installs another policy contract.
            install_qwen35_runtime_efficiency(
                self.autotune,
                self.hardware_policy,
                self.runtime_tuning,
            )
            # This is intentionally last: the profile/native context is the default
            # authority and explicit operator overrides retain their exact value.
            self._install_profile_context_authority()
            install_single_stream_agentic_policy(
                agentic_optimization_contract,
                repair_engine,
            )

        return (
            TuningStage("hardware", install_hardware_stage),
            TuningStage(
                "efficiency",
                lambda: install_efficiency(self.autotune, self.hardware_policy),
            ),
            TuningStage("runtime", lambda: install_runtime_tuning(self.autotune)),
            TuningStage(
                "cache-reuse",
                lambda: install_cache_reuse(
                    self.autotune,
                    self.hardware_policy,
                    self.runtime_tuning,
                ),
            ),
            TuningStage("decode-speed", install_decode_speed_stage),
            # Outermost cold-start search: select Flash Attention, logical batch and
            # independent K/V cache types first, then let the already-installed
            # MTP/ubatch/cache-reuse stages refine that measured winner.
            TuningStage(
                "kernel-autotune",
                lambda: install_kernel_autotune(self.autotune, self.runtime_tuning),
            ),
        )

    def install(self) -> None:
        installed_version = int(
            getattr(self.autotune, "_mmm_tuning_pipeline_version", 0) or 0
        )
        if installed_version >= _TUNING_PIPELINE_VERSION:
            return
        installed: list[str] = []
        for stage in self.stages():
            # Individual stage installers are idempotent, so an older live Colab
            # process can safely receive newly-added policy without a full restart.
            stage.install()
            installed.append(stage.name)
        self.autotune._mmm_tuning_pipeline_stages = tuple(installed)
        self.autotune._mmm_tuning_pipeline_installed = True
        self.autotune._mmm_tuning_pipeline_version = _TUNING_PIPELINE_VERSION


def install_native_llama_tuning_pipeline(
    *,
    autotune: Any,
    hardware_policy: Any,
    runtime_tuning: Any,
) -> None:
    NativeLlamaTuningPipeline(
        autotune=autotune,
        hardware_policy=hardware_policy,
        runtime_tuning=runtime_tuning,
    ).install()


__all__ = [
    "NativeLlamaTuningPipeline",
    "TuningStage",
    "install_native_llama_tuning_pipeline",
]
