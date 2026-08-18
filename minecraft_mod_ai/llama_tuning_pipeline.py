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


_TUNING_PIPELINE_VERSION = 29
_PROFILE_CONTEXT_MARKER = "_mmm_profile_context_authority_v5"


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
        """Resolve the final llama context without creating a second Qwen authority."""
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
        if qwen35_mtp:
            from .qwen35_mtp_hotpath_contract import _context_size

            return _context_size(config)

        raw = os.environ.get("MMM_LLAMA_SERVER_CTX", "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = -1
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
        from .llama_multimodal_contract import install as install_multimodal
        from .llama_server_efficiency_contract import install as install_efficiency
        from .llama_server_hardware_policy import install as install_hardware
        from .llama_server_kernel_autotune import install as install_kernel_autotune
        from .llama_server_runtime_tuning import install as install_runtime_tuning
        from .llama_structured_decode_policy import bind_structured_decode_policy
        from .llama_vram_parallel_policy import install as install_vram_parallel
        from .planner_single_stream_search_contract import (
            install as install_single_stream_agentic_policy,
        )
        from .qwen35_mtp_hotpath_contract import install as install_qwen35_hotpath
        from .qwen35_request_policy import install as install_qwen35_request_policy
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
            install_qwen35_runtime_efficiency(
                self.autotune,
                self.hardware_policy,
                self.runtime_tuning,
            )
            # Install last among Qwen wrappers so role/task semantics remain the
            # authoritative request policy and the full tuning benchmark is scoped.
            install_qwen35_request_policy(self.autotune, self.hardware_policy)
            self._install_profile_context_authority()
            install_single_stream_agentic_policy(
                agentic_optimization_contract,
                repair_engine,
            )

        def install_kernel_stage() -> None:
            original_ubatch_candidates = self.runtime_tuning._ubatch_candidates
            install_kernel_autotune(self.autotune, self.runtime_tuning)
            try:
                hardware = str(self.autotune._hardware_identity()).casefold()
            except Exception:
                hardware = ""
            if "t4" not in hardware:
                self.runtime_tuning._ubatch_candidates = original_ubatch_candidates
            install_vram_parallel(self.runtime_tuning)

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
            TuningStage("kernel-autotune", install_kernel_stage),
            # This must be outermost. A media request may retire a text-only managed
            # process; Qwen/T4 wrappers underneath must then observe the cold launch
            # and reapply their draft-GPU/KV launch scope around the cached winner.
            TuningStage(
                "multimodal",
                lambda: install_multimodal(self.autotune, self.hardware_policy),
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
    from .runtime_stability_contract import install as install_runtime_stability

    NativeLlamaTuningPipeline(
        autotune=autotune,
        hardware_policy=hardware_policy,
        runtime_tuning=runtime_tuning,
    ).install()
    install_runtime_stability()


__all__ = [
    "NativeLlamaTuningPipeline",
    "TuningStage",
    "install_native_llama_tuning_pipeline",
]
