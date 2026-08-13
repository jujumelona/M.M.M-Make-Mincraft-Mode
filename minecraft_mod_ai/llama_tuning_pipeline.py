from __future__ import annotations

"""Single ownership point for native llama-server tuning composition.

The individual tuning modules own one concern each, while this pipeline is the only
place allowed to compose them. Runtime bootstrap installs exactly this pipeline,
which makes ordering explicit and prevents cross-module re-entry or accidental
multiple installation.
"""

from dataclasses import dataclass
from typing import Any, Callable


_TUNING_PIPELINE_VERSION = 5


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

    def stages(self) -> tuple[TuningStage, ...]:
        from . import agentic_optimization_contract, repair_engine
        from .llama_cache_reuse_efficiency_contract import install as install_cache_reuse
        from .llama_decode_speed_contract import install as install_decode_speed
        from .llama_server_efficiency_contract import install as install_efficiency
        from .llama_server_hardware_policy import install as install_hardware
        from .llama_server_runtime_tuning import install as install_runtime_tuning
        from .llama_structured_decode_policy import bind_structured_decode_policy
        from .planner_single_stream_search_contract import (
            install as install_single_stream_agentic_policy,
        )
        from .qwen35_mtp_hotpath_contract import install as install_qwen35_hotpath
        from .qwen35_t4_single_stream_tuning import (
            install as install_qwen35_t4_single_stream,
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
            # Keep the low-startup generic Qwen hotpath as the fail-safe, then let
            # Tesla T4 replace its fixed MTP width with a measured single-stream choice.
            install_qwen35_hotpath(self.autotune)
            install_qwen35_t4_single_stream(self.autotune)
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
