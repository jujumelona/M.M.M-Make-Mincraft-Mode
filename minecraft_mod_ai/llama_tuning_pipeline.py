from __future__ import annotations

"""Single ownership point for native llama-server tuning composition.

The individual tuning modules own one concern each, while this pipeline is the only
place allowed to compose them.  Runtime bootstrap installs exactly this pipeline,
which makes ordering explicit and prevents cross-module re-entry or accidental
multiple installation.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TuningStage:
    name: str
    install: Callable[[], None]


class NativeLlamaTuningPipeline:
    """Install the native llama tuning stack once, in dependency order."""

    def __init__(
        self,
        *,
        autotune: Any,
        hardware_policy: Any,
        runtime_tuning: Any,
    ) -> None:
        self.autotune = autotune
        self.hardware_policy = hardware_policy
        self.runtime_tuning = runtime_tuning

    def stages(self) -> tuple[TuningStage, ...]:
        from .llama_cache_reuse_efficiency_contract import install as install_cache_reuse
        from .llama_decode_speed_contract import install as install_decode_speed
        from .llama_server_efficiency_contract import install as install_efficiency
        from .llama_server_hardware_policy import install as install_hardware
        from .llama_server_runtime_tuning import install as install_runtime_tuning

        return (
            TuningStage(
                "hardware",
                lambda: install_hardware(self.autotune),
            ),
            TuningStage(
                "efficiency",
                lambda: install_efficiency(self.autotune, self.hardware_policy),
            ),
            TuningStage(
                "runtime",
                lambda: install_runtime_tuning(self.autotune),
            ),
            TuningStage(
                "cache-reuse",
                lambda: install_cache_reuse(
                    self.autotune,
                    self.hardware_policy,
                    self.runtime_tuning,
                ),
            ),
            TuningStage(
                "decode-speed",
                lambda: install_decode_speed(self.autotune, self.runtime_tuning),
            ),
        )

    def install(self) -> None:
        if getattr(self.autotune, "_mmm_tuning_pipeline_installed", False):
            return
        installed: list[str] = []
        for stage in self.stages():
            stage.install()
            installed.append(stage.name)
        self.autotune._mmm_tuning_pipeline_stages = tuple(installed)
        self.autotune._mmm_tuning_pipeline_installed = True


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


__all__ = ["NativeLlamaTuningPipeline", "TuningStage", "install_native_llama_tuning_pipeline"]
