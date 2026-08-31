from __future__ import annotations

"""Single ownership point for native llama-server tuning composition.

The individual tuning modules own one concern each, while this pipeline is the only
place allowed to compose them. Runtime bootstrap installs exactly this pipeline,
which makes ordering explicit and prevents cross-module re-entry or accidental
multiple installation.
"""

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .runtime_contract_composer import (
    ContractStage,
    call_shape,
    callable_boundary,
    compose_contract_stages,
)

_PROFILE_CONTEXT_MARKER = "_mmm_profile_context_authority"
_RUNTIME_TYPE_OWNER_MARKER = "_mmm_runtime_tuning_type_owner"
_AUTOTUNE_LIVENESS_MARKER = "_mmm_autotune_wall_clock_guard"
_AUTOTUNE_DEADLINE_LOCK = threading.RLock()
_AUTOTUNE_DEADLINE: float | None = None


def _bounded_seconds(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _autotune_wall_seconds() -> float:
    return _bounded_seconds(
        "MMM_LLAMA_AUTOTUNE_MAX_SECONDS", 300.0, minimum=30.0, maximum=1800.0
    )


def _autotune_step_seconds() -> int:
    return int(
        _bounded_seconds(
            "MMM_LLAMA_AUTOTUNE_STEP_TIMEOUT_SECONDS",
            90.0,
            minimum=10.0,
            maximum=300.0,
        )
    )


def _remaining_autotune_seconds() -> float | None:
    with _AUTOTUNE_DEADLINE_LOCK:
        deadline = _AUTOTUNE_DEADLINE
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _require_autotune_budget() -> None:
    remaining = _remaining_autotune_seconds()
    if remaining is not None and remaining <= 0:
        raise RuntimeError(
            "llama-server autotune wall-clock budget exhausted; refusing another tuning step"
        )


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


@dataclass(frozen=True)
class TuningStage:
    name: str
    install: Callable[[], None]


class NativeLlamaTuningPipeline:
    """Install one graph-pinned native llama tuning stack per process."""

    def __init__(self, *, autotune: Any, hardware_policy: Any, runtime_tuning: Any) -> None:
        self.autotune = autotune
        self.hardware_policy = hardware_policy
        self.runtime_tuning = runtime_tuning

    def _install_runtime_type_ownership(self) -> None:
        """Publish the canonical extended tuning type before any wrapper captures it."""

        for name in ("ServerVariant",):
            canonical = getattr(self.runtime_tuning, name, None)
            if canonical is None:
                raise RuntimeError(f"runtime tuning does not export canonical {name}")
            setattr(self.autotune, name, canonical)
        setattr(
            self.autotune,
            _RUNTIME_TYPE_OWNER_MARKER,
            str(getattr(self.runtime_tuning, "__name__", type(self.runtime_tuning).__name__)),
        )

    @staticmethod
    def _context_value(config: Any) -> int:
        """Resolve the launch context from explicit operator or registry policy.

        ``--ctx-size 0`` remains available for profiles that intentionally use the
        model-native context. Resource-constrained profiles may declare a
        ``runtime_context_default`` so runtime allocation can differ from the model's
        advertised maximum without changing that model capability.
        """

        extra = getattr(config, "extra", {})
        metadata = extra if isinstance(extra, dict) else {}
        qwen_t4_hotpath = (
            str(metadata.get("runtime_contract", "")).strip().casefold() == "qwen"
            and str(metadata.get("decode_hotpath", "")).strip().casefold() == "t4_mtp"
        )

        if qwen_t4_hotpath:
            raw = os.environ.get("MMM_QWEN35_MTP_CTX", "").strip()
            if raw:
                from .qwen35_mtp_hotpath_contract import _context_size

                return _context_size(config)
        else:
            raw = os.environ.get("MMM_LLAMA_SERVER_CTX", "").strip()
            if raw:
                try:
                    value = int(raw)
                except ValueError:
                    value = -1
                if value >= 0:
                    return value

        configured_default = metadata.get("runtime_context_default")
        if configured_default not in (None, ""):
            try:
                value = int(configured_default)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "runtime_context_default must be a positive integer"
                ) from exc
            if value <= 0:
                raise ValueError("runtime_context_default must be a positive integer")
            maximum = int(getattr(config, "max_context", 0) or 0)
            if maximum > 0 and value > maximum:
                raise ValueError(
                    "runtime_context_default cannot exceed the registered max_context"
                )
            return value
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

    def _install_autotune_liveness_guard(self) -> None:
        """Bound the complete native tuning search and every expensive probe step."""
        current_benchmark = getattr(self.autotune, "_benchmark", None)
        current_start_server = getattr(self.autotune, "_start_server", None)
        current_probe_server = getattr(self.autotune, "_probe_server", None)
        if not all(
            callable(value)
            for value in (current_benchmark, current_start_server, current_probe_server)
        ):
            raise RuntimeError(
                "native llama autotune liveness guard requires benchmark/start/probe owners"
            )
        if getattr(current_benchmark, _AUTOTUNE_LIVENESS_MARKER, False):
            return

        @wraps(current_start_server)
        def bounded_start_server(*args: Any, **kwargs: Any) -> Any:
            _require_autotune_budget()
            return current_start_server(*args, **kwargs)

        @wraps(current_probe_server)
        def bounded_probe_server(*args: Any, **kwargs: Any) -> Any:
            _require_autotune_budget()
            return current_probe_server(*args, **kwargs)

        @wraps(current_benchmark)
        def bounded_benchmark(*args: Any, **kwargs: Any) -> Any:
            global _AUTOTUNE_DEADLINE
            wall_seconds = _autotune_wall_seconds()
            step_seconds = min(_autotune_step_seconds(), max(10, int(wall_seconds)))
            previous_start_timeout = os.environ.get("MMM_LLAMA_SERVER_START_TIMEOUT")
            previous_probe_timeout = os.environ.get("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT")
            with _AUTOTUNE_DEADLINE_LOCK:
                previous_deadline = _AUTOTUNE_DEADLINE
                _AUTOTUNE_DEADLINE = time.monotonic() + wall_seconds
            try:
                current_start = (
                    int(previous_start_timeout) if previous_start_timeout else step_seconds
                )
            except ValueError:
                current_start = step_seconds
            try:
                current_probe = (
                    int(previous_probe_timeout) if previous_probe_timeout else step_seconds
                )
            except ValueError:
                current_probe = step_seconds
            os.environ["MMM_LLAMA_SERVER_START_TIMEOUT"] = str(
                max(1, min(step_seconds, current_start))
            )
            os.environ["MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT"] = str(
                max(1, min(step_seconds, current_probe))
            )
            try:
                return current_benchmark(*args, **kwargs)
            finally:
                _restore_env("MMM_LLAMA_SERVER_START_TIMEOUT", previous_start_timeout)
                _restore_env("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", previous_probe_timeout)
                with _AUTOTUNE_DEADLINE_LOCK:
                    _AUTOTUNE_DEADLINE = previous_deadline

        setattr(bounded_benchmark, _AUTOTUNE_LIVENESS_MARKER, True)
        bounded_start_server._mmm_autotune_budget_guard = True
        bounded_probe_server._mmm_autotune_budget_guard = True
        self.autotune._start_server = bounded_start_server
        self.autotune._probe_server = bounded_probe_server
        self.autotune._benchmark = bounded_benchmark

    def stages(self) -> tuple[TuningStage, ...]:
        from . import agentic_optimization_contract, repair_engine
        from .llama_cache_reuse_efficiency_contract import (
            install as install_cache_reuse,
        )
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
        from .qwen_runtime_transport_contract import (
            install as install_qwen_runtime_transport,
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
            except Exception:  # noqa: BLE001 - optional hardware identity must not block portable tuning
                hardware = ""
            if "t4" not in hardware:
                self.runtime_tuning._ubatch_candidates = original_ubatch_candidates
            install_vram_parallel(self.runtime_tuning)

        def install_multimodal_stage() -> None:
            install_multimodal(self.autotune, self.hardware_policy)
            self._install_autotune_liveness_guard()

        return (
            TuningStage("runtime-types", self._install_runtime_type_ownership),
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
            TuningStage("qwen-transport", install_qwen_runtime_transport),
            TuningStage("multimodal", install_multimodal_stage),
        )

    def _callable_boundaries(self):
        """Production call shapes every tuning wrapper must continue to accept."""

        return (
            callable_boundary(
                "autotune.server_variant",
                self.autotune,
                "ServerVariant",
                call_shapes=(
                    call_shape(
                        3,
                        "ubatch",
                        "parallel",
                        "cache_reuse",
                        "draft_p_min",
                    ),
                ),
            ),
            callable_boundary(
                "autotune.base_args",
                self.autotune,
                "_base_args",
                call_shapes=(call_shape(4),),
            ),
            callable_boundary(
                "autotune.fingerprint",
                self.autotune,
                "_fingerprint",
                call_shapes=(call_shape(3),),
            ),
            callable_boundary(
                "autotune.probe_server",
                self.autotune,
                "_probe_server",
                call_shapes=(call_shape(2, "max_tokens", "variant"),),
            ),
            callable_boundary(
                "autotune.start_server",
                self.autotune,
                "_start_server",
                call_shapes=(call_shape(5),),
            ),
            callable_boundary(
                "autotune.launch_selected",
                self.autotune,
                "_launch_selected",
                call_shapes=(call_shape(4),),
            ),
            callable_boundary(
                "autotune.benchmark",
                self.autotune,
                "_benchmark",
                call_shapes=(call_shape(5),),
            ),
            callable_boundary(
                "autotune.run_tuning_variant",
                self.autotune,
                "_mmm_run_tuning_variant",
                call_shapes=(call_shape(5, "probe_tokens"),),
            ),
            callable_boundary(
                "hardware.server_payload",
                self.hardware_policy,
                "_server_payload",
                call_shapes=(call_shape(2),),
            ),
            callable_boundary(
                "runtime.ubatch_candidates",
                self.runtime_tuning,
                "_ubatch_candidates",
                call_shapes=(call_shape(1),),
            ),
        )

    def install(self) -> None:
        if bool(getattr(self.autotune, "_mmm_tuning_pipeline_installed", False)):
            return

        receipts = compose_contract_stages(
            owner_name="native-llama-tuning",
            state_owner=self.autotune,
            stages=(
                ContractStage(stage.name, stage.install)
                for stage in self.stages()
            ),
            boundaries=self._callable_boundaries(),
        )
        self.autotune._mmm_tuning_pipeline_stages = tuple(
            receipt.name for receipt in receipts
        )
        self.autotune._mmm_tuning_pipeline_receipts = receipts
        self.autotune._mmm_tuning_pipeline_installed = True


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
