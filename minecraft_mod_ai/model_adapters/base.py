from __future__ import annotations

import gc
import importlib.metadata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import Version


class ModelBackendError(RuntimeError):
    """A configured model backend failed; callers must not silently substitute another backend."""

    def __init__(self, *, role: str, model_id: str, cause: BaseException | str) -> None:
        self.role = role
        self.model_id = model_id
        self.cause = cause
        super().__init__(f"Model backend failed for role={role!r}, model={model_id!r}: {cause}")


class ModelConfigurationError(ValueError):
    pass


class HardwarePreflightError(ModelBackendError):
    pass


@dataclass(frozen=True)
class AdapterConfig:
    role: str
    adapter: str
    model_id: str = ""
    provider: str = "local"
    quantization: str | None = None
    torch_dtype: str = "auto"
    max_context: int = 8192
    # Hardware-safe prefill budget for one paginated request. Zero means the
    # native context window is the only bound. This is deliberately separate
    # from max_context: projects grow through pages instead of one quadratic
    # attention allocation.
    max_input_tokens: int = 0
    max_new_tokens: int = 1200
    min_free_vram_mb: int = 0
    exclusive_gpu: bool = False
    cpu_offload: bool = False
    base_url: str = ""
    api_key: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass(frozen=True)
class GenerationResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str = ""


@dataclass(frozen=True)
class GenerationRequest:
    messages: Sequence[Mapping[str, Any]]
    media_paths: tuple[Path, ...] = ()
    response_format: str = "text"
    response_schema: Mapping[str, Any] | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    tool_choice: str | Mapping[str, Any] | None = None
    parallel_tool_calls: bool = True


class ModelAdapter(ABC):
    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        raise NotImplementedError

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        """Return one structured assistant turn.

        Legacy adapters remain valid: adapters without native function calling simply
        produce a text-only turn. Tool-aware adapters override this method.
        """

        return GenerationResponse(content=self.generate(request))

    def close(self) -> None:
        _release_cuda()


def require_package(
    distribution: str,
    *,
    minimum: str | None = None,
    maximum_exclusive: str | None = None,
) -> None:
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModelConfigurationError(
            f"Required package {distribution!r} is not installed."
        ) from exc
    if minimum is not None and Version(installed) < Version(minimum):
        raise ModelConfigurationError(
            f"{distribution}>={minimum} is required; found {installed}."
        )
    if (
        maximum_exclusive is not None
        and Version(installed) >= Version(maximum_exclusive)
    ):
        raise ModelConfigurationError(
            f"{distribution}<{maximum_exclusive} is required; found {installed}."
        )


def preflight_cuda(config: AdapterConfig) -> None:
    if config.provider != "local" or config.min_free_vram_mb <= 0:
        return
    try:
        import torch
    except ImportError as exc:
        raise HardwarePreflightError(
            role=config.role,
            model_id=config.model_id,
            cause="PyTorch is not installed.",
        ) from exc
    if not torch.cuda.is_available():
        raise HardwarePreflightError(
            role=config.role,
            model_id=config.model_id,
            cause="CUDA is required by this local profile but no CUDA device is available.",
        )
    free_bytes, _ = torch.cuda.mem_get_info()
    free_mb = int(free_bytes / (1024 * 1024))
    if free_mb < config.min_free_vram_mb:
        raise HardwarePreflightError(
            role=config.role,
            model_id=config.model_id,
            cause=(
                f"Insufficient free VRAM: {free_mb} MiB available, "
                f"{config.min_free_vram_mb} MiB required. Unload other models first."
            ),
        )


def torch_dtype(name: str):
    import torch

    values = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ModelConfigurationError(f"Unsupported torch_dtype: {name!r}")
    return values[name]


def quantization_config(config: AdapterConfig):
    if config.quantization is None:
        return None
    if config.quantization != "bnb_4bit":
        raise ModelConfigurationError(
            f"Unsupported quantization for {config.role}: {config.quantization!r}"
        )
    require_package("bitsandbytes", minimum="0.45.0")
    from transformers import BitsAndBytesConfig
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
