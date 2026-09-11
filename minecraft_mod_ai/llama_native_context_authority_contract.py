from __future__ import annotations

"""Keep llama.cpp native context sizing authoritative when no override exists."""

import os
from functools import wraps
from types import ModuleType
from typing import Any

_MARKER = "_mmm_native_context_authority_final"


def _drop_context(args: list[str]) -> list[str]:
    result = list(args)
    for name in ("--ctx-size", "-c"):
        while name in result:
            index = result.index(name)
            del result[index]
            if index < len(result):
                del result[index]
    return result


def _is_qwen_hotpath(config: Any) -> bool:
    extra = getattr(config, "extra", {})
    return bool(
        isinstance(extra, dict)
        and str(extra.get("runtime_contract", "")).strip().casefold() == "qwen"
        and str(extra.get("decode_hotpath", "")).strip().casefold() == "t4_mtp"
    )


def _explicit_context(config: Any) -> int | None:
    if _is_qwen_hotpath(config):
        raw = os.environ.get("MMM_QWEN35_MTP_CTX", "").strip()
    else:
        raw = os.environ.get("MMM_LLAMA_SERVER_CTX", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value > 0:
            return value
    extra = getattr(config, "extra", {})
    configured = extra.get("runtime_context_default") if isinstance(extra, dict) else None
    if configured not in (None, ""):
        value = int(configured)
        if value <= 0:
            raise ValueError("runtime_context_default must be a positive integer")
        maximum = int(getattr(config, "max_context", 0) or 0)
        if maximum > 0 and value > maximum:
            raise ValueError("runtime_context_default cannot exceed the registered max_context")
        return value
    return None


def _set_context(args: list[str], value: int) -> list[str]:
    result = list(args)
    for name in ("--ctx-size", "-c"):
        if name in result:
            index = result.index(name)
            if index + 1 < len(result):
                result[index + 1] = str(value)
                return result
    result.extend(["--ctx-size", str(value)])
    return result


def install(autotune_module: ModuleType, tuning_pipeline_module: ModuleType) -> None:
    """Finalize both active launch args and future pipeline instances."""

    current = autotune_module._base_args
    if not bool(getattr(current, _MARKER, False)):

        @wraps(current)
        def native_context_args(
            binary: str, model_path: str, config: Any, port: int
        ) -> list[str]:
            args = list(current(binary, model_path, config, port))
            context = _explicit_context(config)
            if context is None:
                return _drop_context(args)
            return _set_context(args, context)

        setattr(native_context_args, _MARKER, True)
        native_context_args.__wrapped__ = current
        autotune_module._base_args = native_context_args

    cls = tuning_pipeline_module.NativeLlamaTuningPipeline
    method = cls._install_profile_context_authority
    if bool(getattr(method, _MARKER, False)):
        return

    def install_profile_context_authority(self: Any) -> None:
        current_base = getattr(self.autotune, "_base_args", None)
        if not callable(current_base) or bool(getattr(current_base, _MARKER, False)):
            return

        @wraps(current_base)
        def authoritative(
            binary: str, model_path: str, config: Any, port: int
        ) -> list[str]:
            args = list(current_base(binary, model_path, config, port))
            context = _explicit_context(config)
            if context is None:
                return _drop_context(args)
            return _set_context(args, context)

        setattr(authoritative, _MARKER, True)
        authoritative.__wrapped__ = current_base
        self.autotune._base_args = authoritative

    setattr(install_profile_context_authority, _MARKER, True)
    install_profile_context_authority.__wrapped__ = method
    cls._install_profile_context_authority = install_profile_context_authority


__all__ = ["install"]
