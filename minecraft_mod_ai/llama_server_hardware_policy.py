from __future__ import annotations

from functools import wraps
from typing import Any


def install(autotune_module: Any) -> None:
    """Keep managed llama-server benchmarking runnable across VRAM sizes.

    The in-process adapter already degrades GPU layer offload when memory is tight.
    The managed server should not hard-fail before benchmarking merely because the
    selected GGUF is larger than available VRAM, so target and MTP draft placement
    both use llama.cpp's native ``auto`` policy.
    """

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
