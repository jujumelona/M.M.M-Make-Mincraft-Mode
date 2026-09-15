from __future__ import annotations

"""Final native cache policy for speculative MTP llama.cpp launches.

The generic runtime reserves a bounded RAM prompt cache. MTP-capable launch profiles
already reuse prompt state through llama.cpp's prompt cache and should not reserve the
additional ``--cache-ram`` arena. This policy is deliberately capability-driven: it
reuses the runtime tuner's canonical MTP support predicate instead of maintaining a
second model-name table.
"""

import hashlib
import json
from functools import wraps
from typing import Any

_BASE_ARGS_MARKER = "_mmm_mtp_prompt_cache_without_cache_ram_v1"
_FINGERPRINT_MARKER = "_mmm_mtp_prompt_cache_policy_fingerprint_v1"


def _drop_value_option(args: list[str], name: str) -> None:
    while name in args:
        index = args.index(name)
        del args[index]
        if index < len(args):
            del args[index]


def install(autotune_module: Any, runtime_tuning_module: Any) -> None:
    current_base = autotune_module._base_args
    if not getattr(current_base, _BASE_ARGS_MARKER, False):

        @wraps(current_base)
        def base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
            args = list(current_base(binary, model_path, config, port))
            if runtime_tuning_module._model_supports_mtp(config):
                _drop_value_option(args, "--cache-ram")
                if "--cache-prompt" not in args:
                    args.append("--cache-prompt")
            return args

        setattr(base_args, _BASE_ARGS_MARKER, True)
        autotune_module._base_args = base_args

    current_fingerprint = autotune_module._fingerprint
    if not getattr(current_fingerprint, _FINGERPRINT_MARKER, False):

        @wraps(current_fingerprint)
        def fingerprint(config: Any, binary: str, model_path: str) -> str:
            base = str(current_fingerprint(config, binary, model_path))
            if not runtime_tuning_module._model_supports_mtp(config):
                return base
            payload = {
                "base": base,
                "mtp_prompt_cache": "cache-prompt-without-cache-ram-v1",
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        setattr(fingerprint, _FINGERPRINT_MARKER, True)
        autotune_module._fingerprint = fingerprint


__all__ = ["install"]
