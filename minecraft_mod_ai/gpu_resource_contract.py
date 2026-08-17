from __future__ import annotations
from dataclasses import replace
from functools import wraps
from typing import Any
_LOCAL_GPU_TEXT_ADAPTERS = frozenset({'llama_cpp', 'vllm', 'transformers_text', 'transformers_multimodal'})

def install(model_registry_module: Any) -> None:
    cls = model_registry_module.ModelRegistry
    original = cls._resolve_role
    if getattr(original, '_mmm_gpu_resource_contract', False):
        return

    @staticmethod
    @wraps(original)
    def resolve_role(role: str, raw: Any):
        config = original(role, raw)
        if config.provider == 'local' and config.adapter in _LOCAL_GPU_TEXT_ADAPTERS and (not config.exclusive_gpu):
            config = replace(config, exclusive_gpu=True)
        return config
    resolve_role._mmm_gpu_resource_contract = True
    cls._resolve_role = resolve_role
