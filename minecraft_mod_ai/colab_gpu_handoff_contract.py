from __future__ import annotations

import os
from functools import wraps
from typing import Any


def _release_native_llama_server() -> None:
    """Release the managed native text server before another exclusive GPU runtime."""
    from . import llama_server_autotune
    process = getattr(llama_server_autotune, '_MANAGED_PROCESS', None)
    if process is None or process.poll() is not None:
        return
    managed_url = getattr(llama_server_autotune, '_MANAGED_URL', None)
    llama_server_autotune._shutdown_managed_server()
    if managed_url and os.environ.get('LLAMA_SERVER_URL') == managed_url:
        os.environ.pop('LLAMA_SERVER_URL', None)
    llama_server_autotune._ATTEMPTED_KEYS.clear()

def _install_asset_handoff(*, services_module: Any, model_router_module: Any) -> None:
    current = services_module.generate_assets
    if getattr(current, '_mmm_atomic_gpu_handoff', False):
        return

    @wraps(current)
    def generate_assets_atomic_gpu_handoff(router: Any, *args: Any, **kwargs: Any):
        registry = getattr(router, 'registry', None)
        profile = getattr(router, 'profile', None)
        local_exclusive_image = False
        if registry is not None and profile is not None:
            try:
                config = registry.role(profile, 'image_generator')
                local_exclusive_image = config.provider == 'local' and config.adapter == 'image_diffusion' and config.exclusive_gpu
            except Exception:
                local_exclusive_image = False
        if not local_exclusive_image:
            return current(router, *args, **kwargs)
        with model_router_module._GPU_EXCLUSIVE_LOCK:
            _release_native_llama_server()
            return current(router, *args, **kwargs)
    generate_assets_atomic_gpu_handoff._mmm_atomic_gpu_handoff = True
    services_module.generate_assets = generate_assets_atomic_gpu_handoff

def install(*, services_module: Any, model_router_module: Any) -> None:
    """Serialize native-server eviction with local exclusive GPU consumers."""
    _install_asset_handoff(services_module=services_module, model_router_module=model_router_module)
__all__ = ['install']
