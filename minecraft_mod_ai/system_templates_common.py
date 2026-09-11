from __future__ import annotations

from .system_source_template import render_system_source


def _persistent_store_java(package_name: str, mod_id: str) -> str:
    return render_system_source(
        "persistent_store", package_name=package_name, mod_id=mod_id
    )


def _config_loader_java(package_name: str, mod_id: str) -> str:
    return render_system_source(
        "config_loader", package_name=package_name, mod_id=mod_id
    )


__all__ = ["_config_loader_java", "_persistent_store_java"]
