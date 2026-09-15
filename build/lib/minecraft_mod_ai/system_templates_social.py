from __future__ import annotations

from .system_source_template import render_system_source


def _gui_java(package_name: str, mod_id: str, class_name: str, resource: str) -> str:
    return render_system_source(
        "gui_networking",
        package_name=package_name,
        mod_id=mod_id,
        class_name=class_name,
        resource=resource,
    )


__all__ = ["_gui_java"]
