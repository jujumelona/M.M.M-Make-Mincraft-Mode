from __future__ import annotations

from .system_source_template import render_system_source


def _class_skill_java(package_name: str, class_name: str, resource: str) -> str:
    return render_system_source(
        "class_skill",
        package_name=package_name,
        class_name=class_name,
        resource=resource,
    )


__all__ = ["_class_skill_java"]
