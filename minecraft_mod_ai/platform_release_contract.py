from __future__ import annotations

from functools import wraps
from typing import Any

from .imported_platform_repair import clear_marker
from .platform_catalog import adapter_for_lock_values, adapter_from_project


def install(mcp_tools_module: Any) -> None:
    cls = mcp_tools_module.MMMToolService
    original = cls.package_release
    if getattr(original, "_mmm_exact_platform_release_gate", False):
        return

    @wraps(original)
    def package_release(
        self: Any,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
        *args: Any,
        **kwargs: Any,
    ):
        approved = self._approved(proposal, approval_hash)
        root = self._existing_dir(project_root)
        expected = adapter_for_lock_values(approved.spec.platform)
        try:
            actual = adapter_from_project(root)
        except Exception as exc:
            raise RuntimeError(
                "Release is blocked until the project resolves to the exact approved "
                f"Fabric toolchain {expected.adapter_id}: {exc}"
            ) from exc
        if actual.adapter_id != expected.adapter_id:
            raise RuntimeError(
                "Release is blocked by platform mismatch: "
                f"approved={expected.adapter_id}, project={actual.adapter_id}."
            )

        # Once exact source metadata has independently resolved, the import-only
        # repair admission marker has served its purpose. Remove it before the normal
        # static/JAR packaging validation so it can never ship as release evidence.
        try:
            clear_marker(root)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        return original(
            self,
            project_root,
            proposal,
            approval_hash,
            *args,
            **kwargs,
        )

    package_release._mmm_exact_platform_release_gate = True
    cls.package_release = package_release
