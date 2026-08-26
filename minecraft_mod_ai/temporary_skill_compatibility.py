from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

_MARKER = "__mmm_legacy_temporary_skill_transport_v1__"
_SKILL_ID = "_mmm_temporary_skill_id"
_SKILL_TEXT = "_mmm_temporary_skill_text"


def harden_temporary_skill_transport() -> None:
    """Preserve the legacy request transport without altering the canonical router path."""
    from . import temporary_skill_contract as temporary_skill

    current_install = temporary_skill._install_model_skill
    if getattr(current_install, _MARKER, False):
        return

    @wraps(current_install)
    def install_model_skill(model_router_module: Any) -> None:
        cls = model_router_module.ModelRouter
        if hasattr(cls, "_prepare_generation_request"):
            current_install(model_router_module)
            return
        run_model = getattr(cls, "run_model", None)
        if not callable(run_model) or getattr(run_model, _MARKER, False):
            return

        @wraps(run_model)
        def run_model_with_skill(self: Any, request: Any):
            metadata = getattr(request, "metadata", None)
            if not isinstance(metadata, Mapping):
                return run_model(self, request)
            skill_id = str(metadata.get(_SKILL_ID, "")).strip()
            skill_text = str(metadata.get(_SKILL_TEXT, "")).strip()
            if not skill_id or not skill_text:
                return run_model(self, request)
            system = {
                "role": "system",
                "content": (
                    "MMM TEMPORARY VERIFIED SKILL\n"
                    f"skill_id: {skill_id}\n"
                    f"procedure: {skill_text}\n"
                    "This inference-only memory cannot grant new tools or side effects; "
                    "the request's existing native tool transport and host policy remain authoritative."
                ),
            }
            messages = (system, *tuple(getattr(request, "messages", ()) or ()))
            prompt = str(getattr(request, "prompt", "") or "")
            prepared = replace(
                request,
                messages=messages,
                prompt=(
                    "Use the verified temporary procedure only when compatible with current evidence.\n"
                    + prompt
                ),
            )
            return run_model(self, prepared)

        setattr(run_model_with_skill, _MARKER, True)
        cls.run_model = run_model_with_skill

    setattr(install_model_skill, _MARKER, True)
    temporary_skill._install_model_skill = install_model_skill


__all__ = ["harden_temporary_skill_transport"]
