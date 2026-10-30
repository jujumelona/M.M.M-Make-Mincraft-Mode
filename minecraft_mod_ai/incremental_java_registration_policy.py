"""Host-owned observation admission for constrained real coding adapters.

Production uses exact ABI-based occurrence wiring instead of asking a local
small model to browse an unrelated event menu. Explicit mock/fixture routers
retain their historical operation protocol for focused AST tests. Other
adapters can opt in through a construction context switch.
"""
from __future__ import annotations


def host_registration_enabled(builder) -> bool:
    context = getattr(builder, "context", None)
    if isinstance(context, dict) and "host_registration_enabled" in context:
        return context["host_registration_enabled"] is True
    from .fixed_template_generation import _adapter_name
    return _adapter_name(getattr(builder, "router", None), "coder") == "llama_cpp"
