from __future__ import annotations

"""Registration-level canonical leaf checks for pre-admission source generation.

A registered implementation may be rendered as a candidate before real execution
evidence exists. This helper deliberately does not turn ``not_reviewed`` into
``admitted``; production reuse remains owned by ``ResolvedVersionContext.require_leaf_binding``.
"""

from collections.abc import Mapping

from .resolved_version_context import VersionContextError

_REGISTRATION_FIELDS = (
    "implementation_id",
    "executor_type",
    "implementation_sha256",
    "validator_profile",
    "validator_sha256",
    "input_schema_sha256",
    "output_schema_sha256",
    "authority_sha256",
)
_HASH_FIELDS = (
    "implementation_sha256",
    "validator_sha256",
    "input_schema_sha256",
    "output_schema_sha256",
    "authority_sha256",
)


def _binding_from_context(resolved, leaf_id: str) -> Mapping:
    """Read registration facts, preserving the legacy strict context protocol."""

    bindings = getattr(resolved, "leaf_bindings", None)
    if isinstance(bindings, Mapping):
        if leaf_id not in bindings:
            raise VersionContextError(
                "HOST_FACT_UNAVAILABLE",
                category="leaf_bindings",
                name=leaf_id,
                context_id=getattr(resolved, "context_id", ""),
            )
        return bindings[leaf_id]

    # Compatibility with narrow/test contexts that expose only the established
    # strict admission method. This path can never admit ``not_reviewed`` because
    # require_leaf_binding() owns that fail-closed decision.
    require_leaf_binding = getattr(resolved, "require_leaf_binding", None)
    if callable(require_leaf_binding):
        return require_leaf_binding(leaf_id)
    raise VersionContextError(
        "HOST_FACT_UNAVAILABLE",
        category="leaf_bindings",
        name=leaf_id,
        context_id=getattr(resolved, "context_id", ""),
    )


def require_registered_leaf_binding(resolved, leaf_id: str) -> Mapping:
    """Return a structurally registered leaf without claiming execution admission.

    ``unsupported`` still fails closed. ``not_reviewed`` is accepted only for the
    registration/candidate-generation boundary; no evidence id is synthesized.
    """

    binding = _binding_from_context(resolved, leaf_id)
    if not isinstance(binding, Mapping):
        raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_id)
    state = binding.get("state")
    if state == "unsupported":
        raise VersionContextError(
            "UNSUPPORTED_LEAF",
            leaf=leaf_id,
            state=state,
            context_id=getattr(resolved, "context_id", ""),
        )
    if state not in {"admitted", "not_reviewed"}:
        raise VersionContextError(
            "HOST_LEAF_BINDING_INVALID",
            leaf=leaf_id,
            state=state,
        )
    impl = binding.get("implementation")
    if not isinstance(impl, Mapping):
        raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_id)
    missing = [
        field
        for field in _REGISTRATION_FIELDS
        if not isinstance(impl.get(field), str) or not impl.get(field, "").strip()
    ]
    if missing:
        raise VersionContextError(
            "HOST_LEAF_BINDING_INVALID",
            leaf=leaf_id,
            missing_fields=missing,
        )
    for field in _HASH_FIELDS:
        if not impl[field].startswith("sha256:"):
            raise VersionContextError(
                "HOST_LEAF_BINDING_INVALID",
                leaf=leaf_id,
                field=field,
            )
    return binding


__all__ = ["require_registered_leaf_binding"]
