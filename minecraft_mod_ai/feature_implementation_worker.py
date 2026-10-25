"""Source-owned gameplay code is assigned to a separate, explicitly configured worker.

The 9B model may draft semantic behavior requirements, but it must never be
secretly promoted into the Java/API implementer for a missing Minecraft
capability. There is no local-model fallback and no invented callback on error.
"""
from __future__ import annotations

from .model_adapters import ModelConfigurationError


class FeatureImplementationUnavailable(RuntimeError):
    """The host cannot implement this capability with the selected execution owner."""


class FeatureImplementationRouter:
    """Keep journal/checkpoint calls intact while changing the inference owner."""

    def __init__(self, underlying):
        self.underlying = underlying

    def __getattr__(self, name):
        return getattr(self.underlying, name)

    @staticmethod
    def _role(role):
        if role in {"planner", "researcher", "coder", "coder_safe"}:
            return "feature_coder"
        return role

    def generate_structured_data(self, role, messages, *, data_name, parameters, **kwargs):
        return self.underlying.generate_structured_data(
            self._role(role), messages, data_name=data_name,
            parameters=parameters, **kwargs,
        )

    def generate_java_source(self, role, messages, *, data_name, source_frame, **kwargs):
        return self.underlying.generate_java_source(
            self._role(role), messages, data_name=data_name,
            source_frame=source_frame, **kwargs,
        )


def require_feature_implementation_router(router):
    """Fail BEFORE Java inference unless an independent implementation model exists.

    Journal wrappers carry the original router in .router; this adapter is
    applied around the journal, not below it, so the persisted inference role
    is 'feature_coder' and cannot replay an older 9B completion.
    """
    root = router
    seen = set()
    while not hasattr(root, "registry") and hasattr(root, "router"):
        if id(root) in seen:
            raise FeatureImplementationUnavailable("FEATURE_CODER_ROUTER_CYCLE")
        seen.add(id(root))
        root = root.router
    registry = getattr(root, "registry", None)
    profile = getattr(root, "profile", None)
    if registry is None or not isinstance(profile, str):
        raise FeatureImplementationUnavailable(
            "FEATURE_CODER_NOT_CONFIGURED: host-owned feature needs a "
            "separate implementation worker; MiMo cannot author unknown Java APIs"
        )
    try:
        feature = registry.role(profile, "feature_coder")
        small = registry.role(profile, "coder")
    except (ModelConfigurationError, KeyError, ValueError) as exc:
        raise FeatureImplementationUnavailable(
            "FEATURE_CODER_NOT_CONFIGURED: set MMM_FEATURE_CODER_MODEL, "
            "MMM_FEATURE_CODER_BASE_URL and MMM_FEATURE_CODER_API_KEY; "
            "new gameplay systems must not run through the 9B coder"
        ) from exc
    if (feature.adapter != "openai_compatible" or not feature.base_url
            or not feature.model_id or not feature.api_key):
        raise FeatureImplementationUnavailable("FEATURE_CODER_UNVERIFIED_PROVIDER")
    if (feature.model_id == small.model_id
            and feature.base_url == small.base_url):
        raise FeatureImplementationUnavailable(
            "FEATURE_CODER_NOT_INDEPENDENT: implementation worker equals small coder"
        )
    return FeatureImplementationRouter(router)
