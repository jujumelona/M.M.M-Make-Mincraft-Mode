from __future__ import annotations

from typing import Any, Mapping


def _blocking_errors(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("diagnostics", {})
    if isinstance(raw, Mapping):
        values = [
            dict(item)
            for group in raw.values()
            if isinstance(group, list)
            for item in group
            if isinstance(item, Mapping)
        ]
    elif isinstance(raw, list):
        values = [dict(item) for item in raw if isinstance(item, Mapping)]
    else:
        values = []
    return [item for item in values if int(item.get("severity", 1)) == 1]


def install(orchestrator_module: Any) -> None:
    """Make the orchestrator's legacy JDT gate consume the real v2 receipt shape.

    JavaLanguageService v2 returns URI -> diagnostics. The orchestrator historically
    iterated that mapping as though it were a list, so real compiler errors became an
    empty list. Its inline predicate also used severity<=2, which incorrectly treated
    warnings as blocking errors. This adapter gives that legacy gate only severity=1
    diagnostics while preserving the complete URI mapping separately for evidence.
    """

    current = orchestrator_module.JavaLanguageService
    if getattr(current, "_mmm_orchestrator_jdt_gate", False):
        return

    class OrchestratorJavaLanguageService(current):
        def diagnostics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            receipt = super().diagnostics(*args, **kwargs)
            if not isinstance(receipt, dict):
                return receipt
            raw = receipt.get("diagnostics")
            result = dict(receipt)
            if isinstance(raw, Mapping):
                result["diagnostics_by_uri"] = {
                    str(uri): [
                        dict(item)
                        for item in values
                        if isinstance(item, Mapping)
                    ]
                    for uri, values in raw.items()
                    if isinstance(values, list)
                }
            result["diagnostics"] = _blocking_errors(receipt)
            return result

    OrchestratorJavaLanguageService.__name__ = current.__name__
    OrchestratorJavaLanguageService.__qualname__ = current.__qualname__
    OrchestratorJavaLanguageService._mmm_orchestrator_jdt_gate = True
    OrchestratorJavaLanguageService.__wrapped__ = current
    orchestrator_module.JavaLanguageService = OrchestratorJavaLanguageService


__all__ = ["install"]
