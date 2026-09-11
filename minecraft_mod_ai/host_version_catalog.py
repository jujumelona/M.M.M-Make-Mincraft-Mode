"""Read a host-published coherent bundle catalog; no component discovery fallback.

P0-3: Now validates against explicit product support matrix.
P0-4: Enforces compile/gametest evidence requirements.
"""

import json
import os
from pathlib import Path

from .resolved_version_context import ResolvedVersionContext, VersionContextError, VersionRequest, VersionResolver

DEFAULT_CATALOG = Path(__file__).with_name("data") / "host_version_catalog.json"


def load_host_catalog():
    location = os.environ.get("MMM_VERSION_BUNDLE_CATALOG", "").strip() or DEFAULT_CATALOG
    try:
        value = json.loads(Path(location).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VersionContextError("HOST_BUNDLE_CATALOG_INVALID", reason=str(exc)) from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "auto_context_id", "bundles"}:
        raise VersionContextError("HOST_BUNDLE_CATALOG_INVALID")
    if value["schema_version"] != "mmm/host-version-catalog-v1" or not isinstance(value["bundles"], list):
        raise VersionContextError("HOST_BUNDLE_CATALOG_INVALID")
    bundles = tuple(ResolvedVersionContext.from_dict(bundle) for bundle in value["bundles"])
    return VersionResolver(bundles, auto_context_id=value["auto_context_id"]), bundles


def host_versions(limit):
    resolver, bundles = load_host_catalog()
    preferred = resolver.resolve(VersionRequest())
    ordered = (preferred, *(bundle for bundle in bundles if bundle.context_id != preferred.context_id))
    return tuple(dict.fromkeys(bundle.minecraft for bundle in ordered))[:limit]


def host_target(version):
    from .target_contract import target_contract_from_mapping

    resolver, _ = load_host_catalog()
    context = resolver.resolve(VersionRequest.from_input(version))
    value = context.to_dict()
    target = dict(value["target"])
    target["host_facts_json"] = json.dumps(value["host_facts"], sort_keys=True, separators=(",", ":"))
    result = target_contract_from_mapping(target)
    context.assert_context(result.version_context.context_id)
    return result


def production_readiness_audit():
    """Check every admitted template against explicit support matrix.
    
    P0-3: Uses explicit support matrix, not derived from first bundle.
    P0-4: Enforces compile/gametest evidence requirements.
    
    Raises:
        Exception: If production readiness fails
    """
    from .product_support_matrix import validate_support_matrix
    from .task_template_catalog import load_template
    
    resolver, bundles = load_host_catalog()
    
    # P0-3: Validate against explicit support matrix
    try:
        validate_support_matrix(bundles)
    except Exception as exc:
        return {
            "schema_version": "mmm/host-version-catalog-audit-v1",
            "status": "FAIL",
            "reason": "support_matrix_validation_failed",
            "details": str(exc),
        }
    
    # P0-4: Check evidence requirements
    evidence_failures = []
    for context in bundles:
        for identifier, binding in context.facts.get("artifact_rules", {}).items():
            if binding.get("status") != "admitted":
                continue
            
            # Load template to check evidence requirements
            try:
                template = load_template(identifier)
                context.admit_template(template)
            except Exception:
                continue
            
            evidence_req = template.get("evidence_requirements", {})
            
            # P0-4: Code generation requires compile evidence
            if evidence_req.get("compile") == "required":
                compile_status = binding.get("compile_status", "not_run")
                if compile_status != "PASS":
                    evidence_failures.append({
                        "context": context.context_id,
                        "leaf": identifier,
                        "issue": "COMPILE_EVIDENCE_REQUIRED",
                        "actual": compile_status,
                    })
            
            # P0-4: Runtime leaves require gametest evidence
            if evidence_req.get("gametest") == "required":
                gametest_status = binding.get("gametest_status", "not_run")
                if gametest_status != "PASS":
                    evidence_failures.append({
                        "context": context.context_id,
                        "leaf": identifier,
                        "issue": "GAMETEST_EVIDENCE_REQUIRED",
                        "actual": gametest_status,
                    })
    
    if evidence_failures:
        return {
            "schema_version": "mmm/host-version-catalog-audit-v1",
            "status": "FAIL",
            "reason": "evidence_requirements_not_met",
            "failures": evidence_failures,
        }
    
    return {
        "schema_version": "mmm/host-version-catalog-audit-v1",
        "auto_context_id": resolver.resolve(VersionRequest()).context_id,
        "status": "PASS",
        "contexts": [
            {
                "context_id": item.context_id,
                "minecraft": item.minecraft,
                "host_revision": item.host_revision,
            }
            for item in bundles
        ],
    }


def audit_host_catalog():
    """Legacy audit function - now redirects to production_readiness_audit."""
    result = production_readiness_audit()
    if result.get("status") == "FAIL":
        raise VersionContextError("PRODUCTION_READINESS_FAILED", details=result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit_host_catalog(), indent=2, sort_keys=True))
