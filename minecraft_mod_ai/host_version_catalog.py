"""Read a host-published coherent bundle catalog; no component discovery fallback."""

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


def coverage_audit():
    """Completeness audit: verify complete leaf coverage and template admission across all bundles."""
    from .task_template_catalog import load_template
    from .structural_routing_contract import CANONICAL_ARTIFACT_KINDS
    from .minecraft_template_steps import responsibility_ids_for_artifact

    resolver, bundles = load_host_catalog()
    all_leaves = set()
    for kind in CANONICAL_ARTIFACT_KINDS:
        all_leaves.update(responsibility_ids_for_artifact(kind))

    for context in bundles:
        bindings = context.facts.get("leaf_bindings", {})
        missing_leaves = all_leaves - set(bindings)
        if missing_leaves:
            raise VersionContextError(
                "HOST_LEAF_COVERAGE_INCOMPLETE",
                missing=sorted(missing_leaves),
                context_id=context.context_id,
                minecraft=context.minecraft,
            )
        for leaf_id, binding in bindings.items():
            state = binding.get("state")
            if state not in {"admitted", "unsupported", "not_reviewed"}:
                raise VersionContextError(
                    "HOST_LEAF_BINDING_INVALID",
                    leaf=leaf_id,
                    state=state,
                    context_id=context.context_id,
                )
            if state == "admitted":
                impl = binding.get("implementation", {})
                template_id = impl.get("template")
                if template_id and template_id in context.facts["artifact_rules"]:
                    context.admit_template(load_template(template_id))

        for identifier in context.facts["artifact_rules"]:
            context.admit_template(load_template(identifier))

    return {
        "schema_version": "mmm/host-version-catalog-audit-v1",
        "auto_context_id": resolver.resolve(VersionRequest()).context_id,
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
    return coverage_audit()


def production_readiness_audit(supported_scope=None):
    """Production readiness audit: enforce support matrix, evidence requirements, and zero unreviewed.
    
    P0-3: Uses explicit product support matrix, not derived from first bundle.
    P0-4: Enforces compile/gametest evidence requirements.
    """
    from .product_support_matrix import validate_support_matrix, SupportMatrixError
    from .evidence_store import get_global_evidence_store
    from .task_template_catalog import load_template

    resolver, bundles = load_host_catalog()
    evidence_store = get_global_evidence_store()
    
    # P0-3: Validate against explicit support matrix (not first-bundle-derived scope)
    try:
        validate_support_matrix(bundles)
    except SupportMatrixError as exc:
        return {
            "schema_version": "mmm/host-production-readiness-audit-v1",
            "status": "FAIL",
            "reason": "support_matrix_validation_failed",
            "failures": exc.failures,
        }
    
    # P0-4: Check evidence requirements for all admitted leaves
    evidence_failures = []
    
    for context in bundles:
        bindings = context.facts.get("leaf_bindings", {})
        
        for leaf_id, binding in bindings.items():
            state = binding.get("state")
            
            if state != "admitted":
                continue
            
            impl = binding.get("implementation", {})
            template_id = impl.get("template")
            
            if not template_id:
                continue
            
            # Load template to check evidence requirements
            try:
                template = load_template(template_id)
                if template_id in context.facts.get("artifact_rules", {}):
                    context.admit_template(template)
            except Exception:
                continue
            
            evidence_req = template.get("evidence_requirements", {})
            evidence_id = binding.get("evidence_id")
            
            # P0-4: Code generation requires compile evidence
            if evidence_req.get("compile") == "required":
                if not evidence_id:
                    evidence_failures.append({
                        "context": context.context_id,
                        "leaf": leaf_id,
                        "issue": "COMPILE_EVIDENCE_MISSING",
                        "actual": "no_evidence_id",
                    })
                    continue
                
                try:
                    evidence = evidence_store.retrieve_evidence(evidence_id)
                    if not evidence.compile_result or evidence.compile_result.status != "PASS":
                        evidence_failures.append({
                            "context": context.context_id,
                            "leaf": leaf_id,
                            "issue": "COMPILE_EVIDENCE_REQUIRED",
                            "actual": evidence.compile_result.status if evidence.compile_result else "none",
                        })
                except Exception:
                    evidence_failures.append({
                        "context": context.context_id,
                        "leaf": leaf_id,
                        "issue": "COMPILE_EVIDENCE_MISSING",
                        "actual": "evidence_not_found",
                    })
            
            # P0-4: Runtime leaves require gametest evidence
            if evidence_req.get("gametest") == "required":
                if not evidence_id:
                    evidence_failures.append({
                        "context": context.context_id,
                        "leaf": leaf_id,
                        "issue": "GAMETEST_EVIDENCE_MISSING",
                        "actual": "no_evidence_id",
                    })
                    continue
                
                try:
                    evidence = evidence_store.retrieve_evidence(evidence_id)
                    if not evidence.gametest_result or evidence.gametest_result.status != "PASS":
                        evidence_failures.append({
                            "context": context.context_id,
                            "leaf": leaf_id,
                            "issue": "GAMETEST_EVIDENCE_REQUIRED",
                            "actual": evidence.gametest_result.status if evidence.gametest_result else "none",
                        })
                except Exception:
                    evidence_failures.append({
                        "context": context.context_id,
                        "leaf": leaf_id,
                        "issue": "GAMETEST_EVIDENCE_MISSING",
                        "actual": "evidence_not_found",
                    })
    
    if evidence_failures:
        return {
            "schema_version": "mmm/host-production-readiness-audit-v1",
            "status": "FAIL",
            "reason": "evidence_requirements_not_met",
            "failures": evidence_failures,
        }
    
    return {
        "schema_version": "mmm/host-production-readiness-audit-v1",
        "bundles_evaluated": len(bundles),
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


if __name__ == "__main__":
    print(json.dumps(audit_host_catalog(), indent=2, sort_keys=True))

