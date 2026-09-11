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
    """Production readiness audit: enforce zero unreviewed leaves and valid implementations across supported scope."""
    from .task_template_catalog import load_template

    resolver, bundles = load_host_catalog()
    if supported_scope is None:
        # Default supported scope: all admitted leaves in the first bundle
        supported_scope = [
            leaf
            for leaf, binding in bundles[0].facts.get("leaf_bindings", {}).items()
            if binding.get("state") == "admitted"
        ]
    scope_set = set(supported_scope)

    for context in bundles:
        bindings = context.facts.get("leaf_bindings", {})
        unreviewed = [leaf for leaf in scope_set if bindings.get(leaf, {}).get("state") == "not_reviewed"]
        if unreviewed:
            raise VersionContextError(
                "PRODUCTION_AUDIT_FAILED",
                unreviewed=sorted(unreviewed),
                context_id=context.context_id,
                minecraft=context.minecraft,
            )
        for leaf in scope_set:
            binding = bindings.get(leaf)
            if not binding:
                raise VersionContextError("HOST_LEAF_UNAVAILABLE", leaf=leaf, minecraft=context.minecraft)
            state = binding.get("state")
            if state == "admitted":
                impl = binding.get("implementation", {})
                template_id = impl.get("template")
                if template_id and template_id in context.facts["artifact_rules"]:
                    context.admit_template(load_template(template_id))

    return {
        "schema_version": "mmm/host-production-readiness-audit-v1",
        "bundles_evaluated": len(bundles),
        "scope_size": len(scope_set),
        "status": "PASS",
    }


if __name__ == "__main__":
    print(json.dumps(audit_host_catalog(), indent=2, sort_keys=True))

