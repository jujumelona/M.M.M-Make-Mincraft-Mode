"""Read a host-published coherent bundle catalog; no component discovery fallback."""

import json
from functools import lru_cache
import os
from pathlib import Path

from .resolved_version_context import ResolvedVersionContext, VersionContextError, VersionRequest, VersionResolver

DEFAULT_CATALOG = Path(__file__).with_name("data") / "host_version_catalog.json"


def load_host_catalog():
    location = os.environ.get("MMM_VERSION_BUNDLE_CATALOG", "").strip() or DEFAULT_CATALOG
    try:
        bundles, auto_context_id = _parse_catalog(Path(location).read_bytes())
    except (OSError, ValueError) as exc:
        raise VersionContextError("HOST_BUNDLE_CATALOG_INVALID", reason=str(exc)) from exc
    return VersionResolver(bundles, auto_context_id=auto_context_id), bundles


@lru_cache(maxsize=2)
def _parse_catalog(content: bytes):
    # Cache by complete bytes, never by timestamps: edits always revalidate.
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"schema_version", "auto_context_id", "bundles"}:
        raise VersionContextError("HOST_BUNDLE_CATALOG_INVALID")
    if value["schema_version"] != "mmm/host-version-catalog-v1" or not isinstance(value["bundles"], list):
        raise VersionContextError("HOST_BUNDLE_CATALOG_INVALID")
    bundles = tuple(ResolvedVersionContext.from_dict(bundle) for bundle in value["bundles"])
    return bundles, value["auto_context_id"]


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
    """Structural completeness audit for every published host bundle."""
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
    """Check every required and admitted leaf against actual execution records."""
    from .product_support_matrix import REQUIRED_CANONICAL_LEAVES, SUPPORTED_MINECRAFT_VERSIONS
    from .integrity_bootstrap import bootstrap_integrity
    from .integrity_evidence import binding_expectations, verify_execution_evidence
    from .evidence_store import get_global_evidence_store
    authority = bootstrap_integrity()
    authority.verify_live()
    _, bundles = load_host_catalog()
    store = get_global_evidence_store()
    scope = set(REQUIRED_CANONICAL_LEAVES if supported_scope is None else supported_scope)
    if not scope:
        raise ValueError("PRODUCTION_SCOPE_EMPTY")
    failures = []
    available = {ctx.minecraft for ctx in bundles}
    for version in set(SUPPORTED_MINECRAFT_VERSIONS) - available:
        failures.append({"minecraft": version, "issue": "SUPPORTED_VERSION_MISSING"})
    for ctx in bundles:
        bindings = ctx.to_dict()["host_facts"]["leaf_bindings"]
        required = scope if ctx.minecraft in SUPPORTED_MINECRAFT_VERSIONS or supported_scope is not None else set()
        leaves = required | {leaf for leaf, row in bindings.items() if row["state"] == "admitted"}
        for leaf in sorted(leaves):
            try:
                row = bindings[leaf]
                if row["state"] != "admitted":
                    raise ValueError("NOT_ADMITTED")
                impl = row["implementation"]
                if impl["authority_sha256"] != authority.content_hash:
                    raise ValueError("AUTHORITY_HASH_MISMATCH")
                verify_execution_evidence(store, impl["evidence_id"], expected=binding_expectations(leaf, impl, ctx.to_dict()["target"]))
            except (KeyError, ValueError, OSError) as exc:
                failures.append({"minecraft": ctx.minecraft, "context": ctx.context_id, "leaf": leaf, "issue": str(exc)})
    return {"schema_version": "mmm/host-production-readiness-audit-v1", "bundles_evaluated": len(bundles),
            "scope_size": len(scope), "status": "FAIL" if failures else "PASS", "failures": failures}