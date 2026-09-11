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


def audit_host_catalog():
    """Check every admitted template against this checkout before using a catalog."""
    from .task_template_catalog import load_template

    resolver, bundles = load_host_catalog()
    for context in bundles:
        for identifier in context.facts["artifact_rules"]:
            context.admit_template(load_template(identifier))
    return {"schema_version": "mmm/host-version-catalog-audit-v1",
            "auto_context_id": resolver.resolve(VersionRequest()).context_id,
            "contexts": [{"context_id": item.context_id, "minecraft": item.minecraft,
                          "host_revision": item.host_revision} for item in bundles]}


if __name__ == "__main__":
    print(json.dumps(audit_host_catalog(), indent=2, sort_keys=True))
