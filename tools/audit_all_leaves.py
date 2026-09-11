#!/usr/bin/env python3
"""Audit all 346 canonical leaves across all supported versions.

P0-9: Comprehensive leaf audit tool.
"""

import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from minecraft_mod_ai.structural_routing_contract import CANONICAL_ARTIFACT_KINDS
from minecraft_mod_ai.minecraft_template_steps import responsibility_ids_for_artifact
from minecraft_mod_ai.product_support_matrix import SUPPORTED_MINECRAFT_VERSIONS
from minecraft_mod_ai.host_version_catalog import load_host_catalog


def get_all_canonical_leaves() -> list[str]:
    """Get all 346 canonical leaves."""
    leaves = []
    for kind in CANONICAL_ARTIFACT_KINDS:
        leaves.extend(responsibility_ids_for_artifact(kind))
    return sorted(set(leaves))


def audit_leaf_status(leaf_id: str, minecraft_version: str, context) -> dict:
    """Audit a single leaf for a specific version."""
    bindings = context.facts.get("leaf_bindings", {})
    binding = bindings.get(leaf_id, {})
    
    state = binding.get("state", "not_reviewed")
    reason = binding.get("reason", "")
    
    impl = binding.get("implementation", {})
    template_id = impl.get("template", "")
    evidence_id = binding.get("evidence_id", "")
    
    return {
        "leaf": leaf_id,
        "version": minecraft_version,
        "context_id": context.context_id,
        "state": state,
        "reason": reason,
        "template_id": template_id,
        "evidence_id": evidence_id,
        "has_implementation": bool(impl),
    }


def main():
    """Run comprehensive leaf audit."""
    print("=" * 80)
    print("P0-9: Comprehensive Canonical Leaf Audit")
    print("=" * 80)
    print()
    
    # Get all leaves
    all_leaves = get_all_canonical_leaves()
    print(f"Total canonical leaves: {len(all_leaves)}")
    print()
    
    # Load catalog
    print("Loading HOST catalog...")
    try:
        resolver, bundles = load_host_catalog()
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return 1
    
    print(f"Loaded {len(bundles)} bundles")
    print()
    
    # Audit matrix
    audit_matrix = {}
    state_counts = {
        "admitted": 0,
        "unsupported": 0,
        "unsupported_by_target": 0,
        "unsupported_by_product": 0,
        "not_reviewed": 0,
        "implementation_missing": 0,
        "missing": 0,
    }
    
    print("Auditing leaf × version matrix...")
    for version in SUPPORTED_MINECRAFT_VERSIONS:
        # Find context for version
        context = None
        for bundle in bundles:
            if bundle.minecraft == version:
                context = bundle
                break
        
        if not context:
            print(f"⚠️  No bundle for {version}")
            continue
        
        version_results = {}
        for leaf_id in all_leaves:
            result = audit_leaf_status(leaf_id, version, context)
            version_results[leaf_id] = result
            
            state = result["state"]
            state_counts[state] = state_counts.get(state, 0) + 1
        
        audit_matrix[version] = version_results
    
    # Print summary
    print()
    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print()
    print(f"Total leaves: {len(all_leaves)}")
    print(f"Total versions: {len(SUPPORTED_MINECRAFT_VERSIONS)}")
    print(f"Total combinations: {len(all_leaves) * len(SUPPORTED_MINECRAFT_VERSIONS)}")
    print()
    print("State Distribution:")
    for state, count in sorted(state_counts.items()):
        percentage = (count / (len(all_leaves) * len(SUPPORTED_MINECRAFT_VERSIONS)) * 100)
        print(f"  {state:30s}: {count:5d} ({percentage:5.1f}%)")
    print()
    
    # Find problem areas
    not_reviewed = []
    for version, results in audit_matrix.items():
        for leaf_id, result in results.items():
            if result["state"] == "not_reviewed":
                not_reviewed.append((version, leaf_id))
    
    if not_reviewed:
        print("=" * 80)
        print(f"⚠️  NOT_REVIEWED LEAVES: {len(not_reviewed)}")
        print("=" * 80)
        print()
        print("These must be audited before production:")
        for version, leaf_id in not_reviewed[:20]:  # Show first 20
            print(f"  {version:10s} × {leaf_id}")
        if len(not_reviewed) > 20:
            print(f"  ... and {len(not_reviewed) - 20} more")
        print()
    
    # Production readiness
    production_ready = state_counts.get("not_reviewed", 0) == 0
    
    print("=" * 80)
    if production_ready:
        print("✅ PRODUCTION READY: All leaves audited")
    else:
        print(f"❌ NOT PRODUCTION READY: {state_counts.get('not_reviewed', 0)} leaves not reviewed")
    print("=" * 80)
    print()
    
    # Save detailed report
    output_file = Path("leaf_audit_report.json")
    report = {
        "total_leaves": len(all_leaves),
        "total_versions": len(SUPPORTED_MINECRAFT_VERSIONS),
        "state_counts": state_counts,
        "production_ready": production_ready,
        "not_reviewed_count": len(not_reviewed),
        "audit_matrix": audit_matrix,
    }
    
    output_file.write_text(json.dumps(report, indent=2))
    print(f"📄 Detailed report saved to: {output_file}")
    print()
    
    return 0 if production_ready else 1


if __name__ == "__main__":
    sys.exit(main())
