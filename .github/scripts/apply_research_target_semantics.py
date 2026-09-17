from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_central_research() -> None:
    path = ROOT / "minecraft_mod_ai/central_research.py"
    replace_once(
        path,
        '''    raw_target = research_brief.get("_mmm_platform_target")\n    try:\n        target = _canonical_platform_target(raw_target) if raw_target is not None else None\n    except SpecValidationError:\n        # Missing/partial/stale platform metadata falls back to target-neutral retrieval.\n        target = None\n''',
        '''    raw_target = research_brief.get("_mmm_platform_target")\n    target = _canonical_platform_target(raw_target) if raw_target is not None else None\n''',
        "strict selected platform target",
    )
    replace_once(
        path,
        '''        if "official_docs" not in domain.providers:\n            results.append(\n                {\n                    "domain_id": domain.domain_id,\n                    "strategy": "routed_to_other_providers",\n                    "queries": [],\n                }\n            )\n            continue\n        query_results: list[dict[str, Any]] = []\n''',
        '''        if "official_docs" not in domain.providers:\n            results.append(\n                {\n                    "domain_id": domain.domain_id,\n                    "strategy": "routed_to_other_providers",\n                    "queries": [],\n                }\n            )\n            continue\n        if target is None:\n            deferred.append(domain.domain_id)\n            results.append(\n                {\n                    "domain_id": domain.domain_id,\n                    "strategy": "deferred_until_platform_selected",\n                    "queries": [],\n                }\n            )\n            continue\n        query_results: list[dict[str, Any]] = []\n''',
        "defer untargeted official research",
    )


def patch_parallel_runtime() -> None:
    path = ROOT / "minecraft_mod_ai/parallel_runtime_contract.py"
    replace_once(
        path,
        '''        # Platform targeting is an optional refinement. Generic Official RAG remains\n        # active when the target is missing, partial, stale, or otherwise non-executable.\n        try:\n            adapter, verified_domains = _require_parallel_research_contract(\n                central_module,\n                research_brief,\n            )\n        except ParallelResearchContractError:\n            return build_research_graph(\n                research_brief,\n                retrieve=selected_retrieve,\n            )\n        domains = verified_domains\n''',
        '''        raw_target = research_brief.get("_mmm_platform_target")\n        if raw_target is None:\n            # Official docs are target-specific.  With no selected target the central\n            # graph records those domains as deferred while retaining non-official routes.\n            return build_research_graph(\n                research_brief,\n                retrieve=selected_retrieve,\n            )\n\n        # A target that exists must be complete and canonical.  Partial or stale\n        # metadata is a contract error, never a reason to silently run generic RAG.\n        adapter, verified_domains = _require_parallel_research_contract(\n            central_module,\n            research_brief,\n        )\n        domains = verified_domains\n''',
        "parallel target fail-closed semantics",
    )


def main() -> None:
    patch_central_research()
    patch_parallel_runtime()


if __name__ == "__main__":
    main()
