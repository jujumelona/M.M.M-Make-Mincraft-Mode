from pathlib import Path

planner_path = Path('minecraft_mod_ai/reuse_planner.py')
planner = planner_path.read_text(encoding='utf-8')

old_register = '''    def register_search_terms(capability: str, values: Iterable[Any]) -> None:\n        if not capability:\n            return\n        bucket = search_terms.setdefault(capability, [])\n        for predefined in search_queries_for_capability(capability):\n            if predefined.casefold() not in {item.casefold() for item in bucket}:\n                bucket.append(predefined)\n        for raw in values:\n            value = " ".join(str(raw or "").split())\n            if value and value.casefold() not in {item.casefold() for item in bucket}:\n                bucket.append(value[:512])\n'''
new_register = '''    def register_search_terms(\n        capability: str,\n        values: Iterable[Any],\n        *,\n        include_predefined: bool = True,\n    ) -> None:\n        if not capability:\n            return\n        bucket = search_terms.setdefault(capability, [])\n        if include_predefined:\n            for predefined in search_queries_for_capability(capability):\n                if predefined.casefold() not in {item.casefold() for item in bucket}:\n                    bucket.append(predefined)\n        for raw in values:\n            value = " ".join(str(raw or "").split())\n            if value and value.casefold() not in {item.casefold() for item in bucket}:\n                bucket.append(value[:512])\n'''
if planner.count(old_register) != 1:
    raise SystemExit('reuse search-term helper drifted')
planner = planner.replace(old_register, new_register, 1)

start = planner.index('    catalog_used = False\n')
end = planner.index('\n    if not catalog_used and isinstance(design, Mapping):', start)
new_catalog = '''    catalog_used = False\n    from .requirement_catalog import build_requirement_catalog\n    ev_catalog = design.get("_evidence_request_catalog") if isinstance(design, Mapping) else None\n    if ev_catalog and isinstance(ev_catalog.get("requirements"), Sequence):\n        req_catalog = build_requirement_catalog(prompt, evidence_request_catalog=ev_catalog)\n        capabilities_by_requirement: dict[str, tuple[str, ...]] = {}\n        for req in req_catalog.requirements:\n            source = f"evidence_request_catalog.{req.id}"\n            bound: list[str] = []\n            semantic = " ".join(str(req.normalized_statement or req.statement).split())\n            original = " ".join(str(req.original_span or req.statement).split())\n            for cap in req.provides:\n                # An approved evidence catalog is the semantic authority. Never\n                # re-expand or rename its capability identity through the ontology.\n                anchor = add(cap, source, expand=False)\n                if not anchor:\n                    continue\n                cap_words = anchor.replace(".", " ").replace("_", " ")\n                register_search_terms(\n                    anchor,\n                    (\n                        f"{cap_words} implementation {semantic}",\n                        f"{cap_words} source {original}",\n                        f"{cap_words} reusable implementation",\n                    ),\n                    include_predefined=False,\n                )\n                bound.append(anchor)\n                catalog_used = True\n            capabilities_by_requirement[req.id] = tuple(dict.fromkeys(bound))\n\n        if not catalog_used and req_catalog.capabilities:\n            for c_spec in req_catalog.capabilities:\n                anchor = add(c_spec.id, "evidence_request_catalog.capability", expand=False)\n                if anchor:\n                    cap_words = anchor.replace(".", " ").replace("_", " ")\n                    register_search_terms(\n                        anchor,\n                        (\n                            f"{cap_words} reusable implementation",\n                            f"{cap_words} source code",\n                            f"{cap_words} minecraft implementation",\n                        ),\n                        include_predefined=False,\n                    )\n                    catalog_used = True\n\n        if catalog_used:\n            # Preserve only authored gameplay dependencies. Semantic derivation is\n            # provenance, not an implementation dependency, and ontology defaults\n            # are forbidden after requirement approval.\n            raw_requirements = ev_catalog.get("requirements")\n            if isinstance(raw_requirements, Sequence) and not isinstance(\n                raw_requirements, (str, bytes, bytearray)\n            ):\n                for index, raw_requirement in enumerate(raw_requirements, 1):\n                    if not isinstance(raw_requirement, Mapping):\n                        continue\n                    child_id = str(\n                        raw_requirement.get("requirement_id")\n                        or raw_requirement.get("id")\n                        or f"REQ-{index:03d}"\n                    )\n                    raw_dependencies = raw_requirement.get("depends_on", ())\n                    if not isinstance(raw_dependencies, Sequence) or isinstance(\n                        raw_dependencies, (str, bytes, bytearray)\n                    ):\n                        continue\n                    for parent_id in raw_dependencies:\n                        for child_cap in capabilities_by_requirement.get(child_id, ()):\n                            for parent_cap in capabilities_by_requirement.get(str(parent_id), ()):\n                                if child_cap != parent_cap and (child_cap, parent_cap) not in edges:\n                                    edges.append((child_cap, parent_cap))\n\n            # Hard authority barrier: no raw-prompt resolver, semantic inference,\n            # opaque fallback node, predefined ontology search query, or default\n            # dependency may execute below this point.\n            return CapabilityGraph(\n                nodes=tuple(ordered),\n                edges=tuple(edges),\n                sources=tuple((node, sources[node]) for node in ordered),\n                search_terms=tuple(\n                    (node, tuple(search_terms.get(node, (node.replace(".", " "),))))\n                    for node in ordered\n                ),\n            )\n'''
planner = planner[:start] + new_catalog + planner[end:]
planner_path.write_text(planner, encoding='utf-8')

catalog_path = Path('minecraft_mod_ai/requirement_catalog.py')
catalog = catalog_path.read_text(encoding='utf-8')
old_normalized = '                normalized_statement=stmt,\n'
new_normalized = '                normalized_statement=(str(r.get("semantic_statement") or stmt).strip() or stmt),\n'
if catalog.count(old_normalized) != 1:
    raise SystemExit('authoritative normalized statement target drifted')
catalog = catalog.replace(old_normalized, new_normalized, 1)
catalog_path.write_text(catalog, encoding='utf-8')

test_path = Path('tests/test_reuse_authoritative_search.py')
test_path.write_text(r'''from __future__ import annotations

from minecraft_mod_ai.requirement_catalog import build_requirement_catalog
from minecraft_mod_ai.reuse_planner import decompose_capability_graph


class _ExplodingSemanticRouter:
    def generate_text(self, *args, **kwargs):
        raise AssertionError("raw prompt semantic inference ran after catalog approval")

    def generate_tool_decision(self, *args, **kwargs):
        raise AssertionError("raw prompt semantic inference ran after catalog approval")


def _catalog():
    return {
        "requirements": [
            {
                "requirement_id": "req_collect",
                "capability": "resource.gathering",
                "provides": ["capability:resource.gathering"],
                "statement": "Broad authored text that must not own donor search semantics.",
                "semantic_statement": "collect luminous shards from world nodes",
                "source_span": {"text": "collect luminous shards"},
                "mandatory": True,
                "depends_on": [],
            },
            {
                "requirement_id": "req_exchange",
                "capability": "economy.exchange",
                "provides": ["capability:economy.exchange"],
                "statement": "Another broad authored sentence.",
                "semantic_statement": "exchange gathered shards for progression value",
                "source_span": {"text": "exchange gathered shards"},
                "mandatory": True,
                "depends_on": ["req_collect"],
            },
        ]
    }


def test_approved_catalog_is_hard_authority_barrier_for_reuse_search():
    graph = decompose_capability_graph(
        "NEVER_SEARCH_THIS_THEME arbitrary unrelated prompt wording",
        design={"_evidence_request_catalog": _catalog()},
        semantic_router=_ExplodingSemanticRouter(),
    )

    assert graph.nodes == ("resource.gathering", "economy.exchange")
    assert graph.edges == (("economy.exchange", "resource.gathering"),)
    assert all(not node.startswith("provisional:semantic_") for node in graph.nodes)

    terms = dict(graph.search_terms)
    flattened = " ".join(term for values in terms.values() for term in values)
    assert "NEVER_SEARCH_THIS_THEME" not in flattened
    assert "collect luminous shards from world nodes" in flattened
    assert "exchange gathered shards for progression value" in flattened
    assert all(
        capability.replace(".", " ") in " ".join(values)
        for capability, values in terms.items()
    )


def test_authoritative_requirement_catalog_prefers_semantic_statement():
    catalog = build_requirement_catalog(
        "raw prompt is not the normalized search meaning",
        evidence_request_catalog=_catalog(),
    )
    by_id = {item.id: item for item in catalog.requirements}
    assert by_id["req_collect"].normalized_statement == "collect luminous shards from world nodes"
    assert by_id["req_exchange"].normalized_statement == "exchange gathered shards for progression value"
''', encoding='utf-8')
