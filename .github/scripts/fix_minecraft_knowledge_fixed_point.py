from __future__ import annotations

from pathlib import Path


SOURCE = Path("minecraft_mod_ai/minecraft_knowledge_contract.py")
TEST = Path("tests/test_minecraft_knowledge_contract.py")


old = '''    receipts, blocking = [], []
    for domain_id, domain in expected.items():
        executed = forced_map.get(domain_id)
        raw_queries = executed.get("queries", []) if isinstance(executed, Mapping) else []
        got = {str(x.get("query_sha256", "")) for x in raw_queries if isinstance(x, Mapping)}
        missing = sorted({_sha(str(q)) for q in domain["queries"]} - got)
        note = notes.get(domain_id)
        if domain_id not in brief_ids:
            status = "MISSING_RESEARCH_DOMAIN"
        elif executed is None:
            status = "MISSING_FORCED_RAG_RECEIPT"
        elif missing:
            status = "MISSING_FORCED_RAG_QUERY"
        elif note is None:
            status = "MISSING_RESEARCH_AGENT_NOTE"
        elif not bool(note.get("sufficient")):
            status = "RESEARCH_UNRESOLVED"
        else:
            status = "ROUTES_EXECUTED"
        if status != "ROUTES_EXECUTED":
            blocking.extend(str(ref) for ref in domain["requirements"])
        receipts.append(
            {
                "domain_id": domain_id,
                "status": status,
                "query_count": len(domain["queries"]),
                "forced_query_count": len(raw_queries),
                "missing_query_sha256": missing,
                "research_agent_sufficient": bool(note.get("sufficient")) if isinstance(note, Mapping) else False,
                "research_agent_fixed_point": bool(note.get("fixed_point")) if isinstance(note, Mapping) else False,
            }
        )
    result = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "status": "PASS" if not blocking else "BLOCK",
        "blocking_requirement_refs": sorted(set(blocking)),
        "domains": receipts,
        "semantics": (
            "PASS proves every host-required domain entered the brief, every deterministic forced-RAG query "
            "has an execution receipt, and every domain research agent marked it sufficient. It does not "
            "falsely claim optional/exact MCP lookups ran unless the research agent needed and used them."
        ),
        "coverage_sha256": "",
    }
'''

new = '''    receipts, blocking, deferred = [], [], []
    for domain_id, domain in expected.items():
        executed = forced_map.get(domain_id)
        raw_queries = executed.get("queries", []) if isinstance(executed, Mapping) else []
        got = {str(x.get("query_sha256", "")) for x in raw_queries if isinstance(x, Mapping)}
        missing = sorted({_sha(str(q)) for q in domain["queries"]} - got)
        note = notes.get(domain_id)
        if domain_id not in brief_ids:
            status = "MISSING_RESEARCH_DOMAIN"
        elif executed is None:
            status = "MISSING_FORCED_RAG_RECEIPT"
        elif missing:
            status = "MISSING_FORCED_RAG_QUERY"
        elif note is None:
            status = "MISSING_RESEARCH_AGENT_NOTE"
        elif bool(note.get("sufficient")):
            status = "ROUTES_EXECUTED"
        elif bool(note.get("fixed_point")):
            # A fixed point is a terminal research outcome, not evidence that the
            # required route failed to execute. Preserve the gaps for downstream
            # exact lookup/validation instead of deadlocking pre-design planning.
            status = "ROUTES_EXECUTED_WITH_GAPS"
        else:
            status = "RESEARCH_UNRESOLVED"
        if status in {
            "MISSING_RESEARCH_DOMAIN",
            "MISSING_FORCED_RAG_RECEIPT",
            "MISSING_FORCED_RAG_QUERY",
            "MISSING_RESEARCH_AGENT_NOTE",
            "RESEARCH_UNRESOLVED",
        }:
            blocking.extend(str(ref) for ref in domain["requirements"])
        elif status == "ROUTES_EXECUTED_WITH_GAPS":
            deferred.extend(str(ref) for ref in domain["requirements"])
        receipts.append(
            {
                "domain_id": domain_id,
                "status": status,
                "query_count": len(domain["queries"]),
                "forced_query_count": len(raw_queries),
                "missing_query_sha256": missing,
                "research_agent_sufficient": bool(note.get("sufficient")) if isinstance(note, Mapping) else False,
                "research_agent_fixed_point": bool(note.get("fixed_point")) if isinstance(note, Mapping) else False,
            }
        )
    result = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "status": "PASS" if not blocking else "BLOCK",
        "blocking_requirement_refs": sorted(set(blocking)),
        "deferred_requirement_refs": sorted(set(deferred)),
        "domains": receipts,
        "semantics": (
            "PASS proves every host-required domain entered the brief, every deterministic forced-RAG query "
            "has an execution receipt, and every domain research agent produced a terminal note. A terminal "
            "fixed point may retain explicit deferred gaps for downstream exact lookup/validation; PASS does "
            "not claim those gaps are resolved or that optional MCP lookups ran."
        ),
        "coverage_sha256": "",
    }
'''

source = SOURCE.read_text(encoding="utf-8")
if old not in source:
    raise SystemExit("minecraft knowledge coverage target not found")
source = source.replace(old, new, 1)
compile(source, str(SOURCE), "exec")
SOURCE.write_text(source, encoding="utf-8")

tests = TEST.read_text(encoding="utf-8")
test_name = "test_route_coverage_accepts_terminal_fixed_point_with_deferred_gaps"
if test_name not in tests:
    tests += '''


def test_route_coverage_accepts_terminal_fixed_point_with_deferred_gaps() -> None:
    plan = compile_minecraft_knowledge_plan("새 보스 몬스터를 추가해줘.")
    research = _fake_research(plan)
    domain = plan["research_domains"][0]
    note = next(
        item for item in research["domain_notes"]
        if item["domain_id"] == domain["domain_id"]
    )
    note["sufficient"] = False
    note["fixed_point"] = True

    coverage = evaluate_route_coverage(plan, research)

    assert coverage["status"] == "PASS"
    assert not coverage["blocking_requirement_refs"]
    assert set(domain["requirements"]) <= set(coverage["deferred_requirement_refs"])
    receipt = next(
        item for item in coverage["domains"]
        if item["domain_id"] == domain["domain_id"]
    )
    assert receipt["status"] == "ROUTES_EXECUTED_WITH_GAPS"
    assert receipt["research_agent_sufficient"] is False
    assert receipt["research_agent_fixed_point"] is True
'''
    TEST.write_text(tests, encoding="utf-8")
