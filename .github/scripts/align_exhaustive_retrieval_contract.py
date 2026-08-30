from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path.cwd()
PD = ROOT / "minecraft_mod_ai" / "pre_design_research_pipeline.py"
EXHAUSTIVE_TEST = ROOT / "tests" / "test_exhaustive_retrieval_contract.py"
PIPELINE_TEST = ROOT / "tests" / "test_pre_design_research_pipeline.py"


def replace_top_level_function(path: Path, name: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = text.splitlines(keepends=True)
            lines[node.lineno - 1 : node.end_lineno] = [replacement.rstrip() + "\n\n"]
            path.write_text("".join(lines), encoding="utf-8")
            return
    raise SystemExit(f"top-level function not found: {path}:{name}")


# Official pre-design corpus selection is relevance-gated, not cardinality-gated.
pd = PD.read_text(encoding="utf-8")
old = '''            selected = [
                document
                for score, document in sorted(
                    scored,
                    key=lambda item: (-item[0], str(item[1].get("document_id", ""))),
                )
                if score > 0
            ][:4]
'''
new = '''            selected = [
                document
                for score, document in sorted(
                    scored,
                    key=lambda item: (-item[0], str(item[1].get("document_id", ""))),
                )
                if score > 0
            ]
'''
if old not in pd:
    raise SystemExit("official evidence ordinal cutoff pattern not found")
PD.write_text(pd.replace(old, new, 1), encoding="utf-8")


replace_top_level_function(
    EXHAUSTIVE_TEST,
    "test_source_document_keeps_complete_content",
    '''def test_source_document_keeps_complete_content():
    text = "complete source body whose terminal marker must remain: TAIL_MARKER"
    doc = rg._source_document(
        source_id="x",
        title="x",
        url="https://example.invalid/x",
        content=text,
        source_type="test",
    )
    assert doc["content"] == text
    assert doc["content"].endswith("TAIL_MARKER")''',
)


helper = '''\n\ndef _grounded_bundle(domain_id: str, query: str) -> dict:
    return {
        "schema_version": "mmm/forced-pre-design-rag",
        "research_sha256": "sha256:grounded-fixture",
        "domains": [
            {
                "domain_id": domain_id,
                "queries": [
                    {
                        "query": query,
                        "query_sha256": "sha256:query-fixture",
                        "code_rag": {
                            "documents": [
                                {
                                    "source_id": f"project:src/main/java/{domain_id}.java",
                                    "url": f"file:///workspace/src/main/java/{domain_id}.java",
                                    "content": (
                                        "Claim-bearing local source implementation evidence "
                                        "for target-neutral pre-design research."
                                    ),
                                }
                            ]
                        },
                    }
                ],
            }
        ],
    }
'''

pipeline_text = PIPELINE_TEST.read_text(encoding="utf-8")
if "def _grounded_bundle(" not in pipeline_text:
    marker = "\n\ndef test_pre_design_parallelizes_target_neutral_evidence_and_defers_target_radar(\n"
    if marker not in pipeline_text:
        raise SystemExit("pipeline test insertion marker not found")
    pipeline_text = pipeline_text.replace(marker, helper + marker, 1)
    PIPELINE_TEST.write_text(pipeline_text, encoding="utf-8")


replace_top_level_function(
    PIPELINE_TEST,
    "test_pre_design_parallelizes_target_neutral_evidence_and_defers_target_radar",
    '''def test_pre_design_uses_single_grounded_owner_and_defers_target_radar(monkeypatch) -> None:
    brief = {
        "domains": [
            {
                "domain_id": "fabric_api",
                "queries": ["Fabric API target-neutral behavior"],
                "providers": ["official_docs", "project_rag"],
            }
        ]
    }
    monkeypatch.setattr(pipeline, "_pre_design_brief", lambda _prompt: brief)
    monkeypatch.setattr(
        pipeline,
        "compile_minecraft_knowledge_plan",
        lambda _prompt: {
            "plan_sha256": "sha256:plan",
            "policy": {"target_frozen": False},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_route_coverage",
        lambda *_args, **_kwargs: {"status": "PASS", "blocking_requirement_refs": []},
    )

    owner_calls = []

    def forced(router, research_brief):
        owner_calls.append((router, research_brief))
        return _grounded_bundle("fabric_api", "Fabric API target-neutral behavior")

    def radar_must_not_run(*_args, **_kwargs):
        raise AssertionError("target-specific technology radar ran before target freeze")

    monkeypatch.setattr(project_rag, "_forced_rag_bundle", forced)
    monkeypatch.setattr(pipeline, "collect_technology_radar", radar_must_not_run)
    monkeypatch.setattr(
        project_rag,
        "_materialize_domain_evidence_document",
        lambda domain_id, evidence: {"domain_id": domain_id, "evidence": evidence},
    )
    monkeypatch.setattr(pipeline, "_validate_document_grounding", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "research_document_domain",
        lambda *_args, **_kwargs: {
            "domain_id": "fabric_api",
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "procedures": [],
            "sufficient": True,
            "fixed_point": True,
            "checkpoint": {"status": "complete"},
        },
    )
    monkeypatch.setattr(pipeline, "attach_procedural_skillbank", lambda _r, _p, value: value)
    monkeypatch.setattr(pipeline, "compose_research_skillbank", lambda _r, _p, value: value)

    payload = collect_design_research(object(), "build a Fabric mechanic")

    assert len(owner_calls) == 1
    assert set(payload["deterministic"]) == {"grounded_rag", "technology_radar"}
    assert payload["deterministic"]["technology_radar"]["status"] == "deferred_until_target_freeze"
    assert "official_rag" not in payload["deterministic"]
    assert "forced_project_rag" not in payload["deterministic"]
    assert payload["domain_notes"][0]["domain_id"] == "fabric_api"
    assert "grounded" in payload["method"]["planning_search"]''',
)


replace_top_level_function(
    PIPELINE_TEST,
    "test_terminal_gap_prints_full_failure_and_stops_before_post_research_work",
    '''def test_terminal_gap_prints_full_failure_and_stops_before_post_research_work(
    monkeypatch, capsys
) -> None:
    brief = {"domains": [{"domain_id": "request", "queries": ["request evidence"]}]}
    monkeypatch.setattr(pipeline, "_pre_design_brief", lambda _prompt: brief)
    monkeypatch.setattr(
        pipeline,
        "compile_minecraft_knowledge_plan",
        lambda _prompt: {"plan_sha256": "sha256:plan", "policy": {"target_frozen": False}},
    )
    monkeypatch.setattr(
        project_rag,
        "_forced_rag_bundle",
        lambda *_args, **_kwargs: _grounded_bundle("request", "request evidence"),
    )
    monkeypatch.setattr(
        project_rag,
        "_materialize_domain_evidence_document",
        lambda domain_id, evidence: {"domain_id": domain_id, "evidence": evidence},
    )
    monkeypatch.setattr(pipeline, "_validate_document_grounding", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "research_document_domain",
        lambda *_args, **_kwargs: {
            "domain_id": "request",
            "claims": [],
            "gaps": ["EXACT_SYNTHESIS_GAP"],
            "next_queries": [],
            "procedures": [],
            "sufficient": False,
            "checkpoint": {"status": "terminal_gap", "request_sha256": "sha256:failure"},
            "research_failures": [
                {
                    "unit": "synthesis:0:0",
                    "error": "EXACT_VALIDATOR_FAILURE: missing claim evidence",
                }
            ],
            "fixed_point": True,
        },
    )

    def must_not_continue(*_args, **_kwargs):
        raise AssertionError("post-research processing must not run after terminal_gap")

    monkeypatch.setattr(pipeline, "attach_procedural_skillbank", must_not_continue)
    monkeypatch.setattr(pipeline, "compose_research_skillbank", must_not_continue)

    with pytest.raises(PreDesignResearchFailure, match="terminal_gap"):
        collect_design_research(object(), "failing request")

    logged = capsys.readouterr().out
    assert "PRE-DESIGN RESEARCH DIAGNOSTIC:" in logged
    assert '\"event\": \"domain_result\"' in logged
    assert "terminal_gap" in logged
    assert "synthesis:0:0" in logged
    assert "EXACT_VALIDATOR_FAILURE: missing claim evidence" in logged
    assert "EXACT_SYNTHESIS_GAP" in logged''',
)


replace_top_level_function(
    PIPELINE_TEST,
    "test_domain_exception_prints_full_traceback_and_escapes",
    '''def test_domain_exception_prints_full_traceback_and_escapes(monkeypatch, capsys) -> None:
    brief = {"domains": [{"domain_id": "request", "queries": ["request evidence"]}]}
    monkeypatch.setattr(pipeline, "_pre_design_brief", lambda _prompt: brief)
    monkeypatch.setattr(
        pipeline,
        "compile_minecraft_knowledge_plan",
        lambda _prompt: {"plan_sha256": "sha256:plan", "policy": {"target_frozen": False}},
    )
    monkeypatch.setattr(
        project_rag,
        "_forced_rag_bundle",
        lambda *_args, **_kwargs: _grounded_bundle("request", "request evidence"),
    )
    monkeypatch.setattr(
        project_rag,
        "_materialize_domain_evidence_document",
        lambda domain_id, evidence: {"domain_id": domain_id, "evidence": evidence},
    )
    monkeypatch.setattr(pipeline, "_validate_document_grounding", lambda *args, **kwargs: None)

    def explode(*_args, **_kwargs):
        raise ValueError("EXACT_DOMAIN_EXCEPTION")

    monkeypatch.setattr(pipeline, "research_document_domain", explode)

    with pytest.raises(ValueError, match="EXACT_DOMAIN_EXCEPTION"):
        collect_design_research(object(), "failing request")

    logged = capsys.readouterr().out
    assert '\"event\": \"domain_execution_exception\"' in logged
    assert "EXACT_DOMAIN_EXCEPTION" in logged
    assert "ValueError" in logged
    assert "Traceback (most recent call last)" in logged''',
)

print("aligned exhaustive retrieval tests and official evidence contract")
