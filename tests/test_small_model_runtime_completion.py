from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from minecraft_mod_ai import model_router
from minecraft_mod_ai.production_tools import ProductionToolService
from minecraft_mod_ai.small_model_context_compaction import compact_messages
from minecraft_mod_ai.small_model_rag_relations import derive_relations


def test_relation_extractor_finds_java_and_resource_edges(tmp_path: Path) -> None:
    root = tmp_path / "project"
    java = root / "src/main/java/demo"
    resources = root / "src/main/resources/assets/demo/models/item"
    java.mkdir(parents=True)
    resources.mkdir(parents=True)
    target = java / "Target.java"
    target.write_text(
        "package demo;\npublic final class Target {}\n",
        encoding="utf-8",
    )
    caller = java / "Caller.java"
    caller.write_text(
        'package demo;\nimport demo.Target;\nclass Caller { String id = "demo:item/widget"; Target value; }\n',
        encoding="utf-8",
    )
    resource = resources / "widget.json"
    resource.write_text('{"parent":"minecraft:item/generated"}', encoding="utf-8")

    edges = derive_relations([root])
    triples = {(row["source"], row["target"], row["kind"]) for row in edges}
    assert (str(caller.resolve()), str(target.resolve()), "java_import") in triples
    assert (str(caller.resolve()), str(resource.resolve()), "resource_ref") in triples


def test_runtime_rag_index_contains_dependency_graph(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    java = project / "src/main/java/demo"
    java.mkdir(parents=True)
    (java / "Target.java").write_text(
        "package demo;\npublic final class Target {}\n",
        encoding="utf-8",
    )
    (java / "Caller.java").write_text(
        "package demo;\nimport demo.Target;\npublic final class Caller { Target target; }\n",
        encoding="utf-8",
    )
    service = ProductionToolService(workspace_root=workspace, profile="t4_local")
    service.index_project_rag(
        ["project"],
        metadata={
            "minecraft_version": "1.20.1",
            "loader": "fabric",
            "mapping_namespace": "yarn",
            "java_version": "17",
            "license": "project-local",
            "source_commit": "test-commit",
        },
        semantic=False,
    )
    index_path = workspace / "rag/project-index.json"
    with sqlite3.connect(index_path) as connection:
        rows = connection.execute(
            "SELECT source, target, kind FROM relations ORDER BY source, target"
        ).fetchall()
    assert any(row[2] == "java_import" for row in rows)

    result = service.search_code_rag(
        "Caller dependency Target",
        semantic=False,
        rerank=False,
    )
    assert result["retrieval_mode"] in {
        "lexical+relations",
        "semantic+rerank+relations",
        "lexical+rerank+relations",
        "caller-fallback",
    }
    assert "related dependency call chain" in result["expanded_query"]
    assert result["receipt"]["route"] == "multi_hop"


def test_context_compaction_preserves_exact_facts_and_recent_protocol(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MMM_SMALL_AGENT_CONTEXT_BYTES", str(24 * 1024))
    digest = "a" * 64
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "repair"},
    ]
    for index in range(7):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{index}", "type": "function"}],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"c{index}",
                "name": "search_code_rag",
                "content": json.dumps(
                    {
                        "ok": True,
                        "tool": "search_code_rag",
                        "result": {
                            "receipt": {
                                "route": "multi_hop",
                                "result_count": 4,
                                "relation_expansions": 2,
                            },
                            "payload": "x" * 6000,
                            "path": "src/main/java/demo/Caller.java",
                            "sha256": digest,
                        },
                    }
                ),
            }
        )
    compacted = compact_messages(messages)
    assert len(compacted) < len(messages)
    rendered = json.dumps(compacted)
    assert "HOST COMPACTED VERIFIED CONTEXT" in rendered
    assert "src/main/java/demo/Caller.java" in rendered
    assert digest in rendered
    assert compacted[-1]["role"] == "tool"
    assert compacted[-2]["role"] == "assistant"


def test_bootstrap_bound_context_compaction() -> None:
    assert getattr(
        model_router.ModelRouter._generate_with_tools,
        "_mmm_lossless_context_compaction",
        False,
    )
