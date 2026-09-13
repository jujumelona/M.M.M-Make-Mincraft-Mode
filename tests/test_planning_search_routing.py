import json
import tomllib
from pathlib import Path

from minecraft_mod_ai.planning_candidate_evidence import (
    expansion_queries,
    global_grounded_pool,
    requirement_candidate_trace,
    semantic_frontier_pool,
)


def _requirement():
    return {
        "decision_type": "requirement",
        "requirement_id": "req_001",
        "semantic_capability": "spacecraft.upgrade",
        "statement": "Players can upgrade their spacecraft through trading.",
        "acceptance": ["Players can upgrade their spacecraft through trading."],
    }


def _grounded(records, query="spacecraft upgrade"):
    return {
        "queries": [
            {
                "query": query,
                "retrieval_queries": [query],
                "provider_receipts": {
                    "modrinth": {
                        "status": "available",
                        "search_requests": 1,
                        "result_count": len(records),
                    }
                },
                "evidence_records": [
                    {
                        "source_id": source_id,
                        "content": content,
                        "url": f"https://example.invalid/{source_id}",
                    }
                    for source_id, content in records
                ],
            }
        ]
    }


def _source_ids(pool):
    return {
        str(record.get("source_id") or "")
        for query in pool.get("queries", [])
        for record in query.get("evidence_records", [])
    }


def test_query_expansion_never_inherits_sibling_requirements():
    context = {
        "requirement": _requirement(),
        "sibling_requirements": [
            {
                "requirement_id": "req_002",
                "semantic_capability": "alien.combat",
                "statement": "Fight alien bosses on remote planets.",
            },
            {
                "requirement_id": "req_003",
                "semantic_capability": "planet.colonization",
                "statement": "Colonize planets and build settlements.",
            },
        ],
    }
    queries = expansion_queries(context)
    joined = " ".join(queries).casefold()
    assert "spacecraft" in joined
    assert "upgrade" in joined
    assert "alien" not in joined
    assert "combat" not in joined
    assert "colonization" not in joined
    assert "settlements" not in joined


def test_direct_requirement_candidate_can_enter_on_material_partial_match():
    pool = global_grounded_pool(
        {
            "r_001": _grounded(
                [
                    ("modrinth:direct", "Spacecraft documentation and examples."),
                    ("modrinth:noise", "Furniture decoration chairs and tables."),
                ]
            )
        }
    )
    trace = requirement_candidate_trace(_requirement(), pool)
    frontier = semantic_frontier_pool(_requirement(), pool, trace)
    assert _source_ids(frontier) == {"modrinth:direct"}


def test_task_cache_projects_to_requirement_local_semantic_frontier():
    unrelated = [
        (f"modrinth:unrelated-{index}", "Furniture decoration chairs and tables.")
        for index in range(128)
    ]
    pool = global_grounded_pool(
        {
            "r_001": _grounded(unrelated),
            "r_002": _grounded(
                [("modrinth:relevant", "Spacecraft upgrade through a trading terminal.")],
                query="space exploration",
            ),
        }
    )
    trace = requirement_candidate_trace(_requirement(), pool)
    frontier = semantic_frontier_pool(_requirement(), pool, trace)

    assert len(_source_ids(pool)) == 129
    assert _source_ids(frontier) == {"modrinth:relevant"}
    unresolved = {
        row["source_id"]
        for row in trace["candidates"]
        if row["status"] == "unresolved_relevance"
    }
    assert "modrinth:unrelated-0" in unresolved


def test_semantic_reviewer_never_uses_excluded_task_cache_sources(monkeypatch):
    from minecraft_mod_ai import planning_semantic_research as semantic

    unrelated = [
        (f"modrinth:unrelated-{index}", "Furniture decoration chairs and tables.")
        for index in range(128)
    ]
    pool = global_grounded_pool(
        {
            "r_001": _grounded(unrelated),
            "r_002": _grounded(
                [("modrinth:relevant", "Spacecraft upgrade through a trading terminal.")],
                query="space exploration",
            ),
        }
    )
    trace = requirement_candidate_trace(_requirement(), pool)
    seen_source_ids = []

    monkeypatch.setattr(semantic, "_semantic_request_budgets", lambda _config: (65536, 65536))
    monkeypatch.setattr(semantic, "router_native_model_parallelism", lambda _router: 1)

    def fake_generation(_router, _role, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        seen_source_ids.append(payload["source_id"])
        if kwargs.get("tool_name") == "assess_requirement_source":
            return {"verdict": "supported", "evidence_start": 0, "evidence_end": 0}
        return {"verdict": "supported"}

    monkeypatch.setattr(semantic, "generate_fixed_template_value", fake_generation)
    review = semantic.review_requirement_sources(object(), _requirement(), pool, trace)

    assert review["complete"] is True
    assert set(seen_source_ids) == {"modrinth:relevant"}
    assert not any(source_id.startswith("modrinth:unrelated-") for source_id in seen_source_ids)


def test_planning_mcp_stage_is_explicitly_configured_for_clients():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["mmm-planning"]
    assert server["env"]["MMM_MCP_STAGE"] == "planning"
    assert server["args"][-1] == "minecraft_mod_ai.mcp_server"

    codex = tomllib.loads((root / ".codex/config.toml").read_text(encoding="utf-8"))
    codex_server = codex["mcp_servers"]["mmm-planning"]
    assert codex_server["env"]["MMM_MCP_STAGE"] == "planning"
    assert codex_server["args"][-1] == "minecraft_mod_ai.mcp_server"


def test_adaptive_evidence_skill_uses_fixed_point_not_attempt_cap():
    root = Path(__file__).resolve().parents[1]
    skill = (root / "skills/gather-adaptive-minecraft-evidence/SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "- planning" in skill
    assert "max_attempts: null" in skill
    assert "no fresh admissible" in skill
    assert "Merge every requirement's candidates" in skill
