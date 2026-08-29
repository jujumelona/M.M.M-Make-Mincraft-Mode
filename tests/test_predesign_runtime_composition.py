from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


_FRESH_RUNTIME_PROBE = r"""
from __future__ import annotations

import json
from types import SimpleNamespace

import minecraft_mod_ai.agentic_pre_design_rag as project_rag
import minecraft_mod_ai.agentic_research_game_design as agentic
import minecraft_mod_ai.pre_design_research_pipeline as pipeline

assert not hasattr(agentic, "collect_pre_design_research")

calls = {
    "official": 0,
    "radar": 0,
    "forced": 0,
    "domain": 0,
}


def official(brief):
    calls["official"] += 1
    return {
        "status": "available",
        "domains": [
            {
                "domain_id": domain["domain_id"],
                "queries": [
                    {"query": query, "query_sha256": "sha256:test"}
                    for query in domain.get("queries", [])
                ],
            }
            for domain in brief.get("domains", [])
            if isinstance(domain, dict)
        ],
    }


def radar(*_args, **_kwargs):
    calls["radar"] += 1
    return {"status": "available", "radar_sha256": "sha256:radar"}


def forced(_router, brief):
    calls["forced"] += 1
    return {
        "schema_version": "mmm/forced-pre-design-rag-v2",
        "domains": [
            {
                "domain_id": domain["domain_id"],
                "queries": [
                    {"query": query, "query_sha256": "sha256:test"}
                    for query in domain.get("queries", [])
                ],
            }
            for domain in brief.get("domains", [])
            if isinstance(domain, dict)
        ],
    }


def domain_worker(_router, *, prompt, domain, deterministic, trace_metadata):
    del prompt, trace_metadata
    calls["domain"] += 1
    assert set(deterministic) == {
        "official_rag",
        "technology_radar",
        "forced_project_rag",
    }
    return {
        "domain_id": domain["domain_id"],
        "claims": [{"claim": "validated request", "evidence_refs": ["probe"]}],
        "gaps": [],
        "next_queries": [],
        "sufficient": True,
        "procedures": [],
    }


pipeline.retrieve_domain_evidence = official
pipeline.collect_technology_radar = radar
project_rag._forced_rag_bundle = forced
agentic._research_domain_with_agent = domain_worker


class ProbeRouter:
    profile = "fresh-runtime-composition"
    registry = SimpleNamespace(
        role=lambda _profile, _role: SimpleNamespace(
            adapter="llama_cpp",
            model_id="probe-model",
            max_context=32768,
            max_input_tokens=0,
            max_new_tokens=2048,
        )
    )


result = pipeline.collect_design_research(
    ProbeRouter(),
    "Add one custom item.",
)

brief_ids = [
    item["domain_id"]
    for item in result["research_brief"]["domains"]
    if isinstance(item, dict)
]
coverage = result["minecraft_knowledge_route_coverage"]
coverage_statuses = {item["status"] for item in coverage["domains"]}

assert brief_ids == ["request"]
assert calls == {"official": 1, "radar": 1, "forced": 1, "domain": 1}
assert coverage["status"] == "PASS"
assert coverage["target_frozen"] is False
assert coverage_statuses == {"DEFERRED_UNTIL_TARGET_FREEZE"}
assert result["minecraft_knowledge_plan"]["policy"]["target_frozen"] is False
assert result["domain_notes"][0]["sufficient"] is True

print(
    "__MMM_RESULT__="
    + json.dumps(
        {
            "brief_ids": brief_ids,
            "calls": calls,
            "coverage_statuses": sorted(coverage_statuses),
            "target_frozen": coverage["target_frozen"],
        },
        sort_keys=True,
    )
)
"""


def test_fresh_runtime_uses_single_owner_and_defers_versioned_routes(
    tmp_path: Path,
) -> None:
    env = dict(os.environ)
    env["MMM_AGENT_WORKSPACE_ROOT"] = str(tmp_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_FRESH_RUNTIME_PROBE)],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        "fresh runtime composition probe failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result_line = next(
        (
            line.removeprefix("__MMM_RESULT__=")
            for line in completed.stdout.splitlines()
            if line.startswith("__MMM_RESULT__=")
        ),
        None,
    )
    assert result_line is not None, completed.stdout
    result = json.loads(result_line)
    assert result["brief_ids"] == ["request"]
    assert result["calls"] == {
        "domain": 1,
        "forced": 1,
        "official": 1,
        "radar": 1,
    }
    assert result["coverage_statuses"] == ["DEFERRED_UNTIL_TARGET_FREEZE"]
    assert result["target_frozen"] is False
