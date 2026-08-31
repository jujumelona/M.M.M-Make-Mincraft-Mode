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
    "grounded": 0,
    "radar": 0,
    "domain": 0,
}


def grounded(_router, brief):
    calls["grounded"] += 1
    domains = []
    for domain in brief.get("domains", []):
        if not isinstance(domain, dict):
            continue
        queries = []
        for query in domain.get("queries", []):
            queries.append(
                {
                    "query": query,
                    "query_sha256": "sha256:test-query",
                    "code_rag": {
                        "documents": [
                            {
                                "source_id": "project:src/main/java/demo/ItemRegistration.java",
                                "url": "file:///workspace/src/main/java/demo/ItemRegistration.java",
                                "content": (
                                    "Claim-bearing local source implementation evidence "
                                    "for target-neutral item registration planning."
                                ),
                            }
                        ]
                    },
                }
            )
        domains.append(
            {
                "domain_id": domain["domain_id"],
                "queries": queries,
            }
        )
    return {
        "schema_version": "mmm/forced-pre-design-rag",
        "research_sha256": "sha256:grounded-fixture",
        "domains": domains,
    }


def radar(*_args, **_kwargs):
    calls["radar"] += 1
    raise AssertionError("target-specific radar ran before target freeze")


def domain_worker(_agentic, _project_rag, _router, *, prompt, domain, document, trace_metadata):
    del prompt, trace_metadata
    calls["domain"] += 1
    assert set(document["source_keys"]) == {"grounded_rag"}
    pages = project_rag._read_evidence_pages(document)
    assert pages
    return {
        "domain_id": domain["domain_id"],
        "claims": [
            {
                "claim": "validated request",
                "evidence_refs": [pages[0]["page_ref"]],
            }
        ],
        "gaps": [],
        "next_queries": [],
        "sufficient": True,
        "fixed_point": False,
        "procedures": [],
        "checkpoint": {"status": "complete"},
    }


project_rag._forced_rag_bundle = grounded
pipeline.collect_technology_radar = radar
pipeline.research_document_domain = domain_worker


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
assert calls == {"grounded": 1, "radar": 0, "domain": 1}
assert set(result["deterministic"]) == {"grounded_rag", "technology_radar"}
assert result["deterministic"]["grounded_rag"]["status"] == "available"
assert result["deterministic"]["technology_radar"]["status"] == "deferred_until_target_freeze"
assert coverage["status"] == "PASS"
assert coverage["target_frozen"] is False
assert coverage_statuses == {"DEFERRED_UNTIL_TARGET_FREEZE"}
assert result["minecraft_knowledge_plan"]["policy"]["target_frozen"] is False
assert result["domain_notes"][0]["sufficient"] is True
assert result["domain_notes"][0]["claims"][0]["evidence_refs"][0].startswith("sha256:")
assert "#page=" in result["domain_notes"][0]["claims"][0]["evidence_refs"][0]

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
        "grounded": 1,
        "radar": 0,
    }
    assert result["coverage_statuses"] == ["DEFERRED_UNTIL_TARGET_FREEZE"]
    assert result["target_frozen"] is False
