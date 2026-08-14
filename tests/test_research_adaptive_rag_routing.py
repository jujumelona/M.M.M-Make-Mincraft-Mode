from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import minecraft_mod_ai.research_adaptive_rag_routing as adaptive


def _hash(value):
    raw = json.dumps(value, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _module(project_calls, code_calls, *, empty_catalog=False):
    def project(query, versions):
        project_calls.append((query, versions))
        return {
            "schema_version": "project",
            "sources": [] if empty_catalog else [{"source_id": query}],
            "errors": [],
        }

    def code(_path, query):
        code_calls.append(query)
        return {
            "schema_version": "code",
            "status": "searched",
            "hits": [{"source_path": query}],
        }

    return SimpleNamespace(
        _forced_rag_bundle=lambda *_args, **_kwargs: {},
        _research_versions=lambda _router: ("1.21.1",),
        _existing_code_index=lambda: Path("/tmp/project-rag.sqlite"),
        _search_authoritative_catalog=project,
        _search_code_index=code,
        _sha256_text=lambda value: "sha256:" + hashlib.sha256(value.encode()).hexdigest(),
        _sha256=_hash,
    )


def test_routes_retrievers_by_provider_and_evidence_kind():
    project_calls = []
    code_calls = []
    module = _module(project_calls, code_calls)
    small = SimpleNamespace(_RAG_ROUTER=ContextVar("test_router", default=None))
    adaptive.harden(module, small)

    brief = {
        "domains": [
            {
                "domain_id": "api",
                "providers": ["official_docs", "project_rag"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["Fabric registry API"],
            },
            {
                "domain_id": "local",
                "providers": ["project_rag"],
                "evidence_kinds": ["source_code"],
                "queries": ["existing registry implementation"],
            },
            {
                "domain_id": "visual",
                "providers": ["project_rag", "openverse_images"],
                "evidence_kinds": ["visual_reference"],
                "queries": ["visual reference"],
            },
        ]
    }

    result = module._forced_rag_bundle(object(), brief)

    assert [item[0] for item in project_calls] == ["Fabric registry API"]
    assert code_calls == ["existing registry implementation"]
    assert result["adaptive_routing"]["catalog_query_count"] == 1
    assert result["adaptive_routing"]["code_query_count"] == 1
    assert result["adaptive_routing"]["fully_skipped_query_count"] == 1


def test_catalog_miss_expands_to_code_rag():
    project_calls = []
    code_calls = []
    module = _module(project_calls, code_calls, empty_catalog=True)
    small = SimpleNamespace(_RAG_ROUTER=ContextVar("test_router_expand", default=None))
    adaptive.harden(module, small)

    brief = {
        "domains": [
            {
                "domain_id": "api",
                "providers": ["project_rag"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["unknown API"],
            }
        ]
    }

    result = module._forced_rag_bundle(object(), brief)
    query = result["domains"][0]["queries"][0]

    assert code_calls == ["unknown API"]
    assert query["retrieval_route"]["expansion"] == "code_on_catalog_miss"
    assert result["adaptive_routing"]["expanded_query_count"] == 1


def test_legacy_brief_keeps_historical_project_and_code_coverage():
    project_calls = []
    code_calls = []
    module = _module(project_calls, code_calls)
    small = SimpleNamespace(_RAG_ROUTER=ContextVar("test_router_legacy", default=None))
    adaptive.harden(module, small)

    result = module._forced_rag_bundle(
        object(),
        {"domains": [{"domain_id": "legacy", "queries": ["legacy query"]}]},
    )

    assert project_calls
    assert code_calls == ["legacy query"]
    route = result["domains"][0]["queries"][0]["retrieval_route"]
    assert route["reason"] == "legacy_brief_compatibility"
