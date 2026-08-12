from types import SimpleNamespace

import minecraft_mod_ai.parallel_runtime_contract as parallel_module
from minecraft_mod_ai.parallel_target_rag_contract import (
    _target_parallel_retrieve_factory,
)


class _Receipt:
    def __init__(self, query: str, *, correction_queries=()):
        self.query = query
        self.correction_queries = tuple(correction_queries)
        self.correction_required = bool(self.correction_queries)
        self.hits = (object(),)

    def to_dict(self):
        return {
            "query": self.query,
            "correction_queries": list(self.correction_queries),
        }


def _central():
    def research_domain(raw):
        return SimpleNamespace(
            domain_id=raw["domain_id"],
            queries=tuple(raw["queries"]),
            providers=tuple(raw["providers"]),
        )

    return SimpleNamespace(
        SpecValidationError=ValueError,
        _research_domain=research_domain,
        _sha256=lambda value: "sha256:" + str(value),
        canonical_json=lambda value: repr(value),
        retrieve_official_evidence=None,
    )


def test_parallel_target_rag_preserves_platform_for_primary_and_corrections():
    central = _central()
    calls = []

    def retrieve(query, *, minecraft_version, loader, mappings, limit):
        calls.append(
            (query, minecraft_version, loader, mappings, limit)
        )
        if query == "registry api":
            return _Receipt(
                query,
                correction_queries=("registry exact symbol",),
            )
        return _Receipt(query)

    central.retrieve_official_evidence = retrieve

    def legacy_parallel(_brief, **_kwargs):
        raise AssertionError("targeted research must not use legacy 1.20.1 path")

    target_parallel = _target_parallel_retrieve_factory(
        central_module=central,
        parallel_module=parallel_module,
        legacy_parallel_retrieve=legacy_parallel,
    )
    brief = {
        "brief_sha256": "sha256:brief",
        "_mmm_platform_target": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "1.21.1+build.3",
        },
        "domains": [
            {
                "domain_id": "api",
                "queries": ["registry api", "item api"],
                "providers": ["official_docs"],
            }
        ],
    }

    payload = target_parallel(brief)

    assert payload["target"] == {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }
    assert {call[0] for call in calls} == {
        "registry api",
        "item api",
        "registry exact symbol",
    }
    assert all(call[1:4] == ("1.21.1", "fabric", "1.21.1+build.3") for call in calls)
    assert sorted(call[4] for call in calls) == [4, 8, 8]


def test_parallel_target_rag_keeps_legacy_behavior_without_target():
    central = _central()
    sentinel = {"status": "legacy"}

    def legacy_parallel(brief, **_kwargs):
        assert "_mmm_platform_target" not in brief
        return sentinel

    target_parallel = _target_parallel_retrieve_factory(
        central_module=central,
        parallel_module=parallel_module,
        legacy_parallel_retrieve=legacy_parallel,
    )

    assert target_parallel({"domains": []}) is sentinel
