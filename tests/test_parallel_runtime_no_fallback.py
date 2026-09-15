from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai.parallel_runtime_contract import (
    ParallelResearchContractError,
    _parallel_retrieve_domain_evidence_factory,
)


def _adapter() -> SimpleNamespace:
    return SimpleNamespace(
        minecraft_version="1.21.1",
        loader="fabric",
        yarn_mappings="1.21.1+build.3",
    )


def _domain(raw: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        domain_id=str(raw.get("domain_id", "implementation")),
        providers=tuple(raw.get("providers", ())),
        queries=tuple(raw.get("queries", ())),
        requirements=tuple(raw.get("requirements", ())),
        evidence_kinds=tuple(raw.get("evidence_kinds", ())),
        objective=str(raw.get("objective", "implementation evidence")),
    )


def _central_module() -> SimpleNamespace:
    return SimpleNamespace(
        retrieve_official_evidence=lambda *_args, **_kwargs: None,
        adapter_for_target=lambda _version, _loader: _adapter(),
        _research_domain=_domain,
        _lossless_query_pages=lambda text, _budget: [str(text)],
    )


def _target() -> dict[str, str]:
    return {
        "minecraft_version": "1.21.1",
        "loader": "fabric",
        "mappings": "1.21.1+build.3",
    }


def _official_domain(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "domain_id": "implementation",
        "providers": ["official_docs"],
        "queries": ["fabric implementation dependencies"],
        "requirements": ["fabric implementation dependencies"],
        "evidence_kinds": [],
        "objective": "implementation evidence",
    }
    value.update(overrides)
    return value


def test_missing_target_is_contract_failure_not_graph_fallback() -> None:
    central = _central_module()
    graph_calls: list[dict[str, Any]] = []

    def build_graph(brief: dict[str, Any], *, retrieve: Any) -> dict[str, Any]:
        graph_calls.append(brief)
        return {"domains": []}

    retrieve = _parallel_retrieve_domain_evidence_factory(central, build_graph)

    with pytest.raises(ParallelResearchContractError, match="_mmm_platform_target"):
        retrieve({"domains": [_official_domain()]})

    assert graph_calls == []


def test_bad_adapter_is_contract_failure_not_graph_fallback() -> None:
    central = _central_module()
    central.adapter_for_target = lambda _version, _loader: (_ for _ in ()).throw(
        ValueError("unsupported target")
    )
    graph_calls: list[dict[str, Any]] = []

    def build_graph(brief: dict[str, Any], *, retrieve: Any) -> dict[str, Any]:
        graph_calls.append(brief)
        return {"domains": []}

    retrieve = _parallel_retrieve_domain_evidence_factory(central, build_graph)

    with pytest.raises(ParallelResearchContractError, match="no platform adapter"):
        retrieve(
            {
                "_mmm_platform_target": _target(),
                "domains": [_official_domain()],
            }
        )

    assert graph_calls == []


def test_invalid_domain_is_contract_failure_not_graph_fallback() -> None:
    central = _central_module()
    central._research_domain = lambda _raw: (_ for _ in ()).throw(
        ValueError("invalid domain")
    )
    graph_calls: list[dict[str, Any]] = []

    def build_graph(brief: dict[str, Any], *, retrieve: Any) -> dict[str, Any]:
        graph_calls.append(brief)
        return {"domains": []}

    retrieve = _parallel_retrieve_domain_evidence_factory(central, build_graph)

    with pytest.raises(ParallelResearchContractError, match="invalid research domain"):
        retrieve(
            {
                "_mmm_platform_target": _target(),
                "domains": [_official_domain()],
            }
        )

    assert graph_calls == []


def test_official_domain_without_primary_query_is_contract_failure() -> None:
    central = _central_module()
    graph_calls: list[dict[str, Any]] = []

    def build_graph(brief: dict[str, Any], *, retrieve: Any) -> dict[str, Any]:
        graph_calls.append(brief)
        return {"domains": []}

    retrieve = _parallel_retrieve_domain_evidence_factory(central, build_graph)

    with pytest.raises(ParallelResearchContractError, match="no primary queries"):
        retrieve(
            {
                "_mmm_platform_target": _target(),
                "domains": [
                    _official_domain(
                        queries=[],
                        requirements=[],
                        evidence_kinds=[],
                    )
                ],
            }
        )

    assert graph_calls == []


def test_non_official_domain_uses_canonical_graph_path_once() -> None:
    central = _central_module()
    graph_calls: list[dict[str, Any]] = []

    def build_graph(brief: dict[str, Any], *, retrieve: Any) -> dict[str, Any]:
        graph_calls.append(brief)
        return {"domains": [], "route": "canonical"}

    retrieve = _parallel_retrieve_domain_evidence_factory(central, build_graph)
    result = retrieve(
        {
            "_mmm_platform_target": _target(),
            "domains": [
                {
                    "domain_id": "community",
                    "providers": ["github"],
                    "queries": ["example mod"],
                    "requirements": [],
                    "evidence_kinds": [],
                    "objective": "community evidence",
                }
            ],
        }
    )

    assert result["route"] == "canonical"
    assert len(graph_calls) == 1


def test_empty_official_result_stays_on_parallel_path_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_RESEARCH_WORKERS", "1")
    central = _central_module()
    graph_calls: list[dict[str, Any]] = []
    retrieval_calls: list[tuple[str, dict[str, Any]]] = []

    def empty_retrieve(query: str, **kwargs: Any) -> SimpleNamespace:
        retrieval_calls.append((query, kwargs))
        return SimpleNamespace(
            query=query,
            quality="weak",
            hits=(),
            coverage=0.0,
            correction_queries=(),
            correction_required=False,
        )

    def build_graph(brief: dict[str, Any], *, retrieve: Any) -> dict[str, Any]:
        graph_calls.append(brief)
        query = brief["domains"][0]["queries"][0]
        receipt = retrieve(
            query,
            minecraft_version="1.21.1",
            loader="fabric",
            mappings="1.21.1+build.3",
            limit=8,
        )
        return {
            "domains": [
                {
                    "domain_id": "implementation",
                    "queries": [
                        {
                            "primary": {
                                "query": query,
                                "quality": receipt.quality,
                                "hits": list(receipt.hits),
                                "coverage": receipt.coverage,
                            },
                            "corrections": [],
                        }
                    ],
                }
            ]
        }

    retrieve = _parallel_retrieve_domain_evidence_factory(central, build_graph)
    result = retrieve(
        {
            "_mmm_platform_target": _target(),
            "domains": [_official_domain()],
        },
        retrieve=empty_retrieve,
    )

    assert len(graph_calls) == 1
    assert len(retrieval_calls) == 1
    assert result["unresolved_official_domains"] == ["implementation"]
